"""Operations for Acly's krita-ai-diffusion plugin (the "AI Image Generation" docker).

These drive the plugin's own per-document model -- the same object its docker
shows -- so every change is visible in the docker, and generation goes through
the plugin's normal preparation: selection masks, regions, control and
reference layers, styles. Nothing here builds a ComfyUI workflow itself.

The plugin has no public API, so this reaches into its modules. Every entry
point used is listed in `_plugin()` and checked there, so an incompatible
plugin update fails with one clear error instead of half-applied changes.

Like everything in ops.py, these run on Krita's GUI thread and must return
quickly: generation is only *started* here. The MCP server waits for jobs by
polling `ai_jobs`.
"""

import sys
import time

from .compat import QImage
from . import imaging
from .ops import (
    OpError,
    _arg,
    _as_bool,
    _as_float,
    _as_int,
    _krita,
    _node_uuid,
    _req,
    op,
    resolve_document,
    resolve_node,
)


# --------------------------------------------------------------------------
# plugin access
# --------------------------------------------------------------------------

class _Plugin(object):
    """The krita-ai-diffusion modules this file depends on."""

    def __init__(self):
        mods = sys.modules
        try:
            self.package = mods["ai_diffusion"]
            self.root = mods["ai_diffusion.model.root"].root
            model = mods["ai_diffusion.model.model"]
            self.Workspace = model.Workspace
            self.QueueMode = model.QueueMode
            self.JobKind = mods["ai_diffusion.model.jobs"].JobKind
            api = mods["ai_diffusion.backend.api"]
            self.InpaintMode = api.InpaintMode
            self.InpaintContext = api.InpaintContext
            self.FillMode = api.FillMode
            self.ControlMode = mods["ai_diffusion.backend.resources"].ControlMode
            self.Styles = mods["ai_diffusion.style"].Styles
            client = mods["ai_diffusion.backend.client"]
            self.resolve_arch = client.resolve_arch
            self.is_style_supported = client.is_style_supported
            settings = mods["ai_diffusion.settings"]
            self.settings = settings.settings
            self.ApplyBehavior = settings.ApplyBehavior
            self.ApplyRegionBehavior = settings.ApplyRegionBehavior
            self.ConnectionState = mods["ai_diffusion.model.connection"].ConnectionState
        except (KeyError, AttributeError) as exc:
            raise OpError(
                "The AI Image Generation plugin (krita-ai-diffusion) is not "
                "loaded, or this version is not supported ({0}). Enable it in "
                "Settings > Configure Krita > Python Plugin Manager."
                .format(exc), kind="ai_plugin_unavailable")
        self.version = getattr(self.package, "__version__", "unknown")

    @property
    def client(self):
        return self.root.connection.client_if_connected


def _plugin():
    return _Plugin()


def _model(p, params):
    """The plugin's model for the requested document, made active if needed.

    The plugin always works on the active document, so a document other than
    the active one is brought to the front first.
    """
    doc = resolve_document(params.get("document"))
    krita = _krita()
    if krita.activeDocument() != doc:
        window = krita.activeWindow()
        view = next((v for v in (window.views() if window else [])
                     if v.document() == doc), None)
        if view is None:
            raise OpError("That document has no open view to activate.",
                          kind="not_found")
        window.setActiveView(view)
    model = p.root.model_for_active_document()
    if model is None:
        raise OpError("The AI plugin has no model for the active document.",
                      kind="no_document")
    return model


def _enum(enum_cls, value, key):
    if value is None:
        return None
    try:
        return enum_cls[str(value)]
    except KeyError:
        raise OpError("{0} must be one of: {1}".format(
            key, ", ".join(e.name for e in enum_cls)))


def _layer(model, doc_ref, ref):
    """Map a layer reference (name, path, index or uuid) to the plugin's Layer."""
    doc = resolve_document(doc_ref)
    node = resolve_node(doc, ref)
    layer = model.layers.updated().find(node.uniqueId())
    if layer is None:
        raise OpError("The AI plugin does not track layer {0!r}.".format(ref),
                      kind="not_found")
    return layer


def _find_style(p, ref):
    styles = p.Styles.list()
    style = styles.find(ref)
    if style is None:
        lowered = str(ref).lower()
        hits = [s for s in styles if s.name.lower() == lowered
                or s.filename.lower().endswith("/" + lowered)
                or s.filename.lower().endswith("/" + lowered + ".json")]
        style = hits[0] if len(hits) == 1 else None
    if style is None:
        raise OpError("No style matches {0!r}. Call ai_list_styles.".format(ref),
                      kind="not_found")
    if not p.is_style_supported(style, p.client):
        raise OpError("Style {0!r} needs models the ComfyUI server does not "
                      "have.".format(style.name), kind="unsupported")
    return style


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _style_entry(p, style):
    arch = p.resolve_arch(style, p.client)
    return {"file": style.filename, "name": style.name, "arch": arch.name,
            "edits_images": bool(arch.supports_edit)}


def _layer_names(layers):
    return [l.name for l in layers]


def _control_entry(control):
    try:
        layer = control.layer.name
    except AssertionError:
        layer = None
    return {"mode": control.mode.name, "layer": layer,
            "strength": control.strength / control.strength_multiplier,
            "start": control.start, "end": control.end,
            "supported": control.is_supported,
            "error": control.error_text or None}


def _job_entry(job):
    params = job.params
    bounds = params.bounds
    return {"id": job.id, "kind": job.kind.name,
            "state": job.state.name.lower() if job.state.name else str(job.state),
            "prompt": params.prompt, "seed": params.seed,
            "strength": params.strength,
            "bounds": {"x": bounds.x, "y": bounds.y,
                       "width": bounds.width, "height": bounds.height},
            "results": len(job.results),
            "used": sorted(i for i, used in job.in_use.items() if used)}


def _status(p, model):
    regions = model.active_regions
    doc = model.document
    selection = doc.selection_bounds
    edit_style = model.edit_style
    return {
        "plugin_version": p.version,
        "connection": p.root.connection.state.name,
        "server": p.settings.server_url,
        "document": doc.filename or "(not saved)",
        "workspace": model.workspace.name,
        "style": _style_entry(p, model.style),
        "edit_mode": model.edit_mode,
        "edit_style": _style_entry(p, edit_style) if edit_style else None,
        "prompt": regions.positive,
        "negative": regions.negative,
        "strength": model.strength,
        "seed": model.seed,
        "fixed_seed": model.fixed_seed,
        "batch_count": model.batch_count,
        "region_only": model.region_only,
        "resolution_multiplier": model.resolution_multiplier,
        "inpaint": {
            "mode": model.inpaint.mode.name,
            "fill": model.inpaint.fill.name,
            "use_inpaint_model": model.inpaint.use_inpaint,
            "use_prompt_focus": model.inpaint.use_prompt_focus,
            "context": model.inpaint.context.name,
        },
        "selection": ({"x": selection.x, "y": selection.y,
                       "width": selection.width, "height": selection.height}
                      if selection else None),
        "controls": [_control_entry(c) for c in regions.control],
        "regions": [{"layers": _layer_names(r.layers), "prompt": r.positive,
                     "controls": [_control_entry(c) for c in r.control]}
                    for r in regions],
        "jobs": [_job_entry(j) for j in list(model.jobs)[-8:]],
        "progress": model.progress,
        "error": model.error.message or None,
        "apply_behavior": p.settings.apply_behavior.name,
        "generation_finished_action": p.settings.generation_finished_action.name,
    }


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------

@op("ai_status", timeout=20.0)
def op_ai_status(params):
    p = _plugin()
    return _status(p, _model(p, params))


@op("ai_list_styles", timeout=20.0)
def op_ai_list_styles(params):
    p = _plugin()
    model = _model(p, params)
    include_all = _as_bool(_arg(params, "include_unsupported", False),
                           "include_unsupported")
    out = []
    for style in p.Styles.list():
        supported = p.is_style_supported(style, p.client)
        if not supported and not include_all:
            continue
        entry = _style_entry(p, style)
        entry["supported"] = supported
        entry["current"] = style == model.style
        out.append(entry)
    return {"styles": out}


_SETTERS = ("workspace", "style", "prompt", "negative", "strength", "seed",
            "fixed_seed", "batch_count", "edit_mode", "region_only",
            "resolution_multiplier", "inpaint_mode", "inpaint_fill",
            "use_inpaint_model", "use_prompt_focus", "inpaint_context",
            "inpaint_context_layer")


@op("ai_configure", timeout=20.0, mutates=True)
def op_ai_configure(params):
    p = _plugin()
    model = _model(p, params)
    unknown = set(params) - set(_SETTERS) - {"document"}
    if unknown:
        raise OpError("Unknown settings: {0}. Known: {1}".format(
            ", ".join(sorted(unknown)), ", ".join(_SETTERS)))

    # Validate everything before changing anything.
    workspace = _enum(p.Workspace, params.get("workspace"), "workspace")
    style = _find_style(p, params["style"]) if "style" in params else None
    inpaint_mode = _enum(p.InpaintMode, params.get("inpaint_mode"), "inpaint_mode")
    fill = _enum(p.FillMode, params.get("inpaint_fill"), "inpaint_fill")
    context = _enum(p.InpaintContext, params.get("inpaint_context"),
                    "inpaint_context")
    context_layer = None
    if params.get("inpaint_context_layer") is not None:
        context_layer = _layer(model, params.get("document"),
                               params["inpaint_context_layer"])
    strength = None
    if "strength" in params:
        strength = _as_float(params["strength"], "strength")
        if not 0.0 < strength <= 1.0:
            raise OpError("strength must be in (0, 1]")

    if workspace is not None:
        model.workspace = workspace
    if style is not None:
        model.style = style
    regions = model.active_regions
    if "prompt" in params:
        regions.positive = str(params["prompt"])
    if "negative" in params:
        regions.negative = str(params["negative"])
    if strength is not None:
        model.strength = strength
    if "seed" in params:
        model.seed = _as_int(params["seed"], "seed")
    if "fixed_seed" in params:
        model.fixed_seed = _as_bool(params["fixed_seed"], "fixed_seed")
    if "batch_count" in params:
        model.batch_count = max(1, _as_int(params["batch_count"], "batch_count"))
    if "edit_mode" in params:
        edit = _as_bool(params["edit_mode"], "edit_mode")
        if edit and not model.can_edit:
            raise OpError("The current style has no edit model available.",
                          kind="unsupported")
        model.edit_mode = edit
    if "region_only" in params:
        model.region_only = _as_bool(params["region_only"], "region_only")
    if "resolution_multiplier" in params:
        model.resolution_multiplier = _as_float(params["resolution_multiplier"],
                                                "resolution_multiplier")
    if inpaint_mode is not None:
        model.inpaint.mode = inpaint_mode
    if fill is not None:
        model.inpaint.fill = fill
    if "use_inpaint_model" in params:
        model.inpaint.use_inpaint = _as_bool(params["use_inpaint_model"],
                                             "use_inpaint_model")
    if "use_prompt_focus" in params:
        model.inpaint.use_prompt_focus = _as_bool(params["use_prompt_focus"],
                                                  "use_prompt_focus")
    if context is not None:
        model.inpaint.context = context
    if context_layer is not None:
        model.inpaint.context_layer_id = context_layer.id
    return _status(p, model)


@op("ai_set_region", timeout=20.0, mutates=True)
def op_ai_set_region(params):
    """Link a regional prompt to a layer (its painted pixels define the area)."""
    p = _plugin()
    model = _model(p, params)
    regions = model.active_regions
    layer = _layer(model, params.get("document"), _req(params, "layer"))
    region = regions.find_linked(layer)

    if _as_bool(_arg(params, "remove", False), "remove"):
        if region is None:
            raise OpError("No region is linked to that layer.", kind="not_found")
        region.remove()
        return _status(p, model)

    if region is None:
        region = regions._add(layer)
    if "prompt" in params:
        region.positive = str(params["prompt"])
    return _status(p, model)


@op("ai_set_control", timeout=20.0, mutates=True)
def op_ai_set_control(params):
    """Add, change or remove a control/reference layer on the root or a region."""
    p = _plugin()
    model = _model(p, params)
    regions = model.active_regions
    doc_ref = params.get("document")
    target = regions
    if params.get("region") is not None:
        target = regions.find_linked(_layer(model, doc_ref, params["region"]))
        if target is None:
            raise OpError("No region is linked to that layer.", kind="not_found")

    layer = _layer(model, doc_ref, _req(params, "layer"))
    controls = target.control
    existing = next((c for c in controls if c.layer_id == layer.id), None)

    if _as_bool(_arg(params, "remove", False), "remove"):
        if existing is None:
            raise OpError("That layer is not a control layer here.",
                          kind="not_found")
        controls.remove(existing)
        return _status(p, model)

    mode = _enum(p.ControlMode, _arg(params, "mode", "reference"), "mode")
    control = existing or controls.emplace()
    control.layer_id = layer.id
    control.mode = mode
    if "strength" in params:
        strength = _as_float(params["strength"], "strength")
        if not 0.0 <= strength <= 1.5:
            raise OpError("strength must be between 0 and 1.5")
        control.use_custom_strength = True
        control.strength = int(round(strength * control.strength_multiplier))
    if "start" in params:
        control.start = _as_float(params["start"], "start")
    if "end" in params:
        control.end = _as_float(params["end"], "end")
    return _status(p, model)


@op("ai_generate", timeout=30.0, mutates=True)
def op_ai_generate(params):
    """Start generation with the current settings, exactly like the Generate button.

    Uses the active selection as the inpaint area if there is one, else the
    active region, else the whole canvas. Returns the ids of jobs that existed
    before, so the caller can tell which new jobs this request created.
    """
    p = _plugin()
    model = _model(p, params)
    if p.root.connection.state is not p.ConnectionState.connected:
        raise OpError("The AI plugin is not connected to its ComfyUI server.",
                      kind="not_connected")
    if model.workspace not in (p.Workspace.generation, p.Workspace.upscaling):
        raise OpError("ai_generate supports the generation and upscaling "
                      "workspaces; the current one is {0}."
                      .format(model.workspace.name), kind="unsupported")
    before = [j.id for j in model.jobs]
    model.clear_error()
    if model.workspace is p.Workspace.upscaling:
        model.upscale_image()
    else:
        model.generate()
    if model.error and not model.error.kind.is_warning:
        raise OpError("The AI plugin refused: {0}".format(model.error.message),
                      kind="ai_error")
    return {"existing_job_ids": before, "expected_jobs": model.batch_count,
            "workspace": model.workspace.name}


@op("ai_jobs", timeout=20.0)
def op_ai_jobs(params):
    p = _plugin()
    model = _model(p, params)
    return {"jobs": [_job_entry(j) for j in model.jobs],
            "progress": model.progress,
            "error": model.error.message or None}


def _job(model, job_id):
    job = model.jobs.find(job_id)
    if job is None:
        raise OpError("No job {0!r} in the AI plugin's history.".format(job_id),
                      kind="not_found")
    return job


@op("ai_get_result", timeout=30.0)
def op_ai_get_result(params):
    p = _plugin()
    model = _model(p, params)
    job = _job(model, _req(params, "job"))
    index = _as_int(_arg(params, "index", 0), "index")
    if not 0 <= index < len(job.results):
        raise OpError("Job has {0} result(s).".format(len(job.results)),
                      kind="not_found")
    image = job.results[index]._qimage
    max_size = max(0, min(4096, _as_int(_arg(params, "max_size", 1024), "max_size")))
    scaled, factor = imaging.scale_to_fit(QImage(image), max_size)
    result = _job_entry(job)
    result.update({"index": index, "scale": round(factor, 6),
                   "returned_size": {"width": scaled.width(),
                                     "height": scaled.height()}})
    if _as_bool(_arg(params, "include_data", True), "include_data"):
        result["png_base64"] = imaging.qimage_to_png_b64(scaled)
    return result


@op("ai_apply", timeout=30.0, mutates=True)
def op_ai_apply(params):
    """Put a result on the canvas as a new layer (or replace the active layer)."""
    p = _plugin()
    model = _model(p, params)
    job = _job(model, _req(params, "job"))
    index = _as_int(_arg(params, "index", 0), "index")
    if not 0 <= index < len(job.results):
        raise OpError("Job has {0} result(s).".format(len(job.results)),
                      kind="not_found")
    behavior = params.get("behavior")
    if behavior is None:
        model.apply_generated_result(job.id, index)
    else:
        behavior = _enum(p.ApplyBehavior, behavior, "behavior")
        model.apply_result(job.results[index], job.params, behavior,
                           p.settings.apply_region_behavior, "[Generated] ")
        model.hide_preview(delete_layer=True)
        model.jobs.selection = []
        model.jobs.notify_used(job.id, index)
    active = model.layers.active
    return {"applied": True, "job": job.id, "index": index,
            "active_layer": active.name,
            "active_layer_id": _node_uuid(active.node)}


@op("ai_discard", timeout=20.0, mutates=True)
def op_ai_discard(params):
    """Remove the on-canvas preview layer, and optionally a job's results."""
    p = _plugin()
    model = _model(p, params)
    model.jobs.selection = []
    model.hide_preview(delete_layer=True)
    job_id = params.get("job")
    if job_id is not None:
        job = _job(model, job_id)
        for i in reversed(range(len(job.results))):
            model.jobs.discard(job.id, i)
    return {"discarded": True}


@op("ai_cancel", timeout=20.0, mutates=True)
def op_ai_cancel(params):
    p = _plugin()
    model = _model(p, params)
    model.cancel(active=_as_bool(_arg(params, "active", True), "active"),
                 queued=_as_bool(_arg(params, "queued", True), "queued"))
    return {"cancelled": True}


# --------------------------------------------------------------------------
# document teardown
# --------------------------------------------------------------------------

def settle_pending_saves(doc, budget=3.0):
    """Wait for the AI plugin to finish writing its state into ``doc``.

    The plugin stores its docker state in the document's annotations about a
    second after any change, from an asyncio task that does not check whether
    the document still exists. Closing the document before that task runs
    crashes Krita (segfault in KisDocument::image() via setAnnotation;
    reproduced on 5.3.4 with plugin 1.53.0). Does nothing when the plugin is
    not loaded.
    """
    from .compat import QCoreApplication, QEventLoop

    root_module = sys.modules.get("ai_diffusion.model.root")
    if root_module is None:
        return
    syncs = [entry.sync for entry in getattr(root_module.root, "_models", [])
             if entry.sync is not None
             and getattr(entry.model.document, "_doc", None) == doc]

    def pending():
        return [t for s in syncs
                for t in (getattr(s, "_save_task", None),
                          getattr(s, "_image_task", None))
                if t is not None and not t.done()]

    app = QCoreApplication.instance()
    deadline = time.time() + budget
    while pending() and app is not None and time.time() < deadline:
        app.processEvents(QEventLoop.ExcludeUserInputEvents, 50)
        time.sleep(0.02)
