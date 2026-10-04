#!/usr/bin/env python3
"""Integration test for the ai_* tools (Acly's krita-ai-diffusion plugin).

Drives mcp_server.py over real JSON-RPC stdio against a live Krita that has
both the MCP Bridge and the AI Image Generation plugin enabled, and the AI
plugin connected to its ComfyUI server.

    python test_ai.py              # settings, regions, controls (no rendering)
    python test_ai.py --generate   # also render one small image (ComfyUI time)
    python test_ai.py -v           # print each tool result

It works in its own new document and closes it at the end.
"""

import sys
import time

from test_mcp import FAIL, PASS, Client, _image_of, _text_of, check, section

GENERATE = "--generate" in sys.argv
DOC = "ai-tools-test-{0}".format(int(time.time()))


def main():
    client = Client()
    try:
        client.request("initialize", {"protocolVersion": "2025-06-18",
                                      "capabilities": {},
                                      "clientInfo": {"name": "test_ai"}})
        client.notify("notifications/initialized")

        section("tools")
        tools = client.request("tools/list")["result"]["tools"]
        names = {t["name"] for t in tools}
        expected = {"ai_status", "ai_list_styles", "ai_configure",
                    "ai_set_region", "ai_set_control", "ai_generate",
                    "ai_wait", "ai_jobs", "ai_get_result", "ai_apply",
                    "ai_discard", "ai_cancel"}
        check("all ai tools listed", expected <= names, sorted(expected - names))

        section("status")
        client.ok("create_document", {"width": 512, "height": 384, "name": DOC})
        deadline = time.time() + 90
        status, _ = client.ok("ai_status")
        while status["connection"] == "connecting" and time.time() < deadline:
            time.sleep(1)
            status, _ = client.ok("ai_status")
        check("plugin connected", status["connection"] == "connected",
              status["connection"])
        styles, _ = client.ok("ai_list_styles")
        supported = [s for s in styles["styles"] if s["supported"]]
        check("at least one usable style", len(supported) > 0)
        generator = next((s for s in supported if not s["edits_images"]), None)

        section("configure")
        before, _ = client.ok("ai_status")
        after, _ = client.ok("ai_configure", {
            "prompt": "test prompt", "negative": "blurry", "strength": 0.4,
            "inpaint_mode": "custom", "inpaint_context": "entire_image",
            "batch_count": 2})
        check("prompt set", after["prompt"] == "test prompt")
        check("negative set", after["negative"] == "blurry")
        check("strength set", abs(after["strength"] - 0.4) < 1e-6)
        check("inpaint mode set", after["inpaint"]["mode"] == "custom")
        check("inpaint context set",
              after["inpaint"]["context"] == "entire_image")
        check("batch count set", after["batch_count"] == 2)

        bad = client.call("ai_configure", {"prompt": "must not stick",
                                           "inpaint_mode": "nonsense"})
        check("invalid value rejected", bad.get("isError") is True,
              _text_of(bad)[:160])
        unchanged, _ = client.ok("ai_status")
        check("nothing changed by a rejected request",
              unchanged["prompt"] == "test prompt", unchanged["prompt"])

        if generator:
            styled, _ = client.ok("ai_configure", {"style": generator["file"]})
            check("style set by file", styled["style"]["file"] == generator["file"])
        client.ok("ai_configure", {
            "prompt": before["prompt"], "negative": before["negative"],
            "strength": before["strength"], "batch_count": 1,
            "inpaint_mode": before["inpaint"]["mode"],
            "inpaint_context": before["inpaint"]["context"]})

        section("regions")
        client.ok("create_layer", {"name": "Region A"})
        client.ok("draw", {"layer": "Region A", "commands": [
            {"type": "fill_rect", "x": 20, "y": 20, "w": 200, "h": 200,
             "color": "#ffffff"}]})
        linked, _ = client.ok("ai_set_region", {"layer": "Region A",
                                                "prompt": "a red apple"})
        region = next((r for r in linked["regions"]
                       if "Region A" in r["layers"]), None)
        check("region linked with prompt",
              region is not None and region["prompt"] == "a red apple",
              str(linked["regions"]))
        removed, _ = client.ok("ai_set_region", {"layer": "Region A",
                                                 "remove": True})
        check("region removed",
              not any("Region A" in r["layers"] for r in removed["regions"]))

        section("controls")
        added, _ = client.ok("ai_set_control", {"layer": "Region A",
                                                "mode": "scribble",
                                                "strength": 0.8})
        control = next((c for c in added["controls"]
                        if c["layer"] == "Region A"), None)
        check("control layer added",
              control is not None and control["mode"] == "scribble"
              and abs(control["strength"] - 0.8) < 1e-6, str(added["controls"]))
        cleared, _ = client.ok("ai_set_control", {"layer": "Region A",
                                                  "remove": True})
        check("control layer removed",
              not any(c["layer"] == "Region A" for c in cleared["controls"]))
        client.ok("delete_layer", {"layer": "Region A"})

        if GENERATE and generator:
            section("generate")
            result = client.call("ai_generate", {
                "style": generator["file"],
                "prompt": "a single red apple on a white table",
                "priority": True, "wait_seconds": 1500, "max_size": 256},
                timeout=1600)
            check("generate succeeded", not result.get("isError"),
                  _text_of(result)[:300])
            check("result image returned", _image_of(result) is not None)
            jobs, _ = client.ok("ai_jobs")
            if not jobs["jobs"]:
                raise AssertionError("no AI job was created")
            job = jobs["jobs"][-1]
            check("job finished with a result",
                  job["state"] == "finished" and job["results"] == 1, str(job))
            single = client.call("ai_get_result", {"job": job["id"],
                                                   "max_size": 128})
            check("ai_get_result returns an image",
                  _image_of(single) is not None)
            applied, _ = client.ok("ai_apply", {"job": job["id"]})
            check("result applied as a layer",
                  applied["active_layer"].startswith("[Generated]"),
                  applied["active_layer"])
            info, _ = client.ok("inspect_document")
            check("no preview layer left behind",
                  not any(l["name"].startswith("[Preview]")
                          for l in info["layers"]),
                  str([l["name"] for l in info["layers"]]))

        section("cleanup")
        client.ok("close_document", {"document": DOC, "discard_changes": True})
    except Exception as exc:
        import traceback
        FAIL.append(("harness", "{0}: {1}".format(type(exc).__name__, exc)))
        print("\nUNEXPECTED: {0}".format(traceback.format_exc()))
    finally:
        client.close()

    print("\n" + "=" * 60)
    print("{0} passed, {1} failed".format(len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAIL {0}: {1}".format(name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
