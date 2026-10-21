"""Execute the one-time diagnostic on retained research artifacts, CPU only."""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TOOLS = "/opt/escalation/diagnose_motion.py"
image = modal.Image.from_id("im-yvwuQXEyvg2yH1dd46r6k0").add_local_file(
    HERE / "diagnose_motion.py", TOOLS, copy=True)
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


@app.function(image=image, volumes={"/evaluation": volume}, cpu=(2, 2), memory=(4096, 4096),
              timeout=600, retries=0, max_containers=1, scaledown_window=2)
def analyze(stage: str, image_id: str):
    if stage == "reference":
        manifests = [Path("/evaluation/unimate-20261021-escalation-a/tiger-baseline/manifest.json")]
        manifests += sorted(Path("/evaluation/unimate-20261021-upstream-fox-a/prompts").glob("*/v3/*/manifest.json"))
        manifests += sorted(Path("/evaluation/unimate-20261021-upstream-go2-b/gpu/candidate-v3").glob("*/manifest.json"))
    elif stage in ("compare-1729", "compare-2718", "compare-31415"):
        seed = stage.split("-")[-1]
        manifests = sorted(Path("/evaluation/unimate-20261021-escalation-b/samples").glob(f"*/*/{seed}/manifest.json"))
        if len(manifests) != 4:
            raise ValueError("Comparison has not finished all four model/rig pairs")
    elif stage == "solver-tiger-euler200":
        root = Path("/evaluation/unimate-20261021-escalation-d/tiger-euler200")
        if json.loads((root / "receipt.json").read_text())["status"] != "passed":
            raise ValueError("Solver experiment has not finished")
        manifests = [root / "motion_expand/manifest.json"]
    elif stage.startswith("expand-"):
        case, seed = stage.removeprefix("expand-").rsplit("-", 1)
        cases = {"tiger", "artist-v3-walk", "artist-v3-trot", "artist-truebones-walk",
                 "artist-truebones-trot", "native-truebones-walk"}
        if case not in cases or seed not in {"0", "1729", "2718", "31415"} or (case == "tiger") != (seed == "0"):
            raise ValueError("Outside the declared expansion cases")
        root = Path("/evaluation/unimate-20261021-escalation-c") / case / seed
        if json.loads((root / "receipt.json").read_text())["status"] != "passed":
            raise ValueError("Expansion has not passed its generation checks")
        manifests = [root / "motion_expand/manifest.json"]
        if not manifests[0].is_file():
            raise FileNotFoundError(manifests[0])
    else:
        raise ValueError("Unknown bounded analysis stage")
    output = Path("/evaluation/unimate-20261021-escalation-analysis") / stage
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"stage": stage, "image": image_id, "status": "running", "results": []}
    try:
        for index, manifest in enumerate(manifests):
            destination = output / f"{index:02d}"
            command = ["/opt/conda/bin/python", TOOLS, "--manifest", str(manifest), "--output", str(destination)]
            if index == 0:
                command.append("--plots")
            with (output / f"{index:02d}.log").open("w") as log:
                subprocess.run(command, check=True, cwd="/opt/unimate-evaluation/UniMate",
                    env={**os.environ, "PYTHONPATH": "/opt/unimate-evaluation/UniMate:/opt/unimate-evaluation/Motion"},
                    stdout=log, stderr=subprocess.STDOUT)
            receipt["results"].append({"manifest": str(manifest), "report": str(destination / "report.json")})
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return receipt


@app.local_entrypoint()
def main(stage: str = "reference"):
    print(json.dumps(analyze.remote(stage, image.object_id)), flush=True)
