"""Ephemeral, serial UniMate research; no deployment or production mutation."""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INPUTS = Path(os.environ.get("UNIMATE_ESCALATION_INPUTS", "/tmp/tin88-lanes/unimate-escalation/inputs"))
RUN = "unimate-20261021-escalation-a"
OUT = Path("/evaluation") / RUN
SOURCE = Path("/opt/unimate-evaluation/UniMate")
PYTHON = "/opt/conda/bin/python"
image = modal.Image.from_id("im-Jaq993BhpAbZIsEYvhjwLQ").add_local_dir(
    HERE, "/opt/escalation/tools", ignore=["__pycache__", "*.pyc"], copy=True
).add_local_dir(INPUTS, "/opt/escalation/tiger", copy=True)
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


def execute(stage, image_id):
    import hashlib

    OUT.mkdir(parents=True, exist_ok=True)
    if stage == "baseline":
        prerequisite = json.loads((OUT / "preflight-receipt.json").read_text())
        if prerequisite["status"] != "passed" or prerequisite["image"] != image_id:
            raise ValueError("Baseline requires the same image that passed CPU preflight")
    receipt = {"stage": stage, "image": image_id,
               "startedUnix": time.time(), "status": "running"}
    path = OUT / (stage + "-receipt.json")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an earlier run: {path}")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    volume.commit()
    command = [PYTHON, "/opt/escalation/tools/worker.py", stage, str(OUT)]
    try:
        with (OUT / (stage + ".log")).open("w") as log:
            completed = subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONPATH": str(SOURCE) + ":/opt/unimate-evaluation/Motion"})
        receipt.update(status="passed" if completed.returncode == 0 else "failed", exitCode=completed.returncode)
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.time() - receipt["startedUnix"]
        receipt["command"] = command
        receipt["files"] = [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in sorted(OUT.rglob("*")) if p.is_file() and p != path]
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    if receipt["status"] != "passed":
        raise RuntimeError(f"{stage} failed; inspect persisted log and receipt")
    return {"stage": stage, "status": receipt["status"], "seconds": receipt["seconds"], "receipt": str(path)}


@app.function(image=image, volumes={"/evaluation": volume}, cpu=(2, 2), memory=(8192, 8192),
              timeout=300, retries=0, max_containers=1, scaledown_window=2)
def preflight(image_id: str):
    return execute("preflight", image_id)


@app.function(image=image, volumes={"/evaluation": volume},
              gpu=["L4", "A10G", "L40S"], cpu=(2, 2), memory=(8192, 8192), timeout=900,
              retries=0, max_containers=1, scaledown_window=2)
def baseline(image_id: str):
    return execute("baseline", image_id)


@app.local_entrypoint()
def main(stage: str = "preflight"):
    if stage == "preflight":
        print(json.dumps(preflight.remote(image.object_id)))
    elif stage == "baseline":
        print(json.dumps(baseline.remote(image.object_id)))
    else:
        raise ValueError("Unknown stage")
