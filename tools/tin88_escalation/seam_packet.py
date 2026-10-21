"""Measure source expansion seams; joint telemetry, never skin/contact acceptance."""

import json
import time
from pathlib import Path

import modal

image = modal.Image.from_id("im-zbZcs3EcWvNACKYNyL0x6D")
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


@app.function(image=image, volumes={"/evaluation": volume}, cpu=(2, 2), memory=(2048, 2048),
              timeout=120, retries=0, max_containers=1, scaledown_window=2)
def measure(stage: str):
    import os
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    import numpy as np
    if not stage.startswith("expand-") or "/" in stage or ".." in stage:
        raise ValueError("An existing expansion analysis stage is required")
    root = Path("/evaluation/unimate-20261021-escalation-analysis") / stage
    if json.loads((root / "receipt.json").read_text())["status"] != "passed":
        raise ValueError("Unfinished analysis")
    output = root / "seams.json"
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    sample = json.loads((root / "00/report.json").read_text())["samples"][0]
    arrays = np.load(sample["raw_arrays"]["path"], allow_pickle=False)
    positions = arrays["fk_positions"]
    if len(positions) != 410:
        raise ValueError("Expected the declared eight-window chain")
    length = sample["normalization"]["L_used"]
    fps = sample["fps"]
    seams = np.arange(60, 410, 50)
    centers = np.arange(1, len(positions) - 1)
    interior = np.all(np.abs(centers[:, None] - seams[None, :]) > 1, axis=1)
    acceleration = np.linalg.norm(np.diff(positions, n=2, axis=0), axis=-1) * fps ** 2 / length
    step = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / length
    p99 = np.quantile(acceleration[interior], .99, axis=0)
    names = arrays["joint_names"].tolist()
    result = {"format": "tin88-unimate-expansion-seams/1", "input": sample["input"],
              "rawArrays": sample["raw_arrays"], "normalization": sample["normalization"],
              "method": "World FK central second difference; interior excludes centers within one frame of every append boundary. Per-joint p99, no contact labels.",
              "acceptance": "Joint-only diagnostic; not the original-skin/per-leg frozen AT2 verdict", "seams": []}
    for frame in seams:
        peak = acceleration[[frame - 2, frame - 1]].max(axis=0)
        ratio = np.divide(peak, p99, out=np.zeros_like(peak), where=p99 > 0)
        result["seams"].append({"firstAppendedFrame": int(frame), "maxJointStepL": float(step[frame - 1].max()),
            "jointsExceedingOwnInteriorP99": int(np.sum(peak > p99)), "maximumRatio": float(ratio.max()),
            "worstJoint": names[int(np.argmax(ratio))],
            "perJoint": [{"name": name, "stepL": float(step[frame - 1, i]),
                "accelerationLPerSecondSquared": float(peak[i]), "interiorP99": float(p99[i]),
                "ratio": float(ratio[i]) if p99[i] > 0 else None} for i, name in enumerate(names)]})
    result["seconds"] = time.monotonic() - started
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return {"output": str(output), "seconds": result["seconds"], "seams": [
        {k: v for k, v in row.items() if k != "perJoint"} for row in result["seams"]]}


@app.local_entrypoint()
def main(stage: str):
    print(json.dumps(measure.remote(stage)), flush=True)
