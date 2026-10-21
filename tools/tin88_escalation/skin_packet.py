"""Integration-only original-fox readback and rendering, on bounded remote CPU."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BUNDLE = Path("/tmp/tin88-lanes/unimate-escalation/source-tools")
HANDOFF = Path("/tmp/unimate-impl-handoff-01a1603d")
CONTROL = Path("/mnt/c/Users/bened/.codex/visualizations/2026/10/21/01a15d5e-b142-7272-acae-7b030533b12c/unimate-procedural-control")
image = (modal.Image.from_id("im-zbZcs3EcWvNACKYNyL0x6D")
    .add_local_file("/home/adibuer/.local/share/fnm/aliases/default/bin/node", "/opt/escalation/node", copy=True)
    .add_local_file(BUNDLE / "native-readback.mjs", "/opt/readback/native-readback.mjs", copy=True)
    .add_local_file(BUNDLE / "dist/readback-core.mjs", "/opt/readback/dist/readback-core.mjs", copy=True)
    .add_local_file(BUNDLE / "readback-source-manifest.json", "/opt/readback/readback-source-manifest.json", copy=True)
    .add_local_file(HANDOFF / "fox-animations.glb", "/opt/fox/fox-animations.glb", copy=True)
    .add_local_file(HANDOFF / "services/unimate-motion-service/evaluation/fox_quality/inputs/actual-fox-geometry.json", "/opt/fox/geometry.json", copy=True)
    .add_local_file(HANDOFF / "services/unimate-motion-service/release/profiles/fox-topology-profile.json", "/opt/fox/profile.json", copy=True)
    .add_local_file(CONTROL / "synthetic-control-motion.json", "/opt/fox/control.json", copy=True)
    .add_local_file(HERE / "render_skin.py", "/opt/escalation/render_skin.py", copy=True)
    .run_commands("chmod 755 /opt/escalation/node")
    .env({k: "2" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")}))
volume = modal.Volume.from_name("tin88-unimate-eval", environment_name="main")
app = modal.App("tin88-unimate-escalation")


def identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@app.function(image=image, volumes={"/evaluation": volume}, cpu=(2, 2), memory=(4096, 4096),
              timeout=600, retries=0, max_containers=1, scaledown_window=2)
def readback(image_id: str):
    root = Path("/evaluation/unimate-20261021-escalation-skin/initial")
    root.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "image": image_id, "startedUnix": time.time(), "results": []}
    try:
        executable = "/opt/escalation/node"
        receipt["nodeVersion"] = subprocess.check_output([executable, "--version"], text=True).strip()
        receipt["code"] = [identity(Path(p)) for p in (executable, "/opt/readback/native-readback.mjs",
            "/opt/readback/dist/readback-core.mjs", "/opt/readback/readback-source-manifest.json", "/opt/escalation/render_skin.py")]
        cases = [
            ("v3-walk-in-place", "uniml3d_v3", "walks_in_place_with_its_tail_swaying"),
            ("truebones-walk-in-place", "truebones80k", "walks_in_place_with_its_tail_swaying"),
            ("truebones-trot-forward", "truebones80k", "trots_forward"),
            ("v3-trot-in-place", "uniml3d_v3", "trots_in_place"),
            ("procedural-known-plant-control", None, None),
        ]
        common = ["--asset", "/opt/fox/fox-animations.glb", "--geometry", "/opt/fox/geometry.json", "--profile", "/opt/fox/profile.json"]
        for label, model, prompt in cases:
            case_root = root / label
            case_root.mkdir()
            output = case_root / "readback"
            if model is None:
                command = [executable, "/opt/readback/native-readback.mjs", "samples", *common,
                           "--decoded", "/opt/fox/control.json", "--output", str(output)]
            else:
                folder = Path("/evaluation/unimate-20261021-escalation-b/samples") / model / "artist_fox/1729/motions"
                paths = list(folder.glob(f"fox-an_object_{prompt}-rep_0-*.npy"))
                if len(paths) != 1:
                    raise ValueError(f"Ambiguous or missing source: {label}")
                manifest = {"sourceCommit": "b78c7807df7cf4231d4e67f8d0ece29c5e11025b",
                    **{key: identity(Path(f"/opt/fox/{file}")) for key, file in
                       (("asset", "fox-animations.glb"), ("geometry", "geometry.json"), ("profile", "profile.json"))},
                    "motions": [{**identity(paths[0]), "id": label}]}
                manifest_path = case_root / "input-manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                command = [executable, "/opt/readback/native-readback.mjs", "samples", "--manifest", str(manifest_path), "--output", str(output)]
            with (case_root / "readback.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            batch = json.loads((output / "batch-receipt.json").read_text())
            motion_directory = Path(batch["results"][0]["directory"])
            rig = json.loads((output / "rig-context.json").read_text())
            render_command = ["/opt/conda/bin/python", "/opt/escalation/render_skin.py", "--input", str(motion_directory),
                "--output", str(case_root / "renders"), "--floor", str(rig["restFullSkinMinimumY"]), "--title", label]
            volume.commit()
            with (case_root / "render.log").open("w") as log:
                subprocess.run(render_command, check=True, stdout=log, stderr=subprocess.STDOUT)
            receipt["results"].append({"label": label, "readback": str(motion_directory), "renders": str(case_root / "renders")})
            volume.commit()
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.time() - receipt["startedUnix"]
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return receipt


@app.local_entrypoint()
def main():
    print(json.dumps(readback.remote(image.object_id)), flush=True)
