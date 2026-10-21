"""Render immutable CPU original-skin vertices; no pose or geometry correction."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor", type=float, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    from PIL import Image

    metadata = json.loads((args.input / "render-inputs.json").read_text())
    frames, vertices, axes = metadata["positions"]["shape"]
    if frames not in (60, 110, 160, 410) or not 1 <= vertices <= 10000 or axes != 3:
        raise ValueError("Outside the bounded original-fox render envelope")
    def array(spec):
        path = args.input / spec["path"]
        dtype = np.dtype(spec["dtype"])
        if path.stat().st_size != int(np.prod(spec["shape"])) * dtype.itemsize:
            raise ValueError(f"Array shape/byte mismatch: {path}")
        return np.memmap(path, dtype=dtype, mode="r", shape=tuple(spec["shape"]))
    positions = array(metadata["positions"])
    triangles = array(metadata["triangles"])
    worlds = array(metadata["jointWorldMatrices"])
    if triangles.min() < 0 or triangles.max() >= vertices or not np.isfinite(positions).all():
        raise ValueError("Invalid original triangle/position values")
    # View transform only: original Y-up -> Matplotlib Z-up; source arrays are untouched.
    view = np.asarray(positions[:, :, [0, 2, 1]])
    roots = np.asarray(worlds[:, 1, [12, 14, 13]])
    horizontal = max(float(np.abs(view[:, :, :2] - roots[:, None, :2]).max()) * 1.15, .01)
    bottom = min(args.floor, float(view[:, :, 2].min()))
    top = float(view[:, :, 2].max())
    height = max(top - bottom, .01)
    chosen = np.linspace(0, frames - 1, 8, dtype=int)
    light = np.array([.4, -.6, .7]); light /= np.linalg.norm(light)
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"scope": "CPU rendering of actual original weighted vertices and triangles; plain material, no pose correction, grounding, IK, or acceptance",
               "input": str(args.input), "frames": frames, "fps": metadata["fps"], "floorY": args.floor,
               "camera": "Orthographic; follow animated original root horizontally, constant scale across this clip",
               "contactSheetFrames": chosen.tolist(), "videos": []}
    for label, elevation, azimuth in [("side", 8, 0), ("top", 90, 0), ("three-quarter", 22, -55)]:
        fig = plt.figure(figsize=(5.6, 4.2), dpi=100)
        axis = fig.add_subplot(111, projection="3d")
        axis.set_proj_type("ortho")
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_box_aspect((2 * horizontal, 2 * horizontal, height * 1.2))
        axis.set_zlim(bottom - .06 * height, top + .14 * height)
        axis.set_xlabel("X (m)"); axis.set_ylabel("Z (m)"); axis.set_zlabel("Y (m)")
        mesh = Poly3DCollection(view[0][triangles], linewidths=0, antialiased=False)
        axis.add_collection3d(mesh)
        floor = Poly3DCollection([], facecolor="#dfe5e8", alpha=.4, edgecolor="#bcc7cd", linewidth=.4)
        axis.add_collection3d(floor)
        title = fig.suptitle("", fontsize=9)
        destination = args.output / f"{label}.mp4"
        stills = []
        with imageio.get_writer(destination, fps=metadata["fps"], codec="libx264", quality=7,
                                macro_block_size=2, output_params=["-threads", "2"]) as writer:
            for frame in range(frames):
                verts = view[frame][triangles]
                normals = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
                lengths = np.linalg.norm(normals, axis=1)
                normals /= np.maximum(lengths[:, None], 1e-12)
                shade = .35 + .65 * np.abs(normals @ light)
                mesh.set_verts(verts)
                mesh.set_facecolor(shade[:, None] * np.array([.82, .54, .29]))
                x, z = roots[frame, :2]
                axis.set_xlim(x - horizontal, x + horizontal)
                axis.set_ylim(z - horizontal, z + horizontal)
                floor.set_verts([[(x-horizontal,z-horizontal,args.floor), (x+horizontal,z-horizontal,args.floor),
                                  (x+horizontal,z+horizontal,args.floor), (x-horizontal,z+horizontal,args.floor)]])
                title.set_text(f"{args.title}\nOriginal fox skin / {label} / frame {frame} / {frame / metadata['fps']:.2f}s")
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                writer.append_data(rgb)
                if frame in chosen:
                    stills.append(rgb)
        rows.append(np.concatenate(stills, axis=1))
        plt.close(fig)
        receipt["videos"].append({"view": label, "path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.output / "contact-sheet.png")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "frames": frames}))


if __name__ == "__main__":
    main()
