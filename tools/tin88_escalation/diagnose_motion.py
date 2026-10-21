"""One-time raw UniMate locomotion telemetry; no gait/contact acceptance.

Run with the executing UniMate and Motion directories on PYTHONPATH. Saved
sample NPYs are already denormalized. No motion alignment, grounding, smoothing,
IK, or foot locking is applied here. Cond NPYs must be trusted (pickle format).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def resolve_reference(reference: str, manifest: Path, root: Path | None = None) -> Path:
    path = Path(reference)
    candidates = [path] if path.is_absolute() else [manifest.parent / path, Path.cwd() / path]
    if root is not None and not path.is_absolute():
        candidates.insert(0, root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve {reference!r}; tried {[str(p) for p in candidates]}")


def cond_path(asset: str, entry: dict, manifest: Path, root: Path | None) -> Path:
    # --asset-root deliberately overrides stale absolute paths in retained runs.
    if root is not None:
        candidates = [root / asset / "cond.npy", root / "cond.npy"]
        if not Path(entry["cond_path"]).is_absolute():
            candidates.insert(0, root / entry["cond_path"])
        found = {p.resolve() for p in candidates if p.is_file()}
        if len(found) == 1:
            return found.pop()
        if len(found) > 1:
            raise ValueError(f"Ambiguous --asset-root for {asset}: {sorted(map(str, found))}")
    return resolve_reference(entry["cond_path"], manifest, root)


def choose_feet(names: list[str], override: dict | None) -> dict:
    schemes = [
        ("reviewed_fox", ["Front_Leg_Foot_L", "Front_Leg_Foot_R", "Back_Leg_Foot_L", "Back_Leg_Foot_R"]),
        ("tiger", ["Front_Leg_Foot_L", "Front_Leg_Foot_R", "Back_Leg_Ankle_L", "Back_Leg_Ankle_R"]),
        ("truebones_fox", ["Bip01_L_Hand", "Bip01_R_Hand", "Bip01_L_Foot", "Bip01_R_Foot"]),
        ("unitree_go2", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
    ]
    primary: dict[str, str] = {}
    scheme = "UNRESOLVED"
    if override is not None:
        if not isinstance(override, dict) or not all(isinstance(v, str) for v in override.values()):
            raise ValueError("Each --feet-json asset entry must map probe labels to exact joint names")
        primary, scheme = override, "explicit_cli_mapping"
    else:
        for label, joints in schemes:
            if all(name in names for name in joints):
                scheme = label
                primary = dict(zip(("FL", "FR", "BL", "BR"), joints))
                break
    missing = [name for name in primary.values() if name not in names]
    if missing:
        raise ValueError(f"Explicit foot names absent from cond: {missing}")
    extra_names = [
        "Front_Leg_Tip_L", "Front_Leg_Tip_R", "Back_Leg_Tip_L", "Back_Leg_Tip_R",
        "Bip01_L_Toe0", "Bip01_R_Toe0", "Bip01_L_Toe0Nub", "Bip01_R_Toe0Nub",
        "Bip01_L_Finger0", "Bip01_R_Finger0", "Bip01_L_Finger01", "Bip01_R_Finger01",
        "Bip01_L_Finger0Nub", "Bip01_R_Finger0Nub",
    ]
    probes = [{"label": label, "name": name, "index": names.index(name), "role": "primary"}
              for label, name in primary.items()]
    if primary:
        probes += [{"label": name, "name": name, "index": names.index(name), "role": "additional_distal_probe"}
                   for name in extra_names if name in names and name not in primary.values()]
    return {"scheme": scheme, "probes": probes,
            "status": "joint_probes_only" if primary else "UNRESOLVED; supply --feet-json; global telemetry retained"}


def ground_setting(run: dict, manifest: Path, root: Path | None, override: str) -> dict:
    config = None
    if run.get("exp_dir"):
        try:
            config = resolve_reference(str(Path(run["exp_dir"]) / "config.json"), manifest, root)
        except FileNotFoundError:
            pass
    actual = None
    provenance: dict[str, Any] = {"config": None}
    if config is not None:
        content = json.loads(config.read_text())
        actual = content.get("dataset", {}).get("ground_motion_height")
        if actual is not None and not isinstance(actual, bool):
            raise ValueError(f"Invalid dataset.ground_motion_height in {config}")
        provenance["config"] = identity(config)
        provenance["config_ground_motion_height"] = actual
    if override != "auto":
        grounded = override == "grounded"
        source = "explicit_cli_override"
    elif actual is not None:
        grounded, source = actual, "run_config.dataset.ground_motion_height"
    else:
        grounded, source = True, "inferred_upstream_schema_default_TRUE; verify or override"
    return {**provenance, "ground_motion_height": grounded, "source": source}


def stats(values, np) -> dict:
    values = np.asarray(values)
    if values.size == 0:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(values.size), "min": float(values.min()), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def spectrum(curves, fps: float, np) -> tuple[dict, dict]:
    """Detrended Hann-window periodogram: candidate repetition, never stance."""
    from scipy.signal import detrend

    count = len(curves)
    frequencies = np.fft.rfftfreq(count, d=1.0 / fps)
    if curves.shape[1] == 0:
        return ({"status": "UNSCORABLE: unresolved foot mapping", "per_probe_axes": []},
                {"spectrum_frequency_hz": frequencies,
                 "body_relative_spectrum_power_L2": np.empty((len(frequencies), 0, 3))})
    centered = detrend(curves, axis=0, type="linear") if count > 1 else np.zeros_like(curves)
    window = np.hanning(count)
    transformed = np.fft.rfft(centered * window[:, None, None], axis=0)
    power = np.abs(transformed) ** 2 / max(float(np.sum(window ** 2)), 1.0)
    candidates = []
    for foot in range(curves.shape[1]):
        axes = []
        for axis in range(3):
            p = power[1:, foot, axis]
            total = float(p.sum())
            order = np.argsort(p)[::-1][:3] + 1 if total > 0 else []
            axes.append({"axis": "xyz"[axis], "detrended_rms_L": float(np.sqrt(np.mean(centered[:, foot, axis] ** 2))),
                         "peaks": [{"frequency_hz": float(frequencies[k]),
                                    "cycles_in_window": float(frequencies[k] * count / fps),
                                    "fraction_of_non_dc_power": float(power[k, foot, axis] / total)} for k in order]})
        candidates.append(axes)
    return ({"method": "linear_detrend_then_Hann_rFFT_squared_over_window_energy",
             "frequency_resolution_hz": fps / count, "window_seconds": count / fps,
             "interpretation": "candidate oscillations only; short windows, trends and harmonics can dominate; no cadence or gait acceptance",
             "per_probe_axes": candidates},
            {"spectrum_frequency_hz": frequencies, "body_relative_spectrum_power_L2": power})


def diagnose(motion, cond: dict, names: list[str], feet: dict, ground: dict, fps: float, l_floor: float, np, mu, animation) -> tuple[dict, dict]:
    parents = np.asarray(cond["parents"], dtype=np.int64)
    rest = np.asarray(cond["tpos_first_frame"], dtype=np.float64)
    if rest.shape != (len(names), 3) or parents.shape != (len(names),):
        raise ValueError(f"Unexpected canonical cond shapes rest={rest.shape}, parents={parents.shape}")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate cond joint_names would make exact probe identity ambiguous")
    if motion.ndim != 3 or motion.shape[1:] != (len(names), 12) or len(motion) < 2:
        raise ValueError(f"Expected denormalized T×{len(names)}×12, T>=2; got {motion.shape}")
    if not np.isfinite(motion).all() or not np.isfinite(rest).all():
        raise ValueError("Non-finite motion/rest input; no numerical conclusions emitted")
    if parents[0] not in (-1, 0) or any(p < 0 or p >= j for j, p in enumerate(parents[1:], 1)):
        raise ValueError("Upstream recovery requires parent-before-child order with root at joint 0")
    lowest = float(rest[:, 1].min())
    raw_l = float(rest[0, 1] - lowest)
    length = max(raw_l, l_floor)
    conditioned_rest = rest.copy()
    if ground["ground_motion_height"]:
        conditioned_rest[:, 1] -= lowest
    ground_y = float(conditioned_rest[:, 1].min())
    # Exactly the inference dataset's canonical offsets, not raw BVH offsets.
    offsets = animation.offsets_from_positions(conditioned_rest, parents)
    root_q, root = mu.recover_unimate_root_quat_and_pos(motion[:, 0, :])
    ric = mu.recover_unimate_joint_pos_from_ric(motion.copy())
    fk = mu.recover_unimate_joint_pos_from_rot(motion.copy(), parents, offsets)
    if not all(np.isfinite(x).all() for x in (root, ric, fk)):
        raise ValueError("Non-finite upstream recovery output")
    root_steps = np.diff(root, axis=0)
    gap = np.linalg.norm(fk - ric, axis=-1)
    indices = [p["index"] for p in feet["probes"]]
    foot_fk, foot_ric = fk[:, indices], ric[:, indices]
    body_fk = np.repeat(root_q[:, None], len(indices), axis=1) * (foot_fk - root[:, None]) if indices else foot_fk.copy()
    body_ric = np.repeat(root_q[:, None], len(indices), axis=1) * (foot_ric - root[:, None]) if indices else foot_ric.copy()
    arrays = {"time_s": np.arange(len(motion)) / fps, "joint_names": np.asarray(names), "parents": parents,
              "conditioned_rest_positions": conditioned_rest, "recovery_offsets": offsets,
              "root_positions": root, "root_facing_quaternions_wxyz": root_q.qs,
              "fk_positions": fk, "ric_positions": ric, "fk_ric_joint_distance_L": gap / length,
              "probe_indices": np.asarray(indices, dtype=np.int64),
              "probe_names": np.asarray([p["name"] for p in feet["probes"]]),
              "fk_foot_positions": foot_fk, "ric_foot_positions": foot_ric,
              "fk_foot_body_relative_positions": body_fk, "ric_foot_body_relative_positions": body_ric}
    net = root[-1] - root[0]
    report: dict[str, Any] = {
        "shape": list(motion.shape), "fps": fps, "last_minus_first_seconds": (len(motion) - 1) / fps,
        "normalization": {"definition": "paper E.5 rest root-to-lowest-joint vertical height",
                          "L_raw": raw_l, "L_used": length, "L_floor": l_floor, "floor_applied": raw_l < l_floor,
                          "raw_rest_lowest_y": lowest, "ground_plane_y": ground_y,
                          "ground_setting": ground, "units": "canonical model coordinates; no metre claim",
                          "limitation": "joint-only probes; no skinned sole/surface contact acceptance"},
        "root": {"net_vector": net.tolist(), "net_vector_L": (net / length).tolist(),
                 "net_horizontal": float(np.linalg.norm(net[[0, 2]])), "net_horizontal_L": float(np.linalg.norm(net[[0, 2]]) / length),
                 "total_horizontal_travel": float(np.linalg.norm(root_steps[:, [0, 2]], axis=-1).sum()),
                 "total_horizontal_travel_L": float(np.linalg.norm(root_steps[:, [0, 2]], axis=-1).sum() / length),
                 "net_3d": float(np.linalg.norm(net)), "net_3d_L": float(np.linalg.norm(net) / length),
                 "total_3d_travel": float(np.linalg.norm(root_steps, axis=-1).sum()),
                 "total_3d_travel_L": float(np.linalg.norm(root_steps, axis=-1).sum() / length),
                 "height_L": stats((root[:, 1] - ground_y) / length, np),
                 "horizontal_speed_L_per_s": stats(np.linalg.norm(root_steps[:, [0, 2]], axis=-1) * fps / length, np)},
        "fk_vs_ric": {"joint_distance_L": stats(gap / length, np),
                      "per_joint_mean_L": (gap.mean(axis=0) / length).tolist(),
                      "per_joint_max_L": (gap.max(axis=0) / length).tolist()},
        "foot_mapping": feet,
        "contact_and_gait_acceptance": {"stance": "UNSCORABLE", "phase": "UNSCORABLE", "F1_F2_F3": "UNSCORABLE",
                                        "reason": "No independently validated contact labels; dynamic height/speed detector rejected against known duty control"},
    }
    for mode, positions, foot, body in (("fk", fk, foot_fk, body_fk), ("ric", ric, foot_ric, body_ric)):
        heights = positions[:, :, 1] - ground_y
        horizontal_step = np.linalg.norm(np.diff(positions[:, :, [0, 2]], axis=0), axis=-1)
        speed = horizontal_step * fps
        arrays[f"{mode}_joint_heights"] = heights
        arrays[f"{mode}_joint_horizontal_speed_per_s"] = speed
        arrays[f"{mode}_foot_heights"] = heights[:, indices]
        arrays[f"{mode}_foot_horizontal_step"] = horizontal_step[:, indices]
        arrays[f"{mode}_foot_horizontal_speed_per_s"] = speed[:, indices]
        per_joint = [{"name": name, "position_min": positions[:, j].min(axis=0).tolist(),
                      "position_max": positions[:, j].max(axis=0).tolist(),
                      "height_L": stats(heights[:, j] / length, np),
                      "horizontal_speed_L_per_s": stats(speed[:, j] / length, np)} for j, name in enumerate(names)]
        per_probe = []
        for k, probe in enumerate(feet["probes"]):
            j = probe["index"]
            h, d = heights[1:, j] / length, horizontal_step[:, j] / length
            near = h < 0.05
            # Destination-frame height paired with the preceding displacement.
            # E.5 near-ground is a heuristic, never a contact/stance label.
            near_d = d[near]
            per_probe.append({"name": probe["name"], "body_relative_ptp_xyz_L": (np.ptp(body[:, k], axis=0) / length).tolist(),
                              "body_relative_horizontal_path_L": float(np.linalg.norm(np.diff(body[:, k, [0, 2]], axis=0), axis=-1).sum() / length),
                              "below_ground_fraction": float(np.mean(heights[:, j] < 0)),
                              "maximum_joint_penetration_L": float(max(0.0, -heights[:, j].min()) / length),
                              "E5_near_ground_heuristic": {
                                  "threshold_height_L": 0.05, "skate_step_threshold_L_per_frame": 0.025,
                                  "pairing": "destination frame height with preceding horizontal displacement",
                                  "near_ground_frames": int(near.sum()), "eligible_steps": len(d),
                                  "near_ground_fraction": float(near.mean()),
                                  "skate_fraction": float(np.mean(near_d > 0.025)) if len(near_d) else None,
                                  "near_ground_step_L": stats(near_d, np),
                                  "slide_0p01L": float(np.mean(near_d * (2 - np.exp2(h[near] / 0.05))) / 0.01) if len(near_d) else None,
                                  "interpretation": "raw joint heuristic; not ground truth or accepted contact"}})
        spectral, spectral_arrays = spectrum(body / length, fps, np)
        report[mode] = {"per_joint": per_joint, "per_probe": per_probe, "oscillation_candidates": spectral}
        arrays.update({f"{mode}_{key}": value for key, value in spectral_arrays.items()})
    return report, arrays


def plot_curves(path: Path, arrays: dict, report: dict, title: str, np) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = report["normalization"]["L_used"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, name in enumerate(arrays["probe_names"]):
        axes[0].plot(arrays["time_s"], arrays["fk_foot_heights"][:, k] / length, label=str(name))
        axes[1].plot(arrays["time_s"][1:], arrays["fk_foot_horizontal_speed_per_s"][:, k] / length)
        axes[2].plot(arrays["time_s"], arrays["fk_foot_body_relative_positions"][:, k, 2] / length)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("FK joint height / L")
    axes[1].set_ylabel("Horizontal speed L/s")
    axes[2].set_ylabel("Facing-frame Z / L")
    axes[2].set_xlabel("Seconds")
    if len(arrays["probe_names"]):
        axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(title + "\nJoint telemetry only; contact/phase UNSCORABLE", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, help="Override cond location: DIR/cond.npy, DIR/ASSET/cond.npy, or DIR/relative_cond_path; ambiguity is an error")
    parser.add_argument("--feet-json", type=Path, help='JSON {"asset": {"FL": "exact joint name", ...}}; optional exact-name override')
    parser.add_argument("--fps", type=float, default=30.0, help="Source stage-4 rate, default 30; affects speed and spectra only")
    parser.add_argument("--ground-mode", choices=("auto", "grounded", "canonical"), default="auto", help="auto reads run config, otherwise explicitly reports inferred upstream default")
    parser.add_argument("--l-floor", type=float, default=1e-8, help="Numerical normalization floor in canonical units; flagged when used")
    parser.add_argument("--plots", action="store_true", help="Plot first sorted sample for each asset/prompt")
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.fps, args.l_floor)):
        parser.error("--fps and --l-floor must be finite and positive")
    import numpy as np
    import Animation as animation
    from unimate.utils import motion_utils as mu

    manifest = args.manifest.resolve()
    content = json.loads(manifest.read_text())
    if content.get("format") != "unimate-samples/1":
        raise ValueError("Expected manifest format unimate-samples/1")
    overrides = json.loads(args.feet_json.read_text()) if args.feet_json else {}
    args.output.mkdir(parents=True, exist_ok=True)
    source_modules = [mu, animation, inspect.getmodule(mu.Quaternions), inspect.getmodule(mu.rotation_6d_to_matrix_np)]
    source_paths = {Path(module.__file__).resolve() for module in source_modules if module is not None and module.__file__}
    source_paths.add(Path(__file__).resolve())
    report = {"format": "tin88-unimate-raw-diagnostic/1", "manifest": identity(manifest),
              "sources": [identity(p) for p in sorted(source_paths)], "runs": content.get("runs", []),
              "fps_source": "explicit --fps" if argv and "--fps" in argv else "CLI value; default 30 from upstream stage-4 convention",
              "feet_override": identity(args.feet_json) if args.feet_json else None,
              "limitations": ["No acceptance contacts or gait phases", "No skinned sole geometry", "No model impossibility conclusion", "Saved features consumed without further normalization"],
              "assets": {}, "samples": []}
    cache = {}
    entries = content.get("samples", {})
    entries = list(entries.items()) if isinstance(entries, dict) else [(item["name"], item) for item in entries]
    plotted = set()
    for ordinal, (name, entry) in enumerate(sorted(entries)):
        asset = entry["asset"]
        if asset not in cache:
            asset_entry = content["assets"][asset]
            path = cond_path(asset, asset_entry, manifest, args.asset_root)
            conds = np.load(path, allow_pickle=True).item()
            cond = conds[asset_entry.get("cond_key", asset)]
            names = [str(n) for n in cond["joint_names"]]
            if asset_entry.get("joint_names") is not None and names != asset_entry["joint_names"]:
                raise ValueError(f"Manifest/cond joint order mismatch for {asset}")
            feet = choose_feet(names, overrides.get(asset))
            cache[asset] = cond, names, feet
            report["assets"][asset] = {"cond": identity(path), "cond_key": asset_entry.get("cond_key", asset),
                                       "joint_names": names, "foot_mapping": feet}
        cond, names, feet = cache[asset]
        run_index = entry.get("run", 0)
        run = content.get("runs", [])[run_index]
        ground = ground_setting(run, manifest, args.asset_root, args.ground_mode)
        motion_path = resolve_reference(str(Path(content.get("motions_dir", "motions")) / name), manifest)
        result, arrays = diagnose(np.load(motion_path, allow_pickle=False), cond, names, feet, ground, args.fps, args.l_floor, np, mu, animation)
        stem = f"{ordinal:04d}-" + re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).stem)
        array_path = args.output / f"{stem}.npz"
        np.savez_compressed(array_path, **arrays)
        result.update({"name": name, "asset": asset, "prompt": entry.get("prompt"), "run": run_index,
                       "kind": entry.get("kind"), "input": identity(motion_path), "raw_arrays": identity(array_path)})
        prompt_key = asset, json.dumps(entry.get("prompt"), sort_keys=True)
        if args.plots and prompt_key not in plotted:
            plot_path = args.output / f"{stem}.png"
            plot_curves(plot_path, arrays, result, f"{asset}: {entry.get('prompt')}", np)
            result["plot"] = identity(plot_path)
            plotted.add(prompt_key)
        report["samples"].append(result)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(report_path.resolve()), "samples": len(report["samples"]), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
