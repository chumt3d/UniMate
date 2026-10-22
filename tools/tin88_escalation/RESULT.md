# UniMate escalation technical result — 22 October 2026

The public author Tiger artifact reproduces. The tested artist-fox raw motions
still do not meet locomotion acceptance. A working same-rig procedural control
and **procedural legs + UniMate overlay** now complete 410 frames with closely
held declared plants. This narrows the useful path without claiming that UniMate
cannot produce a good creature motion.

Recommend **v3/150k as the pinned inference reference and non-leg overlay
candidate**, with the exact tested settings in [SAMPLING-GRID.json](SAMPLING-GRID.json).
No tested checkpoint or browser step count is approved here for raw fox
locomotion. Keep procedural locomotion ownership explicit. Do not promote
Truebones80k because it travels farther, or Euler200 because it uses more steps.
This recommendation is for the internal previs-artist Increment1 investigation;
it does not complete Check rig, steering, Describe, Accept/Undo/Redo, reload,
export, or the real 8 GB device workflow.

## Reproduction and discriminating experiments

- The [author's issue-8 Tiger fixture](https://github.com/Friedrich-M/UniMate/issues/8#issuecomment-6002771945),
  seed0, CFG3, eight 60-frame outputs, reproduced with maximum absolute feature
  difference **6.7726e-6**, RMS **2.19e-7–4.99e-7**. Its original input arrays and
  our generated arrays are retained. This is positive inference evidence for
  one exact public fixture, not an animal-contact ground truth.
- The complete screen covered **60 clips**: two checkpoints, two conditioned
  rigs, three seeds and five exact prompts. Each model used its own bound config,
  EMA and statistics. Native Fox conditioning equality to its old training
  preprocessing remains unverified; equal integer seeds across model widths
  are not paired initial noise.
- Eleven selected **410-frame, 30fps** chains completed: eight windows of60,
  overlap10, with seven50-frame additions. The first window uses released
  adaptive dopri5; continuations use the unchanged constrained Euler sampler
  with num_steps50. There is no repetition of playback frames to manufacture
  strides. A 13.67-second clip does not by itself prove three valid strides.
- One paired Tiger110-frame experiment changed only continuation num_steps50
  to200. Maximum seam joint step changed **0.1849479→0.1769884 canonical L
  (−4.30%)**; acceleration/interior-p99 **11.1863→11.0347**. Finer discretization
  alone did not remove this observed discontinuity. This does not rule out every
  other solver or conditioning change.

The [pinned released source](https://github.com/Friedrich-M/UniMate/tree/b78c7807df7cf4231d4e67f8d0ece29c5e11025b)
uses free adaptive dopri5, atol1e-6/rtol1e-3 and50 saved times; that is not50
denoiser evaluations. It differs from the paper's fixed-Euler description.
Browser midpoint10/20/50, CFG5/7, native WebGPU timing and full E.5 foot locking
were not qualified. E.5's numerical policy was not invented to fill its gaps.

## Actual original-skin comparison

All nine cases completed readback and corrected **1000×750 side, three-quarter
and top videos**. They use the artist's original weighted vertices and triangles,
a plain material, fixed clip scale and a thin opaque floor outline. The original
skin measurements are independent of rendering. The side figure is only about
180px tall, which limits fine visual judgments even though the full image is750px.

Lengths below are the bound asset's metre convention; normalized sole values
use each leg's own length. The earlier screen's canonical root-to-lowest-joint L
is a different quantity and must not be substituted.

| Case | Maximum full-skin burial (m) | Root net/path (m) | Largest append acceleration / own interior p99 |
|---|---:|---:|---:|
| v3 walk1729 | 0.049819 | 0.503 /0.911 | 8.650 |
| v3 walk2718 | 0.037412 | 0.437 /0.735 | 7.308 |
| v3 walk31415 | 0.074362 | 0.455 /0.773 | 8.533 |
| Truebones forward1729 | 0.087433 | 16.586 /18.949 | 2.332 |
| Truebones forward2718 | 0.078789 | 16.272 /18.318 | 2.356 |
| Truebones forward31415 | 0.147221 | 15.804 /18.369 | 2.826 |
| v3 walk1729, source floor-only | 2.38e-16 | 0.503 /0.911 | 8.785 |
| Procedural control | 7.25e-7 | 8.454 /8.454 | 1.000 |
| Procedural legs + UniMate overlay | 1.84e-6 | 8.454 /8.454 | 2.809 |

The seam ratios are descriptive, not a pass threshold or an absolute comparison
between motions with different interior acceleration. Raw boundaries occur at
60,110,…,360; procedural windows at60,120,…,360.

For v3 seeds1729/2718/31415, the left-hind minimum sole heights are
**−0.030407/−0.022835/−0.045387 per-leg L**, with **18/3/20 frames** below the
reviewer's−0.02L burial band. Right-hind minima are
**+0.124915/+0.036825/+0.132999L**; the first and third never enter the descriptive
≤0.05L low-height band. These facts do not require a stance classifier.

The full report retains every foot's low-height intervals, minimum world
horizontal speed, interval root travel and translation-relative foot travel.
The latter subtracts root translation but does not remove root orientation.
The in-place prompt is a treadmill-like case: world foot motion alone cannot
be treated as a world-plant failure. **Raw F1–F3 remain unscorable** because a
qualified stance evaluator is absent.

ANIM_POSE_2 independently inspected side-video samples every10 frames over each
clip and every5 frames over40–159. Its scoped ruling is **F4/V reopening not met
on any of the three seeds; U26 remains closed**. It observed a repeatedly tucked,
hovering hind leg and frequent hind-leg crossing; side views cannot identify the
hind side or certify exact paw contact. It did not review the three-quarter,
top, Truebones or floor-only videos or grade control naturalness. This is an
independent model review, not practitioner or customer validation.

The owner accepts that bounded ruling and corrects three labels in the earlier
review: Truebones-forward is another generated output, not source mocap; the
procedural declared stance is8 frames per20-frame cycle, not the approximately
15-frame descriptive low-height band; and a legitimate in-place prompt must
not fail because net travel is small. The camera follows rootXZ, so screen
stationarity is not an independent motion measurement. The persistent hind
hover and measured burial remain after those corrections. Review record:
`.../01a15f22-ef4e-72b2-8da7-50aba0d11a02/REVIEW.md`, final side-view section.

Source floor-only correction removes below-floor vertices by raising the root;
it leaves the right-hind hover and horizontal movement. It is not foot locking.
Truebones' greater forward travel is accompanied by actual sole burial and
large front-foot excursions; travel alone is not useful gait acceptance.

The procedural and hybrid controls have maximum declared-stance target residuals
**9.88e-7L and1.62e-6L**, and maximum planar stance excursions
**9.76e-7L and1.11e-6L**. All retained touchdown/phase comparisons against the
declared20-frame/.4 control match. These are authored target states checked
against actual skin probes, not detected raw contacts. The hybrid authors legs
and root and retains generated non-leg rotations. Its plant quality therefore
belongs to the procedural core. Unrestricted diagnostic bend limits, mean shared
swing height and absent optional body/axial motion prevent a professional
naturalness claim. The next work belongs in that user-facing control path and
the source-reference motion defect, before a broad faster-sampler sweep.

## Exact published-case reproduction limits

The [paper](https://arxiv.org/html/2609.05415v1) and
[project](https://linzhanmou.com/unimate/) inventory retained58 held-out prompt
rows and22 interactive creature cases. Their public catalog does not bind exact
per-case checkpoint, seed, solver, feature stream and contact timings. Interactive
trajectories are protected encrypted .rig files; guessed public .glb paths404.
No access control was bypassed. Some displayed captions are descriptions added
after generation, not verified literal prompts. Exact paper/showcase reproduction
therefore needs the original manifest and permitted trajectory inputs. The
separately trained180-frame model is not one of the inspected f60 releases.
These limits do not negate the successful exact author Tiger reproduction.

## Evidence, resource use and remaining scope

Durable evidence root:
`/mnt/c/Users/bened/.codex/visualizations/2026/10/21/01a1603d-4243-75a1-ad8d-47414b83c28e/`.
`final-evidence/` contains655 recovered files (285,610,586 bytes) from15 verified
archives, with no missing roots. `evidence-recovery-receipt.json` binds every file.
Redundant local archive staging was removed after verified extraction; source
archives remain on the Modal volume. Corrected27-video metadata independently
reads410 frames,30fps and1000×750 in `video-metadata-final.json`.
`nine-case-consumer-verification.json` joins all nine measurement-input hashes
and27 video hashes; all corrected views use renderer SHA3208654e.
The full measurement report is
`skin-metrics-final/unimate-20261021-escalation-skin-metrics/final/report.json`.
`published-cases/INVENTORY.md` contains the source-by-source reproduction record.

Original asset GLB SHA7610eda5, geometryed5ed33e,48 original/47 effective joints,
reader sourcecf183 and decoderb78 remain bound. The410-frame adapter retains
the original root/rotation channels and calls the unchanged60-frame reader.
Its diagnostic composition does not establish arbitrary-rig admission or a
complete product/native-DCC source-equivalence proof.

Latest Modal provider report: **$0.89963776 total**, including **$0.72013197 GPU**;
**$0.05567480 additional** against the resumed$3 allowance. No new GPU generation
was launched in the final phase. The report includes the19:00 CEST hour on22Oct;
later recovery egress may not yet appear. It is a provider usage snapshot, not a
final invoice. Remote case bodies took114–184seconds, final measurements16.15s,
packing172.18s, recovery47s and local guarded extraction143s. Laptop restarts and
shared-memory waits interrupted elapsed delivery; these body timings do not
claim the complete product workflow meets its latency target. No CI or production
deployment ran. `modal-shutdown-receipt.json` confirms all14 listed apps for
this research name are stopped, with zero running tasks.

Weights are declared CC BY-NC4.0 in the
[pinned HF model card](https://huggingface.co/Linzhan/UniMate/blob/174fc755ec248d0c98f27759b7a95648e7f05905/README.md).
DEC10 remains the commercial decision. Browser/device, real application history,
native Blender/export, professional gait quality and commercial acceptance are
outside this completed technical evidence packet and remain unproven.
