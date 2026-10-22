# UniMate escalation, 21 October 2026

Current state, 22 October evening: all declared generation and all nine final
original-skin comparisons are complete and recovered durably after two laptop
restarts. The packet includes three seeds per raw checkpoint condition, source
floor-only correction, pure procedural and procedural legs + UniMate overlay.
See RESULT.md and SAMPLING-GRID.json for findings and the Increment1 handoff.
Raw F1-F3 remain unscorable; no product, native DCC/export, device, naturalness
or commercial acceptance is claimed. ANIM_POSE_2 completed its independent
three-seed metrics/side-view review: F4/V reopening not met; U26 stays closed.
The owner accepted its bounded hind-hover finding and corrected comparison
labels and the interpretation of in-place travel, as recorded in RESULT.md.

The earlier fixed 2 GiB pause was lifted by ROOT. All resumed local CPU work
used heavy-run admission, and remote calls used submit-and-disconnect with
durable call IDs. The final CPU phase used no additional GPU generation.
Fifteen verified archives preserved 655 final-channel, skin-array, render,
recipe, measurement and receipt files, omitting duplicate per-window skins.
The later paragraphs below retain experiment declarations and historical states.

Owner: TIN-88 ROOT, delegated to UNIMATE_ESCALATION. Research and bounded repair;
no production release or admission. Total authorised GPU budget $4; no training
above $1 without owner approval. Scratch is disposable, evidence copied to the
delegate's Windows visualization folder and checkpoint source pushed without CI.

Outcome: a useful creature locomotion path on the artist's rig. Reproduce public
author examples first, then test failure hypotheses against the real fox and
the existing procedural baseline. Product/export acceptance stays distinct.

First fixed experiment: author issue #8 Tiger v3-150k, original conditioning,
seed 0, CFG 3, eight repetitions, prompt `An object walks in place.` Use the
retained successful Go2-b image, strict checkpoint/EMA loading, global stats,
and byte-verified b78c7807 upstream inference source. Compare all eight outputs
with the author's arrays. No conditioning, decoder, prompt, or sampler changes.

Before GPU: CPU checks model load, input identities, runtime source and imports.
Persist each run's output and failure receipt. A successful wrapper is not proof
of successful motion generation. Check eight finite arrays with correct shape.

Next questions (experiments declared after baseline, before their output):
- Can upstream expansion sustain three complete strides without seam drift?
- Does in-place versus travelling conditioning explain the fox gap?
- Do RIC target positions and rotation/FK skin positions disagree?
- Do caption-style prompt/CFG changes alter gait instead of only energy?
- Does existing E.5 or foot IK improve contacts without erasing the gait?

Prior dynamic stance detector was rejected. Its gait/duty/phase outputs are
telemetry, not F1-F3 acceptance. Report raw and corrected motion separately.

Checkpoint A result: CPU preparation17.82s; GPU generation/render106.32s;
eight author motions reproduced with maximum absolute feature error6.7726e-6,
RMS2.19e-7..4.99e-7. This validates the released inference path for this fixture,
not contact quality. App ap-hNhdDMjUoZfwWJuq5IsThd, imageim-yvwuQXEyvg2yH1dd46r6k0.

Experiment B, declared before output: compare released truebones_f60_v2/80k
against uniml3d_f60_v3/150k on native Fox39j and reviewed artist fox47j.
Use each exact config/stats/EMA; explicitly select truebones stats for the
source-specific model. Seeds1729,2718,31415, one repetition per seed, five
prompts: exact old training captions `An object walks in place with its tail
swaying.` and `An object bounds in place.`, plus `An object trots forward.`,
`An object trots in place.`, `An object walks in place.`. All use unchanged
adaptive dopri5/CFG3. Suppress only video generation for the numerical screen.
No contact, phase or gait admission inferred from a60-frame window. Checkpoint
source is HF174fc755, Truebones training caption source is UniML3Dfaaa81773b315247b03f183548e3898dbec2ce80/export/truebones/motion_captions.json.
Native conditioning comes from the prior retained pipeline; its equality to
old training preprocessing is unverified and cannot be claimed.

Experiment C, declared before GPU output: unchanged upstream expansion with
eight 60-frame windows and overlap10 produces410 fresh generated frames
(13.67s at30fps). First window uses release dopri5; seven constrained windows
use release Euler50. Reproduce Tiger seed0 first, then first-screen seed1729
artist v3/Truebones walk/trot and native Truebones walk. Extend promising
conditions to seeds2718/31415; single-seed screening cannot meet acceptance.
No repeated playback frames count as strides. Original48-joint artist-skin
decoding remains a separate consumer check. IMPL owns its source-based bundle.
Read-only GPT-6.1 Sol/high child experiment_complexity_review inspects the
experiment/source/resource contracts; escalation owner integrates findings.

Diagnostic length L is canonical rest-root-to-lowest-joint height, not the
rubric's per-leg length. Joint heuristics and spectral peaks remain telemetry;
they do not determine contact phase, gait pass/fail, or original-skin quality.

C first screen generated410finiteframes for all six declared cases. Tiger
first60features matched authorrep0 within3.40e-6; full chain took46.94sL40S.
Raw Tiger append boundaries show per-joint acceleration/interior-p99 ratios
7.50..17.39; artistv3walk2.57..8.65; artistTrueboneswalk9.31..17.94.
These are FK joint telemetry at canonical length, not the skin/leg AT2 verdict.
Same seed integers across model widths71/86 do not establish paired latent noise.

D hypothesis: is continuation Euler50 discretization responsible for seams?
Change only the existing inbetween sampler's num_steps to200, Tiger seed0,
two windows110frames. Keep every other behavior and initial free window.
Runtime override is explicitly research-only. Generation passed59.24sA10;
first60 versus C maxabs1.46e-6; continuation feature RMS difference.00315.
Paired FK seam comparison remains required before interpreting this difference.

Original-skin consumer closure from IMPL binds exact GLB7610eda5 and
geometryed5ed33e, sourcecf183, decoderb78. The diagnostic-only exact-asset guard
replaces broad generic Studio authoring admission; no arbitrary-asset or product
admission follows. Exposed procedural known-plant control uses diagnostic
unrestricted bend limits and is not a naturalness/professional-limit baseline.

D paired first110 comparison: Euler50 seam maximum joint step .18495L versus
Euler200 .17699L (-4.30%); peak acceleration/interior-p99 11.186 versus11.035.
First60 FK position maximum difference5.97e-7. Finer Euler integration alone
does not resolve this observed continuation discontinuity.

E declared after all60 B samples: Truebones artist trot-forward root travel is
1.697..1.927L across three seeds, versus v3 .140..1.547L, but mean FK/RIC gap
is larger (.155 versus.075L). Test unchanged source8-window Truebones travel
for seeds1729,2718,31415, and complete the v3 in-place walk's three seeds.
This screens useful travel and multi-stride evidence; no phase/contact pass
follows from motion energy or root displacement.

All E3seeds and the remaining C v3walk2seeds generated410finiteframes.
Original artist skin now has5short60frame readbacks/renders and v3walk1729
long410readback/render (114.10s CPU). The long integration uses reviewed
compose-long-fox-v2 f3002d01, unchanged native-readback-v2 977410b4 and core
3ea7b56c. Eight60frame skin windows retain0:60then10:60, exactly410 frames;
all original rotation/root channels are checked unchanged. Root-join tolerance
metadata is an uncertified diagnostic roundoff estimate in root-parent-local
coordinates, not a source-equivalence or motion-quality proof.

The unmodified long v3 skin minimum varies -0.05250..0.14839m, against rest
floor -0.002677655m. Rendered height alone was ambiguous: revise the diagnostic
side camera to exact elevation0 and replace filled floor with outline to avoid
whole-collection occlusion. This is view-only; the revised render remains to run.

Next predeclared comparator: unchanged procedural-window kernel71205, contiguous
windows60x6+50 with carried state. Derive a constant diagnostic speed profile
from exposed known-plant control speed.620105m/s,20frame period,.4duty, original
phases and measured average hip height. Shared swing height is the mean of
control per-leg heights, an explicit deviation. Schema requires the single
Froude0 anchor; actual calibration Froude is separate provenance. No optional
body bob/tilt/shift, axial synthesis or variation. Mandatory clearance/reach
spring still controls rootY. The endpoint extends two cycles beyond410 to cover
late footprint lookahead. Pure uses rest nonleg rotations; hybrid keeps generated
nonleg rotations exactly and authors legs/root. This is not raw model-gait proof.
Retain emitted stance/touchdown differences from the exact20frame control;
do not tune phase arithmetic after seeing output. Unrestricted diagnostic bend
limits and mean swing do not establish naturalness or professional joint limits.

Focused read-only GPT6.1Sol/high complexity review accepted410 assembly and
bounded resources; corrected schema anchor and late-footprint endpoint before
comparator execution. Numerical phase sensitivity remains measured-only.

At00:12Paris22Oct ROOT prioritized RC5.1 under WSL pressure. All three queued
own384MiB submissions were cancelled before remote launch. All stale attached
clients ended; future calls use SDK app.run(detach=True)+spawn, retaincallID,
then disconnect. Resume only at MemAvailable>=2GiB, check about10min apart.
Prepared source and actual retained results are preserved separately. LongTB
skin, remainingseedskins, sourcefloor-only and procedural/hybrid remain pending.

22 October evening resume: ROOT authorized at most $3 additional Modal spend.
The restart cleared scratch; restored exact pushed349ccad and verified all44
durable runtime bundle files. Provider baseline is $0.84396296 total, including
$0.72013197 GPU. No further GPU generation is planned.

First corrected render reached execution but failed before rendering: Matplotlib
3.10.8 tried to autoscale an empty Line3DCollection. The complete skin-array
bounds already govern the camera, so disable that collection's automatic limit
update. Preserve the failed log and use a new renders-clear-floor-v2 directory.
Verify the complete corrected clip before dispatching other renders.
