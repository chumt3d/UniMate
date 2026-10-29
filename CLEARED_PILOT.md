# Cleared-data v3 pilot scope

The Tiny3D pilot keeps v3 model dimensions and losses. It starts the denoiser,
EMA and optimizer afresh, fits one global normalization pool from its frozen
training split, and uses only separately licensed frozen text-encoder weights.
Published UniMate model/optimizer/statistics artifacts and learned annotation
vocabularies are excluded.

`unimate.utils.quaternions` and `unimate.utils.animation` provide the numerical
operations required by the actual preprocessing, augmentation and recovery
consumers. They use MIT UniMate math and independent NumPy implementations.
They do not copy or import the external Motion implementation. Preserve this
repository's MIT notice. The removed Motion dependency's linked original code
restricts use to academic/noncommercial purposes; it must not re-enter the
cleared runtime through requirements installation or an old image/volume.

These utilities implement the scoped caller contracts, not the whole Motion
library. Original animated local translations are retained during loading;
the unchanged fixed-offset RIC/FK recovery gate decides compatibility. Six-D
rotations follow the executing first-two-columns convention. Fixed training
padding remains71 joints/depth20 without selecting inference mode or disabling
training augmentation. Invocation-only update stops preserve the run's
optimizer, LR schedule and exact-resume identity.

Independent SciPy/matrix checks and33 real cleared Labrador source-function
recoveries passed before packaging. The export-loader regression covers
translation preservation through original-name alignment, trim and source-rate
downsampling and fails on the original loader. Full dataset, actual text cache,
GPU training/resume, model quality, retargeting, skin, contacts and the artist's
Take/Blender workflow require their own evidence; these checks do not establish
them. Physical scale, anatomical labels and independent generalization are not
claimed for the dimensionless within-source pilot.

Optional external BVH/IK workflows are outside the pilot and have not been
reimplemented or qualified. In particular, the lazy RIC-to-IK branch in
`data_process/mesh_animation/animate_motion.py` remains unsupported in the
cleared runtime. Its existence grants no permission to install the removed
dependency. Use the fixed-offset decoder for this pilot. The original broad
requirements file is source provenance; the pilot image uses a separately
hashed minimal dependency list and a code-only allowlist.

The dataset consumer also translates each stored window by its initial root
XZ before feature extraction. Stage4 intentionally retains the full action's
coordinates, whereas feature recovery integrates root XZ from zero. Previously
only the first window passed recovery for a moving root. This consumer deviation
retains every source/window NPZ, preserves motion and absolute Y, and leaves
the1e-4 RIC/FK gate unchanged. The regression loads two real window NPZs from
an analytic turning chain and checks both recovered trajectories.
