"""Export loading preserves original animated channels through time/name mapping."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_process.utils.motion_features import _load_npz_anim


class ExportChannelPreservation(unittest.TestCase):
    def test_translating_child_survives_name_alignment_trim_and_downsample(self):
        names = ['root', 'leg', 'tip']
        parents = [-1, 0, 1]
        source_order = [0, 2, 1]
        positions = np.zeros((9, 3, 3))
        positions[:, 0, 2] = np.arange(9) * 0.1
        positions[:, 1, 0] = 1 + np.arange(9) * 0.03
        positions[:, 2, 1] = 2 - np.arange(9) * 0.02
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (9, 3, 1))
        rest = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'translated.npz'
            np.savez(path, names=np.array(names)[source_order],
                     parents=np.array([-1, 2, 0]),
                     rest_local_pos=rest[source_order],
                     anim_local_pos=positions[:, source_order],
                     anim_local_rot=rotations[:, source_order], fps=120)
            motion, fps, reason = _load_npz_anim(
                path, parents, 3, None, [0, 1, 2], 30,
                raw_names=names, head_trim=1)
        self.assertIsNone(reason)
        self.assertEqual(fps, 30)
        np.testing.assert_array_equal(motion.positions, positions[[1, 5]])
        np.testing.assert_array_equal(motion.rotations.qs, rotations[[1, 5]])
        np.testing.assert_array_equal(motion.offsets, rest)
        np.testing.assert_array_equal(motion.parents, parents)


if __name__ == '__main__':
    unittest.main()
