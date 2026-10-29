"""Whole-action windows remain usable when their initial world origin differs."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from unimate.dataset.mixture.dataset import MotionDataset
from unimate.utils.motion_utils import (
    recover_unimate_joint_pos_from_ric,
    recover_unimate_joint_pos_from_rot,
)


class WindowedMotionLoading(unittest.TestCase):
    def test_all_windows_load_and_recover_their_root_relative_trajectory(self):
        # Analytic rotating three-joint chain, independent of the feature/FK code.
        frame = np.arange(12, dtype=float)
        angle = frame * 0.07
        root = np.stack([frame * 0.4, 1 + frame * 0.05, frame * -0.2], axis=-1)
        arm = np.stack([np.cos(angle), np.zeros(12), -np.sin(angle)], axis=-1)
        positions = np.stack([root, root + arm, root + arm + [0, 1, 0]], axis=1)
        rotations = np.tile([1.0, 0, 0, 0], (12, 3, 1))
        rotations[:, 0, 0] = np.cos(angle / 2)
        rotations[:, 0, 2] = np.sin(angle / 2)
        facing = rotations[:, 0].copy()
        facing[:, 1:] *= -1
        names = ['root', 'arm', 'tip']
        parents = np.array([-1, 0, 1])
        condition = dict(parents=parents, spectral_feats=np.zeros((3, 8)),
                         edge_indexs=np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
                         joint_graph_dists=np.zeros((3, 3)),
                         joint_relations=np.zeros((3, 3)), joint_depths=np.arange(3),
                         tpos_first_frame=np.array([[0., 1, 0], [1, 1, 0], [1, 2, 0]]),
                         tpos_local_rotations=np.tile([1., 0, 0, 0], (3, 1)),
                         joint_names=names, clean_joint_names=names)
        # Text encoding and constructor orchestration have separate acceptance;
        # this boundary reads real NPZ files and runs the actual loader/validator.
        dataset = MotionDataset.__new__(MotionDataset)
        dataset.max_freqs = 8
        dataset.ground_motion_height = False
        dataset._joint_name_emb = {name: np.zeros(2) for name in names}
        dataset._caption_emb = dataset._extra_caption_emb = {}
        dataset._target_object_types = dataset._target_clip_stems = None
        dataset._fps_unset_dirs = set()
        dataset.dataset_object_count = {}
        dataset.cond_dict = {}
        captions = {f'chain-capture-{i:03d}': 'moving chain' for i in range(2)}
        motions, loaded_names, types = {}, [], set()
        with tempfile.TemporaryDirectory() as directory:
            for i, begin in enumerate((0, 6)):
                np.savez(Path(directory) / f'chain-capture-{i:03d}.npz',
                         global_positions=positions[begin:begin + 6],
                         local_rotations=rotations[begin:begin + 6],
                         root_facing_quat=facing[begin:begin + 6], fps=30)
            dataset._load_multi_topology(
                dict(cond_dict={'chain': condition}, motion_dir=directory, captions=captions),
                'general', motions, types, loaded_names)
            self.assertEqual(set(motions), {f'{name}.npz' for name in captions})
            for i, begin in enumerate((0, 6)):
                entry = motions[f'chain-capture-{i:03d}.npz']
                expected = positions[begin:begin + 5].copy()
                expected[..., 0] -= root[begin, 0]
                expected[..., 2] -= root[begin, 2]
                np.testing.assert_allclose(recover_unimate_joint_pos_from_ric(entry['motion']),
                                           expected, atol=1e-10)
                np.testing.assert_allclose(recover_unimate_joint_pos_from_rot(
                    entry['motion'], parents, entry['offsets']), expected, atol=1e-10)
                with np.load(Path(directory) / f'chain-capture-{i:03d}.npz') as stored:
                    np.testing.assert_array_equal(stored['global_positions'], positions[begin:begin + 6])


if __name__ == '__main__':
    unittest.main()
