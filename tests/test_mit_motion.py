# SPDX-License-Identifier: MIT
"""Numerical contracts; independent SciPy/matrix oracles, no external Motion."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from unimate.utils.quaternions import Quaternions
from unimate.utils.animation import (Animation, positions_global,
    rotations_global, transforms_local, transforms_global, offsets_from_positions)


class QuaternionContracts(unittest.TestCase):
    def test_scipy_composition_inverse_vectors_and_broadcasting(self):
        rng = np.random.default_rng(82365)
        a = rng.normal(size=(19, 4));a /= np.linalg.norm(a, axis=-1, keepdims=True)
        b = rng.normal(size=(19, 4));b /= np.linalg.norm(b, axis=-1, keepdims=True)
        vectors = rng.normal(size=(19, 3))
        ra = Rotation.from_quat(a[:, [1, 2, 3, 0]])
        rb = Rotation.from_quat(b[:, [1, 2, 3, 0]])
        qa, qb = Quaternions(a), Quaternions(b)
        np.testing.assert_allclose(qa.rotation_matrix(), ra.as_matrix(), atol=2e-15)
        np.testing.assert_allclose((qa * qb).rotation_matrix(), (ra * rb).as_matrix(), atol=3e-15)
        np.testing.assert_allclose(qa * vectors, ra.apply(vectors), atol=3e-15)
        np.testing.assert_allclose((-qa).rotation_matrix(), ra.inv().as_matrix(), atol=3e-15)
        np.testing.assert_allclose(qa[:, None] * vectors[:, None], ra.apply(vectors)[:, None], atol=3e-15)
        np.testing.assert_allclose(qa[0] * vectors, ra[0].apply(vectors), atol=3e-15)

    def test_intrinsic_euler_and_column6d(self):
        angles = np.array([[0, np.pi/2, 0], [0.4, -0.2, 0.9], [np.pi, 0, 0]])
        for order in ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'):
            q = Quaternions.from_euler(angles, order)
            expected = Rotation.from_euler(order.upper(), angles).as_matrix()
            np.testing.assert_allclose(q.rotation_matrix(), expected, atol=2e-15)
            np.testing.assert_allclose(q.rotation_matrix(cont6d=True),
                np.concatenate((expected[..., :, 0], expected[..., :, 1]), axis=-1), atol=2e-15)
        self.assertEqual(Quaternions.id(0).qs.shape, (0, 4))

    def test_between_parallel_antiparallel_and_invalid(self):
        source = np.array([[0., 0., -1.], [1., 0., 0.], [0., 0., 1.], [1., 2., 3.]])
        target = np.array([[0., 0., 1.], [1., 0., 0.], [0., 0., -1.], [-3., 2., 1.]])
        q = Quaternions.between(source, target)
        np.testing.assert_allclose(q * (source/np.linalg.norm(source, axis=-1, keepdims=True)),
            target/np.linalg.norm(target, axis=-1, keepdims=True), atol=1e-15)
        with self.assertRaises(ValueError):Quaternions.between(np.zeros(3), np.ones(3))
        with self.assertRaises(ValueError):-Quaternions(np.zeros((1, 4)))
        with self.assertRaises(ValueError):Quaternions([[np.nan, 0, 0, 1]])

    def test_numpy_repeat_index_assignment_and_copy_boundaries(self):
        q = Quaternions.id(7)
        self.assertEqual(np.repeat(q[..., None, :], 5, axis=-2).shape, (7, 5))
        self.assertEqual(q[0].repeat(7, axis=0).shape, (7,))
        self.assertEqual(np.repeat(q[:, None], 4, axis=1).shape, (7, 4))
        copy = q.copy();q[2] = Quaternions.from_euler([0, 0.4, 0])
        self.assertFalse(np.array_equal(q.qs, copy.qs))
        part = q[1:3];part.qs[0] = [0, 1, 0, 0]
        np.testing.assert_array_equal(q.qs[1], [0, 1, 0, 0])
        with self.assertRaises(ValueError):q.repeat(2, axis=-1)


class AnimationContracts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9181)
        self.parents = np.array([-1, 0, 0, 1, 1, 2])
        self.local = rng.normal(size=(11, 6, 3))
        self.matrices = Rotation.random(66, random_state=rng).as_matrix().reshape(11, 6, 3, 3)
        raw = Rotation.from_matrix(self.matrices.reshape(-1, 3, 3)).as_quat().reshape(11, 6, 4)
        self.rotations = Quaternions(raw[..., [3, 0, 1, 2]])
        self.animation = Animation(self.rotations, self.local, Quaternions.id(0),self.local[0].copy(),self.parents)
        self.expected = np.broadcast_to(np.eye(4), (11, 6, 4, 4)).copy()
        self.expected[..., :3, :3] = self.matrices;self.expected[..., :3, 3] = self.local
        for j,p in enumerate(self.parents):
            if p >= 0:self.expected[:, j] = self.expected[:, p] @ self.expected[:, j]

    def test_full_animated_translation_fk_against_independent_matrix_chain(self):
        np.testing.assert_allclose(positions_global(self.animation), self.expected[..., :3, 3], atol=3e-15)
        np.testing.assert_allclose(transforms_global(self.animation), self.expected, atol=3e-15)
        np.testing.assert_allclose(rotations_global(self.animation).rotation_matrix(), self.expected[..., :3, :3], atol=3e-15)
        np.testing.assert_allclose(transforms_local(self.animation)[..., :3, 3], self.local, atol=0)
        self.animation.orients = Quaternions.id(6);self.animation.orients.qs[:] = [0, 0, 1, 0]
        np.testing.assert_allclose(positions_global(self.animation), self.expected[..., :3, 3], atol=3e-15)

    def test_joint_permutation_time_slice_and_independent_copy(self):
        order = [5, 3, 0, 2, 1, 4];reordered = self.animation[::2, order]
        np.testing.assert_allclose(positions_global(reordered), self.expected[::2][:, order][..., :3, 3], atol=3e-15)
        copied = reordered.copy();copied.positions[0, 0] += 10
        self.assertFalse(np.array_equal(copied.positions, reordered.positions))
        np.testing.assert_allclose(positions_global(self.animation[:, [0, 1, 3]]),self.expected[:, [0, 1, 3]][..., :3, 3],atol=3e-15)
        with self.assertRaises(ValueError):self.animation[:, [3]]
        with self.assertRaises(ValueError):Animation(self.rotations,self.local,Quaternions.id(0),self.local[0],np.array([1, 0, 0, 1, 1, 2]))
        self.assertEqual(positions_global(self.animation[:0]).shape, (0, 6, 3))
        offsets = offsets_from_positions(self.expected[0, :, :3, 3], self.parents)
        np.testing.assert_allclose(offsets[0], self.expected[0, 0, :3, 3])
        np.testing.assert_allclose(offsets[3],self.expected[0,3,:3,3]-self.expected[0,1,:3,3])


if __name__=='__main__':unittest.main(verbosity=2)
