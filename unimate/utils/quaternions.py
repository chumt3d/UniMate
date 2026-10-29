# SPDX-License-Identifier: MIT
"""NumPy quaternion operations required by UniMate's cleared-data pilot.

Independent implementation from quaternion algebra and the executing MIT
UniMate call-site contracts. Not an implementation of the external Motion API.
Quaternions use wxyz and active column-vector rotations. Unary minus denotes
rotation inversion because UniMate's existing formulas use that notation.
"""

import numpy as np


class Quaternions:
    """Mutable quaternion array whose ``shape`` excludes the4component axis.

    Scalar indexing retains one quaternion (shape(1,)), matching the consumers'
    ``q[0].repeat(F, axis=0)`` use. ``qs`` exposes the original NumPy view;
    ``copy()`` is the explicit independent-copy boundary.
    """

    def __init__(self, values):
        values = values.qs if isinstance(values, Quaternions) else values
        self.qs = np.asarray(values, dtype=np.float64)
        if self.qs.ndim == 1:
            self.qs = self.qs[None]
        if self.qs.ndim < 2 or self.qs.shape[-1] != 4:
            raise ValueError('Quaternions require (...,4) wxyz values')
        if not np.isfinite(self.qs).all():
            raise ValueError('Quaternion components must be finite')

    @property
    def shape(self):
        return self.qs.shape[:-1]

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, key):
        return Quaternions(self.qs[key])

    def __setitem__(self, key, value):
        self.qs[key] = value.qs if isinstance(value, Quaternions) else value

    def __array__(self, dtype=None, copy=None):
        return np.array(self.qs, dtype=dtype, copy=copy) if copy is not None else np.asarray(self.qs, dtype=dtype)

    def __array_function__(self, function, types, args, kwargs):
        if function is np.repeat:
            array = args[0]
            repeats = args[1] if len(args) > 1 else kwargs['repeats']
            rest = args[2:]
            axis = rest[0] if rest else kwargs.get('axis')
            return array.repeat(repeats, axis=axis)
        return NotImplemented

    def copy(self):
        return Quaternions(self.qs.copy())

    def repeat(self, repeats, axis=None):
        if axis is None:
            return Quaternions(np.repeat(self.qs.reshape(-1, 4), repeats, axis=0))
        # Existing MIT callers index/repeat with explicit component slices:
        # q[...,None,:] followed by axis=-2 repeats its joint axis.
        axis = int(axis)
        if not -self.qs.ndim <= axis < self.qs.ndim:
            raise ValueError('Quaternion repeat axis is out of range')
        axis %= self.qs.ndim
        if axis == self.qs.ndim - 1:
            raise ValueError('Repeat quaternion entries, not their component axis')
        return Quaternions(np.repeat(self.qs, repeats, axis=axis))

    def __neg__(self):
        squared_norm = np.sum(self.qs * self.qs, axis=-1, keepdims=True)
        if np.any(squared_norm == 0):
            raise ValueError('Zero quaternion has no inverse')
        return Quaternions(self.qs * np.array([1., -1., -1., -1.]) / squared_norm)

    def __mul__(self, other):
        if isinstance(other, Quaternions):
            a, b = np.broadcast_arrays(self.qs, other.qs)
            aw, av = a[..., :1], a[..., 1:]
            bw, bv = b[..., :1], b[..., 1:]
            return Quaternions(np.concatenate((aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
                                                aw * bv + bw * av + np.cross(av, bv)), axis=-1))
        vectors = np.asarray(other, dtype=np.float64)
        if vectors.ndim == 0 or vectors.shape[-1] != 3:
            raise ValueError('Quaternion multiplication expects quaternions or (...,3) vectors')
        q = self._unit_values()
        # The active quaternion-vector formula is Rodrigues' algebraic form.
        cross = 2 * np.cross(q[..., 1:], vectors)
        return vectors + q[..., :1] * cross + np.cross(q[..., 1:], cross)

    def _unit_values(self):
        norm = np.linalg.norm(self.qs, axis=-1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError('Zero quaternion cannot represent a rotation')
        return self.qs / norm

    def rotation_matrix(self, cont6d=False):
        w, x, y, z = np.moveaxis(self._unit_values(), -1, 0)
        matrix = np.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                           2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                           2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), axis=-1)
        matrix = matrix.reshape(self.shape + (3, 3))
        # Executing rotation_6d_to_matrix_np stacks decoded vectors as columns.
        return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1) if cont6d else matrix

    def transforms(self):
        """Rotation matrices required by the MIT mesh-animation consumer."""
        return self.rotation_matrix()

    @classmethod
    def id(cls, shape):
        shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        values = np.zeros(shape + (4,))
        values[..., 0] = 1
        return cls(values)

    @classmethod
    def from_euler(cls, angles, order='xyz'):
        """Intrinsic Euler sequence, radians: orderxyz means Rx @ Ry @ Rz.

        Current pilot callers use one nonzero X or Y angle only. Multi-axis
        order is explicit and independently checked rather than inferred from
        an unavailable library implementation.
        """
        angles = np.asarray(angles, dtype=np.float64)
        if angles.shape[-1:] != (3,) or len(order) != 3 or set(order) != set('xyz'):
            raise ValueError('Euler angles need (...,3) and an xyz-axis permutation')
        result = cls.id(angles.shape[:-1] or (1,))
        for i, axis in enumerate(order):
            half_angle = angles[..., i] / 2
            values = np.zeros(angles.shape[:-1] + (4,))
            values[..., 0] = np.cos(half_angle)
            values[..., 'xyz'.index(axis) + 1] = np.sin(half_angle)
            result = result * cls(values)
        return result

    @classmethod
    def between(cls, source, target):
        """Shortest active rotation from one nonzero vector to another.

        Antiparallel vectors use a deterministic orthogonal half-turn axis;
        the facing consumer explicitly chooses its ownYaxis half-turn where
        that domain convention matters. Undefined zero-vector input is refused.
        """
        source, target = np.broadcast_arrays(np.asarray(source, dtype=np.float64),
                                             np.asarray(target, dtype=np.float64))
        if source.shape[-1:] != (3,) or not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError('between requires finite (...,3) vectors')
        sn = np.linalg.norm(source, axis=-1, keepdims=True)
        tn = np.linalg.norm(target, axis=-1, keepdims=True)
        if np.any(sn == 0) or np.any(tn == 0):
            raise ValueError('Zero vector has no orientation')
        a, b = source / sn, target / tn
        dot = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1, 1)
        values = np.concatenate((1 + dot, np.cross(a, b)), axis=-1)
        # Exact/roundoff antiparallel: the ordinary construction vanishes.
        opposite = np.linalg.norm(values, axis=-1) <= 8 * np.finfo(np.float64).eps
        if np.any(opposite):
            basis = np.eye(3)[np.argmin(np.abs(a), axis=-1)]
            axis = np.cross(a, basis)
            axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
            values = np.where(opposite[..., None], np.concatenate((np.zeros(dot.shape), axis), axis=-1), values)
        return cls(values / np.linalg.norm(values, axis=-1, keepdims=True))
