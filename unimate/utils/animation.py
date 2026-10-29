# SPDX-License-Identifier: MIT
"""Local-translation Animation/FK operations used by the UniMate pilot.

This is a scoped NumPy implementation of MIT consumer contracts, not a clone
of the restricted Motion package. ``orients`` is carried metadata: the MIT
export/feature consumers already include actual rest/animated orientation in
``rotations``, and their independent kinematics.fk does not multiply orients.
"""

import numpy as np

from .quaternions import Quaternions


def _parent_order(parents):
    parents = np.asarray(parents)
    if parents.ndim != 1 or parents.dtype.kind not in 'iu':
        raise ValueError('Parents must be a1D integer array')
    count = len(parents)
    if np.any(parents < -1) or np.any(parents >= count):
        raise ValueError('Parent index is outside the skeleton')
    order = []
    state = np.zeros(count, dtype=np.uint8)
    for first in range(count):
        current = first
        path = []
        while current != -1 and state[current] == 0:
            state[current] = 1
            path.append(current)
            current = int(parents[current])
        if current != -1 and state[current] == 1:
            raise ValueError('Skeleton contains a parent cycle')
        for joint in reversed(path):
            state[joint] = 2
            order.append(joint)
    return order


class Animation:
    """Frames of parent-relative positions/rotations plus immutable topology.

    Static joint metadata and full animated translations are retained. Slicing
    can reorder or select closed subtrees; removing an ancestor requires an
    explicit transform composition by the caller rather than silent reparenting.
    """

    def __init__(self, rotations, positions, orients, offsets, parents):
        self.rotations = rotations if isinstance(rotations, Quaternions) else Quaternions(rotations)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.orients = orients if isinstance(orients, Quaternions) else Quaternions(orients)
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.parents = np.asarray(parents)
        _parent_order(self.parents)
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError('Animation positions require (F,J,3)')
        if self.rotations.shape != self.positions.shape[:2]:
            raise ValueError('Animation rotations/positions have different frame/joint axes')
        if self.offsets.shape != (len(self.parents), 3) or self.positions.shape[1] != len(self.parents):
            raise ValueError('Animation topology and array widths differ')
        if not np.isfinite(self.positions).all() or not np.isfinite(self.offsets).all():
            raise ValueError('Animation translations must be finite')

    @property
    def shape(self):
        return self.positions.shape[:2]

    def __len__(self):
        return self.shape[0]

    def copy(self):
        return Animation(self.rotations.copy(), self.positions.copy(), self.orients.copy(),
                         self.offsets.copy(), self.parents.copy())

    def __getitem__(self, key):
        frame, joint = key if isinstance(key, tuple) else (key, slice(None))
        # Sequential indexing gives the Cartesian frame/joint slice when both
        # selectors are arrays; NumPy's simultaneous fancy indexing would pair.
        frames = np.atleast_1d(np.arange(self.shape[0])[frame])
        joints = np.atleast_1d(np.arange(self.shape[1])[joint])
        if len(set(joints.tolist())) != len(joints):
            raise ValueError('A skeleton slice cannot duplicate joints')
        lookup = {int(old): new for new, old in enumerate(joints)}
        parents = []
        for old in joints:
            parent = int(self.parents[old])
            if parent != -1 and parent not in lookup:
                raise ValueError('Selected joint lost its parent; compose transforms before pruning')
            parents.append(-1 if parent == -1 else lookup[parent])
        orientations = self.orients.copy()
        if self.orients.shape == (self.shape[1],):
            orientations = self.orients[joints]
        elif self.orients.shape == self.shape:
            orientations = Quaternions(self.orients.qs[frames][:, joints])
        # Empty orients and generated per-frame facing metadata do not enter FK.
        return Animation(Quaternions(self.rotations.qs[frames][:, joints]),
                         self.positions[frames][:, joints], orientations,
                         self.offsets[joints], np.asarray(parents, dtype=np.int64))


def transforms_local(animation):
    """Homogeneous active transforms, including every animated local position."""
    result = np.zeros(animation.shape + (4, 4))
    result[..., :3, :3] = animation.rotations.rotation_matrix()
    result[..., :3, 3] = animation.positions
    result[..., 3, 3] = 1
    return result


def transforms_global(animation):
    result = transforms_local(animation)
    for joint in _parent_order(animation.parents):
        parent = int(animation.parents[joint])
        if parent >= 0:
            result[:, joint] = result[:, parent] @ result[:, joint]
    return result


def positions_global(animation):
    """Forward positions by Hamilton composition and rotated local translation."""
    result = np.empty_like(animation.positions)
    rotations = animation.rotations.copy()
    for joint in _parent_order(animation.parents):
        parent = int(animation.parents[joint])
        if parent < 0:
            result[:, joint] = animation.positions[:, joint]
        else:
            result[:, joint] = result[:, parent] + rotations[:, parent] * animation.positions[:, joint]
            rotations[:, joint] = rotations[:, parent] * animation.rotations[:, joint]
    return result


def rotations_global(animation):
    result = animation.rotations.copy()
    for joint in _parent_order(animation.parents):
        parent = int(animation.parents[joint])
        if parent >= 0:
            result[:, joint] = result[:, parent] * animation.rotations[:, joint]
    return result


def offsets_from_positions(positions, parents):
    """World-frame rest offsets used by MIT rest-relative feature construction."""
    positions = np.asarray(positions, dtype=np.float64)
    parents = np.asarray(parents)
    _parent_order(parents)
    if positions.shape[-2:] != (len(parents), 3):
        raise ValueError('Positions and parent topology disagree')
    offsets = positions.copy()
    children = np.flatnonzero(parents >= 0)
    offsets[..., children, :] -= positions[..., parents[children], :]
    return offsets


def offset_lengths(animation):
    return np.linalg.norm(animation.offsets, axis=-1)
