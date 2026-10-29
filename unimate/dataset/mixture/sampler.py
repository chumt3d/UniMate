"""Power-law mixture sampling, independent of motion decoding and text models."""

from collections import defaultdict

import numpy as np
import torch
from torch.utils import data

from unimate.utils.logger import get_logger

logger = get_logger(file_name=__file__)


class MixtureSampler(data.Sampler):
    """Power-law balanced sampling across datasets and skeleton types.

    Two levels. Dataset d with N_d motions gets total probability
    ∝ N_d^(1-dataset_alpha); inside it, object type k with n_k motions gets a
    share ∝ n_k^(1-alpha), split evenly over its motions.

      alpha=0.0 : uniform per sample within the dataset (data-rich types dominate)
      alpha=0.5 : square-root balanced (default — good compromise)
      alpha=1.0 : uniform per type (ignores motion count entirely)
      dataset_alpha=0.0 : each dataset keeps its natural share of the clips
      dataset_alpha=1.0 : equal probability per dataset
      dataset_weights={d: w}: explicit first level instead — dataset d gets
          total probability w_d / sum(w) whatever its size (a dataset with no
          training clips drops out and the rest are renormalized).
      dataset_alpha=None: single level (default) —
          object types pooled across all datasets. A dataset of one rig (Mixamo)
          then counts as one type and falls below 1% of a UniML3D mixture.

    With one dataset both levels coincide with the legacy behaviour.

    This is a plain Sampler (not DistributedSampler) that yields all indices.
    Accelerate's BatchSamplerShard handles per-rank sharding automatically.
    Each epoch draws with a generator seeded from ``seed`` and the epoch — the
    same on every rank, so the shards stay disjoint — so runs with different
    seeds see different draw orders (seed 0 gives the epoch-only order).
    """

    def __init__(self, data_source, alpha: float = 0.5, dataset_alpha=None,
                 dataset_weights=None, seed: int = 0):
        super().__init__(data_source)

        name_list = data_source.motion_dataset.train_name_list
        motion_dict = data_source.motion_dataset.train_motion_dict
        total_samples = len(name_list)

        # Keyed by (dataset_type, object_type): nothing stops two datasets from
        # using the same object-type name.
        indices_by_object = defaultdict(list)
        for i, name in enumerate(name_list):
            m = motion_dict[name]
            indices_by_object[(m['dataset_type'], m['object_type'])].append(i)
        objects_by_dataset = defaultdict(list)
        for key in indices_by_object:
            objects_by_dataset[key[0]].append(key)

        weights = np.zeros(total_samples)
        if dataset_alpha is not None and dataset_weights is not None:
            raise ValueError("Pass dataset_alpha or dataset_weights, not both")
        if dataset_weights is not None:
            missing = sorted(set(objects_by_dataset) - set(dataset_weights))
            if missing:
                raise ValueError(f"dataset_weights has no weight for {missing}")
            absent = sorted(set(dataset_weights) - set(objects_by_dataset))
            if absent:
                logger.warning(f'MixtureSampler: no training clips in {absent}; '
                               f'their weight is dropped and the rest renormalized')
        if dataset_alpha is None and dataset_weights is None:
            # w_i = n_k^(-alpha) for sample i in type k, giving type k total
            # probability ∝ n_k * n_k^(-alpha) = n_k^(1-alpha).
            for indices in indices_by_object.values():
                weights[indices] = len(indices) ** (-alpha)
        else:
            for d_type, keys in objects_by_dataset.items():
                n_d = sum(len(indices_by_object[k]) for k in keys)
                d_share = (float(dataset_weights[d_type]) if dataset_weights is not None
                           else n_d ** (1 - dataset_alpha))
                type_mass = {k: len(indices_by_object[k]) ** (1 - alpha) for k in keys}
                z = sum(type_mass.values())
                for k in keys:
                    indices = indices_by_object[k]
                    weights[indices] = d_share * type_mass[k] / z / len(indices)

        weights /= weights.sum()
        dataset_probs = {d: sum(weights[indices_by_object[k]].sum() for k in keys)
                         for d, keys in objects_by_dataset.items()}
        level1 = (f'dataset_weights={dataset_weights}' if dataset_weights is not None
                  else f'dataset_alpha={dataset_alpha}')
        logger.info(f'MixtureSampler (alpha={alpha}, {level1}): '
                    + ', '.join(f'{d} n={sum(len(indices_by_object[k]) for k in keys)} '
                                f'p={dataset_probs[d]:.3f}'
                                for d, keys in objects_by_dataset.items()))
        type_probs = {
            object_type: weights[indices].sum()
            for object_type, indices in indices_by_object.items()
        }
        sorted_types = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)
        logger.info(f'MixtureSampler: {len(sorted_types)} object types')
        for name, prob in sorted_types[:5]:
            logger.info(f'  top: {name[1]} ({name[0]}) — n={len(indices_by_object[name])}, p={prob:.4f}')
        if len(sorted_types) > 10:
            logger.info('  ...')
        for name, prob in sorted_types[-5:]:
            logger.info(f'  bot: {name[1]} ({name[0]}) — n={len(indices_by_object[name])}, p={prob:.4f}')

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.total_samples = total_samples
        self.seed = seed
        self.epoch = 0
        self.data_source = data_source
        self.generator = torch.Generator()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = self.generator
        # Spread seeds apart so seed s at epoch e never replays seed s + 1;
        # wrapped to 63 bits, which manual_seed accepts for any seed.
        g.manual_seed((self.seed * 1_000_003 + self.epoch) % (1 << 63))
        indices = torch.multinomial(
            self.weights, self.total_samples, replacement=True, generator=g
        ).tolist()
        return iter(indices)

    def __len__(self):
        return self.total_samples

