# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np
import torch

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask
from utils.tensor_utils import vectorize, unvectorize, shuffle_tensor


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None
    seed: int = 123

    _name = 'Hyperparameters for Sparse Global Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model) if current_mask is None else current_mask

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        mask_as_vector = vectorize(current_mask)
        unpruned_indexes = [i for i, x in enumerate(mask_as_vector) if x == 1]
        indexes_to_prune = shuffle_tensor(torch.tensor(unpruned_indexes), seed=pruning_hparams.seed)[:number_of_weights_to_prune]
        for i in indexes_to_prune:
            mask_as_vector[i] = 0

        new_mask = Mask(unvectorize(mask_as_vector))

        return new_mask
