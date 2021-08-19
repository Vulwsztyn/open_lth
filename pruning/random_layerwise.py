# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning import sparse_global
from pruning.mask import Mask
from utils.tensor_utils import vectorize, unvectorize, shuffle_tensor, shuffle_state_dict
import torch

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
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        new_mask = Mask(current_mask)

        for k,v in current_mask.items():

            if pruning_hparams.pruning_layers_to_ignore and k in pruning_hparams.pruning_layers_to_ignore:
                continue

            # Determine the number of weights that need to be pruned.
            number_of_remaining_weights = np.sum(v)
            number_of_weights_to_prune = np.ceil(
                pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)


            layer_mask_as_vector = v.reshape(-1)
            unpruned_indexes = [i for i, x in enumerate(layer_mask_as_vector) if x == 1]
            indexes_to_prune = shuffle_tensor(torch.tensor(unpruned_indexes), seed=pruning_hparams.seed)[:number_of_weights_to_prune]
            for i in indexes_to_prune:
                layer_mask_as_vector[i] = 0
            
            new_mask[k] = layer_mask_as_vector.reshape(v.shape)

        return new_mask
