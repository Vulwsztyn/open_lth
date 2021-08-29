# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams, paths
import models.base
from pruning import base
from pruning.mask import Mask
import torch


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_on_test_or_train_gradient: str = 'test'
    pruning_by_max_or_min_gradient: str = 'max'
    pruning_layers_to_ignore: str = None
    _output_location: str = None

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

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        grad_location = {
            'test': paths.gradient_on_test(pruning_hparams._output_location),
            'train': paths.accumulated_training_gradient(pruning_hparams._output_location),
            'train_after': paths.gradient_on_train(pruning_hparams._output_location),
        }[pruning_hparams.pruning_on_test_or_train_gradient]
        grad_dict = torch.load(
            grad_location
        )
        # Get the model gradient.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in grad_dict.items()
                   if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        if pruning_hparams.pruning_by_max_or_min_gradient == 'max':
            threshold = -np.sort(-np.abs(weight_vector))[number_of_weights_to_prune]
            within_threshold = lambda x: x < threshold
        else:
            threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]
            within_threshold = lambda x: x > threshold

        new_mask = Mask({k: np.where(within_threshold(np.abs(v)), current_mask[k], np.zeros_like(v))
                         for k, v in weights.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
