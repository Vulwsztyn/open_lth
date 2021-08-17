# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning import gradient_global
from pruning.mask import Mask
from foundations import paths
import torch

PruningHparams = gradient_global.PruningHparams


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        new_mask = Mask(current_mask)

        grad_location = paths.gradient_on_test(
            pruning_hparams._output_location) if pruning_hparams.pruning_on_test_or_train_gradient == 'test' else paths.accumulated_training_gradient(
            pruning_hparams._output_location)
        grad_dict = torch.load(
            grad_location
        )

        for k, v in current_mask.items():

            if pruning_hparams.pruning_layers_to_ignore and k in pruning_hparams.pruning_layers_to_ignore:
                continue

            # Determine the number of weights that need to be pruned.
            number_of_remaining_weights = np.sum(v)
            number_of_weights_to_prune = np.ceil(
                pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

            weights = grad_dict[k].clone().cpu().detach().numpy()

            weight_vector = weights[current_mask[k] == 1]
            if pruning_hparams.pruning_by_max_or_min_gradient == 'max':
                threshold = -np.sort(-np.abs(weight_vector))[number_of_weights_to_prune]
                within_threshold = lambda x: x <= threshold
            else:
                threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]
                within_threshold = lambda x: x >= threshold

            new_mask[k] = np.where(within_threshold(np.abs(weights)), current_mask[k], np.zeros_like(v))

        return new_mask
