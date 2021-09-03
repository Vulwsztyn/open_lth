# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import vectorize, unvectorize, shuffle_tensor, shuffle_state_dict
import datasets.registry
from foundations import hparams
from foundations.step import Step
from lottery.branch import base
import argparse
from cli import arg_utils
import pruning.registry
from pruning.sparse_global import PruningHparams
class Branch(base.Branch):
    def branch_function(self, reprune: hparams.PruningHparams,):
        trained_model = models.registry.load(self.level_root, self.lottery_desc.train_end_step, self.lottery_desc.model_hparams)
        untrained_model = models.registry.load(self.level_root, self.lottery_desc.train_start_step, self.lottery_desc.model_hparams)
        # Randomize the mask.
        mask = Mask.load(self.level_root)
        x = pruning.registry.get_pruning_hparams('sparse_global')('sparse_global', 0.8)
        if self.lottery_desc.model_hparams.model_name == 'mnist_lenet_300_100':
            x.pruning_layers_to_ignore = 'fc.weight'
        # Train the model with the new mask.
        new_mask = pruning.registry.get(x)(trained_model, mask)
        model = PrunedModel(untrained_model, new_mask)

        train.standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, verbose=self.verbose)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        pruning_strategy = arg_utils.maybe_get_arg('reprune_pruning_strategy')
        pruning_hparams = pruning.registry.get_pruning_hparams(pruning_strategy)
        pruning_hparams.add_args(parser, prefix='reprune')

    @staticmethod
    def description():
        return "Reprune the model."

    @staticmethod
    def name():
        return 'randomly_prune'
