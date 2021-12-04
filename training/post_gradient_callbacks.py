from platforms.platform import get_platform
from foundations import paths
from datasets.base import DataLoader
from foundations import hparams
from foundations.step import Step
import torch
import os


def create_accumulated_training_gradient_callback(training_hparams: hparams.TrainingHparams, start: Step, end: Step):
    def accumulated_training_gradient_callback(output_location, step, model, optimizer, logger):
        should_save_gradient = training_hparams.epochs_to_accumulate_gradient_from is None or \
                               (end.ep - step.ep) <= training_hparams.epochs_to_accumulate_gradient_from
        if get_platform().is_primary_process and should_save_gradient:
            grad_dict = {x[0]: x[1].grad for x in model.model.named_parameters()}
            if os.path.isfile(paths.accumulated_training_gradient(output_location)):
                current_grad_dict = torch.load(paths.accumulated_training_gradient(output_location))
                grad_dict = {k: v + current_grad_dict[k] for k, v in grad_dict.items()}
            os.makedirs(paths.gradients(output_location), exist_ok=True)
            torch.save(grad_dict, paths.accumulated_training_gradient(output_location))

    return accumulated_training_gradient_callback


def post_gradient_callbacks(training_hparams: hparams.TrainingHparams, train_set_loader: DataLoader,
                            test_set_loader: DataLoader, eval_on_train: bool = False, verbose: bool = True,
                            start_step: Step = None, evaluate_every_epoch: bool = True):
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    end = Step.from_str(training_hparams.training_steps, train_set_loader.iterations_per_epoch)
    accumulated_training_gradient_callback = create_accumulated_training_gradient_callback(training_hparams, start, end)
    result = [
        accumulated_training_gradient_callback,
    ]
    return result
