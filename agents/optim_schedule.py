#!usr/bin/env python
# -*- coding:utf-8 -*-
import functools
from math import sqrt
from torch import optim
from transformers import AdamW


def _get_optimizer(model, opt):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon)


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    @classmethod
    def add_cmdline_args(cls, argparser):
        # super(SoftMaskedBertTrainer, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Optim Arguments')
        # add_common_cmdline_args(agent)
        # memory and knowledge arguments

        agent.add_argument("--decay_method", type=str,
                           choices=['linear', 'noam', 'noamwd', 'None'], default='None')
        agent.add_argument("--warmup_steps", default=0, type=int)
        agent.add_argument("--weight_decay", default=0.01, type=float)
        agent.add_argument("--adam_epsilon", default=1e-8, type=float)
        agent.add_argument("--start_decay_steps", default=None, type=str)

    def __init__(self, optim_opt, optimizer, num_training_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = optim_opt.warmup_steps
        self.init_lr = optim_opt.learning_rate
        self.lr = optim_opt.learning_rate
        self._learning_rate_decay_fn = make_learning_rate_decay_fn(optim_opt, num_training_steps)
        self.training_step = 1
        self._decay_step = 1

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()
        self._decay_step += 1
        self.training_step += 1

    # def _get_lr_scale(self):
    #     # if self._learning_rate_decay_fn is None:
    #     #     return self.init_lr
    #     # return self._learning_rate_decay_fn(self._decay_step)
    #     return np.min([
    #         np.power(self.n_current_steps, -0.5),
    #         np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def get_lr(self):
        return self.lr

    def set_lr(self, lr):
        self.lr = lr

    def learning_rate(self):
        """Returns the current learning rate."""
        if self._learning_rate_decay_fn is None:
            return self.lr
        scale = self._learning_rate_decay_fn(self._decay_step)
        self.lr = scale * self.init_lr
        return self.lr

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        lr = self.learning_rate()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def make_learning_rate_decay_fn(opt, num_training_steps):
    """Returns the learning decay function from options."""
    if opt.decay_method == "linear":
        return functools.partial(
            linear_decay,
            warmup_steps=opt.warmup_steps,
            num_training_steps=num_training_steps)
    elif opt.decay_method == 'noam':
        return functools.partial(
            noam_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)
    elif opt.decay_method == 'noamwd':
        return functools.partial(
            noamwd_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps)
    elif opt.decay_method == 'rsqrt':
        return functools.partial(
            rsqrt_decay, warmup_steps=opt.warmup_steps)
    elif opt.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=opt.learning_rate_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps)
    else:
        return None


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return (
            model_size ** (-0.5) *
            min(step ** (-0.5), step * warmup_steps ** (-1.5)))


def noamwd_decay(step, warmup_steps,
                 model_size, rate, decay_steps, start_step=0):
    """Learning rate schedule optimized for huge batches
    """
    return (
            model_size ** (-0.5) *
            min(step ** (-0.5), step * warmup_steps ** (-1.5)) *
            rate ** (max(step - start_step + decay_steps, 0) // decay_steps))


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))


def linear_decay(step, warmup_steps, num_training_steps, stop_steps=0):
    """Decay based on the reciprocal of the step."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return max(0.0, float(max(num_training_steps - step, stop_steps)) / float(max(1, num_training_steps - warmup_steps)))
