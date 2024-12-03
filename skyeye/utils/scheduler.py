"""
This module contains copies of the main LR schedulers from Pytorch 1.0, as well as some additional schedulers
and utility code. This is mostly intended as a work-around for the bugs and general issues introduced in Pytorch 1.1
and should be reworked as soon as a proper (and stable) scheduler interface is introduced in Pytorch.
"""
import types
from bisect import bisect_right

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class MultiStepMultiGammaLR(LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (list): Multiplicative factor of learning rate decay wrt to the base LR
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=[0.1, 0.01])
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, milestones, gamma=[0.1], last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)

        if not len(list(milestones)) == len(list(gamma)):
            raise ValueError("Number of milestones should be the same as number of gammas")

        self.milestones = milestones
        self.gamma = [1] + gamma
        super(MultiStepMultiGammaLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma[int(bisect_right(self.milestones, self.last_epoch))] for base_lr in self.base_lrs]


class BurnInLR(LRScheduler):
    def __init__(self, base, steps, start):
        self.base = base
        self.steps = steps
        self.start = start
        super(BurnInLR, self).__init__(base.optimizer, base.last_epoch)

    def step(self, epoch=None):
        super(BurnInLR, self).step(epoch)

        # Also update epoch for the wrapped scheduler
        if epoch is None:
            epoch = self.base.last_epoch + 1
        self.base.last_epoch = epoch

    def get_lr(self):
        beta = self.start
        alpha = (1. - beta) / self.steps
        if self.last_epoch <= self.steps:
            return [base_lr * (self.last_epoch * alpha + beta) for base_lr in self.base_lrs]
        else:
            return self.base.get_lr()
