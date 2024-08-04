# coding:utf-8
from torch.optim import Optimizer
from math import pi, cos


class WarmUpCosLRSchedule:
    """ Hot start Yuxian learning rate planner """

    def __init__(self, optimizer: Optimizer, lr: float, min_lr: float, total_epoch: int, warm_up_ratio=0.05, no_aug_ratio=0.05, warm_up_factor=1/3):
        """
        Parameters
        -----------
        Optimizer: Optimizer
            Optimizer

        LR: Float
            Initial learning rate

        min_lr: Float
            Learning rate at the end of the closing stage

        Total_iters: int
            A total of generations of iteration

        warm_up_ratio: int
            Hot start iteration ratio

        no_aug_ratio: Float
            Iteration ratio without mosaic data enhancement

        warm_up_factor: Float
            In the first iteration, the ratio of the learning rate and the initial learning rate
        """
        self.lr = lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.warm_up_factor = warm_up_factor
        self.total_epoch = total_epoch
        self.warm_up_epoch = int(warm_up_ratio*total_epoch)
        self.no_aug_epoch = int(no_aug_ratio*total_epoch)

    def step(self, epoch: int):
        """ Adjust the learning rate of the optimizer """
        if epoch < self.warm_up_epoch:
            delta = (1 - self.warm_up_factor) * epoch / self.warm_up_epoch
            lr = (self.warm_up_factor + delta) * self.lr
        elif epoch >= self.total_epoch-self.no_aug_epoch:
            lr = self.min_lr
        else:
            cos_iters = self.total_epoch - self.warm_up_factor - self.no_aug_epoch
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1 + cos(pi * (epoch - self.warm_up_epoch) / cos_iters)
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def set_lr(self, lr, min_lr):
        """ Setting learning rate """
        self.lr = lr
        self.min_lr = min_lr


def get_lr(optimizer):
    """ Get the current learning rate """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def determin_lr(lr, batch_size):
    """ Calculate learning rate based on batch size """
    bs = 64
    lr_max = 5e-2
    lr_min = 5e-4
    lr_fit = min(max(batch_size/bs*lr, lr_min), lr_max)
    lr_min_fit = min(max(batch_size/bs*(lr/100), lr_min/100), lr_max/100)
    return lr_fit, lr_min_fit
