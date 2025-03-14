import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as tmp
from functools import partial
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader as DataLoader
from focal_frequency_loss import FocalFrequencyLoss as FFL
from torch.autograd import Variable
import torch.nn.functional as F

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_dist(local_rank=0, backend='nccl'):
    tmp.set_start_method('spawn')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    return rank, world_size


# ==========
# Dataloader
# ==========


class DistSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    
    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(
            self, dataset, num_replicas=None, rank=None, ratio=1
            ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # enlarged by ratio, and then divided by num_replicas
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on ite_epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        # enlarge indices
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]
        

        # ==========subsample
        # e.g., self.rank=1, self.total_size=4, self.num_replicas=2
        # indices = indices[1:4:2] = indices[i for i in [1, 3]]
        # for the other worker, indices = indices[i for i in [0, 2]]
        # ==========
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # print("[indices]",indices)
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """For shuffling data at each epoch. See train.py."""
        self.epoch = epoch


def create_dataloader(
        dataset, opts_dict, sampler=None, phase='train', seed=None
        ):
    """Create dataloader."""
    if phase == 'train':
        # >I don't know why BasicSR have to detect `is_dist`
        dataloader_args = dict(
            dataset=dataset,
            batch_size=opts_dict['dataset']['train']['batch_size_per_gpu'],
            shuffle=False,  # sampler will shuffle at train.py
            num_workers=opts_dict['dataset']['train']['num_worker_per_gpu'],
            sampler=sampler,
            drop_last=True,
            pin_memory=True
            )
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            _worker_init_fn, 
            num_workers=opts_dict['dataset']['train']['num_worker_per_gpu'], 
            rank=opts_dict['train']['rank'],
            seed=seed
            )
        
    elif phase == 'val':
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
            )
    
    return DataLoader(**dataloader_args)


def _worker_init_fn(worker_id, num_workers, rank, seed):
    # func for torch.utils.data.DataLoader
    # set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========
# Loss & Metrics
# ==========

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6,type=None):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class MSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6,type=None):
        super(MSELoss, self).__init__()
        self.eps = eps
        loss = nn.MSELoss()
    def forward(self, X, Y):
        return loss(X, Y)



class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6,type=None):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss



class MultiOFR_loss(torch.nn.Module):
    def __init__(self, eps=1e-6,type=None):
        super(MultiOFR_loss, self).__init__()
        self.eps = eps
        # self.warp = optical_flow_warp()

    def forward(self, x_data, optical_flow):
        # print('optical_flow',x_data.shape, optical_flow.shape)
        b, T, h, w = x_data.shape
        b, T, mc, h, w = optical_flow.shape
        sum_loss = 0
        for i in range(T-1):
            x0 = x_data[:,i,...].unsqueeze(1).cuda()
            x1 = x_data[:,i+1,...].unsqueeze(1).cuda()
            flow_0 = optical_flow[:,i,...].cuda()
            # print('optical_flow',x0.shape, flow_0.shape)
            warped = optical_flow_warp(x0, flow_0)
            # print('optical_flow',x0.shape, x1.shape,flow_0.shape, warped.is_cuda)
            sum_loss = sum_loss + torch.mean(torch.abs(x1 - warped)) + 0.1 * L1_regularization(flow_0)

        return sum_loss




class OFR_loss(torch.nn.Module):
    def __init__(self, eps=1e-6,type=None):
        super(OFR_loss, self).__init__()
        self.eps = eps
        # self.L1 = L1_regularization()

    def forward(self, x0, x1, optical_flow):
        warped = optical_flow_warp(x0, optical_flow)
        loss = torch.mean(torch.abs(x1 - warped)) + 0.1 * L1_regularization(optical_flow)

        return loss


# def OFR_loss(x0, x1, optical_flow):
#     warped = optical_flow_warp(x0, optical_flow)
#     loss = torch.mean(torch.abs(x1 - warped)) + 0.1 * L1_regularization(optical_flow)
#     return loss


def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 1:, 0:w-1]
    reg_y_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 0:h-1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b*(h-1)*(w-1))


def optical_flow_warp(image, image_optical_flow):
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b, 2, h, w)
    """
    b, _ , h, w = image.size()
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))
    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda()

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 31 / (w - 1), dim=1)
    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 31 / (h - 1), dim=1)
    grid = grid + torch.cat((flow_0, flow_1),1)
    grid = grid.transpose(1, 2)
    grid = grid.transpose(3, 2)
    output = F.grid_sample(image, grid, padding_mode='border')
    return output



class Focal_Frequecny_Loss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(Focal_Frequecny_Loss, self).__init__()
        self.eps = eps
        self.ffl = FFL(loss_weight=1.0, alpha=1.0)  

    def forward(self, X, Y):
        # diff = torch.add(X, -Y)
        # error = torch.sqrt(diff * diff + self.eps)
        ffloss = self.ffl(X,Y)
        return loss



class Charbonnier_FFL_Loss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(Charbonnier_FFL_Loss, self).__init__()
        self.eps = eps
        self.ffl = FFL(loss_weight=1.0, alpha=1.0)

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        cb_loss = torch.mean(error)
        # print("[x]",X.shape)
        # print("[y]",Y.shape)
        ffl_loss = self.ffl(X.unsqueeze(0),Y.unsqueeze(0))
        loss = cb_loss + ffl_loss
        return loss



class PSNR(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(PSNR, self).__init__()
        self.mse_func = nn.MSELoss()

    def forward(self, X, Y):
        # print("{X Y }",X.shape,Y.shape)
        mse = self.mse_func(X, Y)
        psnr = 10 * math.log10(1 / mse.item())
        return psnr


# ==========
# Scheduler
# ==========


import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """ MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=[0],
                 restart_weights=[1],
                 last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=[1],
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
