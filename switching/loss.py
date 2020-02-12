import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    print(index)
    return mask.scatter_(1, index, ones)


# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class FocalLossWithOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()


class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=1.0, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = input.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss

        return loss

class SelectKLLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SelectKLLoss, self).__init__()
        self.eps=eps
        self.softmax = nn.Softmax(dim=2)
        self.kldiv = nn.KLDivLoss(reduction='none')
        self.switch_weight = 0.9

    def forward(self, select_probs, gt_switch):
        print(select_probs.size())
        seq_num = select_probs.size()[1]
        cam_num = select_probs.size()[2]
        select_probs = self.softmax(select_probs)
        prev_prob = select_probs[:, :-1, :].contiguous().view(-1, cam_num)
        next_prob = select_probs[:, 1: , :].contiguous().view(-1, cam_num)
        kl_div = self.kldiv(torch.log(next_prob), prev_prob).view(-1, seq_num - 1)
        switch_loss = (1.0 - self.switch_weight) * (1.0 - gt_switch) * kl_div - self.switch_weight * gt_switch * kl_div
        switch_loss = switch_loss ** 2
        return switch_loss.sum(dim=1)


class SwitchingLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SwitchingLoss, self).__init__()
        self.eps=eps

    def forward(self, preds_indices, gt_switch):
        prev_indices_pred = preds_indices[:, :-1]
        next_indices_pred = preds_indices[:, 1: ]
        diff = torch.abs(next_indices_pred - prev_indices_pred)
        switched = diff / (diff + self.eps)
        switch_loss = gt_switch * (switched - 1.0) ** 2 + \
            (1.0 - gt_switch) * switched ** 2
        return switch_loss.sum(dim=1)

if __name__ == "__main__":
    device = torch.device("cuda")
    '''
    focal_without_onehot = FocalLossWithOutOneHot(gamma=1)
    focal_with_onehot = FocalLossWithOneHot(gamma=1)
    input = torch.Tensor([[0.3, 0.1, 0.1], [0.3, 0.6, 0.001], [0.01, 0.002, 2.3], [0.01, 0.002, 2.3]]).to(device)
    target = torch.Tensor([0, 1, 1, 2]).long().to(device)

    focal_without_onehot(input, target)
    # exception will occur when input and target are stored to GPU(s).
    focal_with_onehot(input, target)
    '''

    switch_loss = SwitchingLoss()
    #gt_switch = torch.Tensor([0, 0, 0, 0], [])
    _indices = torch.Tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 2, 1, 1], [1, 1, 2, 1, 2], [1, 2, 1, 3, 1]])
    print(switch_loss(_indices))
