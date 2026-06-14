import torch
import torch.nn.functional as F


def kl_div_loss(pred, target):

    pred = pred.squeeze(1)
    target = target.squeeze(1)

    pred = F.softmax(pred.view(pred.size(0), -1), dim=1)
    target = target.view(target.size(0), -1)

    target = target / (target.sum(dim=1, keepdim=True) + 1e-8)

    loss = target * torch.log(
        (target + 1e-8) /
        (pred + 1e-8)
    )

    return loss.sum(dim=1).mean()


def cc_loss(pred, target):

    pred = pred.squeeze(1)
    target = target.squeeze(1)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    numerator = (pred * target).sum(dim=1)

    denominator = torch.sqrt(
        (pred.pow(2).sum(dim=1)) *
        (target.pow(2).sum(dim=1))
    )

    cc = numerator / (denominator + 1e-8)

    return cc.mean()


def nss_loss(pred, fixation):

    pred = pred.squeeze(1)
    fixation = fixation.squeeze(1)

    pred = pred.view(pred.size(0), -1)
    fixation = fixation.view(fixation.size(0), -1)

    pred = (
        pred - pred.mean(dim=1, keepdim=True)
    ) / (
        pred.std(dim=1, keepdim=True) + 1e-8
    )

    nss = (pred * fixation).sum(dim=1) / (
        fixation.sum(dim=1) + 1e-8
    )

    return nss.mean()


def sam_loss(pred, fixation):

    kl = kl_div_loss(pred, fixation)

    cc = cc_loss(pred, fixation)

    nss = nss_loss(pred, fixation)

    loss = (
        10 * kl
        - 2 * cc
        - 1 * nss
    )

    return loss