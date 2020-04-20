import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers import smooth_l1_loss


class EstLoss(nn.Module):
	def __init__(self, loss="smooth_l1"):
		super(EstLoss, self).__init__()
		if loss == "smooth_l1":
			self.loss = smooth_l1_loss
		elif loss == "l2":
			self.loss = l2_loss

	def forward(self, reg_vals, reg_gts):
		eff = ~(reg_gts==1)
		return self.loss(reg_vals[eff], self.mapping(reg_gts[eff]))

	def mapping(self, reg_gts):
		# mapped to [-1, 1]
		return est_encode(reg_gts)


def est_encode(reg_gts):
	return 2*reg_gts - 1

def est_decode(est_vals):
	return (est_vals+1) / 2


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):

    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def l2_loss(input, target, size_average=True):
	loss = torch.pow(input-target, 2)
	if size_average:
		return loss.mean()
	return loss.sum()