"""
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):
    def __init__(self, lut, cq, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.cq = cq
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        #print(inputs.shape, self.lut.shape)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.cq.t())

        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([self.lut, self.cq], dim=0))

        for x, y in zip(inputs, targets):
            if y < len(self.lut):
                self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
                self.lut[y] /= self.lut[y].norm()
            else:
                tmp = torch.cat([self.cq[1:], x.view(-1, x.size(0))], dim=0)
                self.cq[:, :] = tmp[:, :]
        return grad_inputs, None


def oim(inputs, targets, lut, cq, momentum=0.5):
    return OIM(lut, cq, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, num_unlabeled=5000, scalar=30, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_unlabeled = num_unlabeled
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight if weight is not None else \
            torch.cat([torch.ones(num_classes), torch.zeros(num_unlabeled)]).cuda()

        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(num_classes, num_features))
        self.register_buffer('cq', torch.zeros(num_unlabeled, num_features))

    def forward(self, inputs, targets):
        unlab = targets < 0
        targets[unlab] = torch.randint(low=len(self.lut), high=len(self.lut) + self.num_unlabeled,
                                       size=(unlab.sum().item(),)).long().cuda()
        # targets for unblabled sample is assigned a random integer between num_classes and num_classes+num_unlabeled
        inputs = oim(inputs, targets, self.lut, self.cq, momentum=self.momentum)
        inputs *= self.scalar


        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               size_average=self.size_average)

        return loss, inputs


"""

import torch
import torch.nn.functional as F
from torch import nn, autograd

class OIM(autograd.Function):

    def __init__(self, lut, cq, header, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.cq = cq
        self.header = header
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([self.lut, self.cq], dim=0))

        for x, y in zip(inputs, targets):
            if y < len(self.lut) and y >= 0:
                self.lut[y] = self.momentum * \
                    self.lut[y] + (1. - self.momentum) * x
                self.lut[y] /= self.lut[y].norm()
            elif y == 5555:
                self.cq[self.header] = x
                self.header = (self.header + 1) % self.cq.size(0)
            elif y == 7777:
                pass
            else:
                print('error')

        return grad_inputs, None

def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM(lut, cq, header, momentum=momentum)(inputs, targets)

class OIMLoss_Part(nn.Module):

    def __init__(self, num_features, num_labeled_pids, cq_size=5000, scalar=30.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss_Part, self).__init__()
        self.num_features = num_features
        self.num_labeled_pids = num_labeled_pids # 482 for PRW
        self.cq_size = cq_size
        self.momentum = momentum
        self.scalar = scalar
        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(
            (self.num_labeled_pids, self.num_features)))
        self.register_buffer('cq',  torch.zeros(
            (self.cq_size, self.num_features)))

        self.header_cq = 0
        self.weight = torch.cat([torch.ones(self.num_labeled_pids),
                                 torch.zeros(self.cq_size)]).cuda()
        ## Unlabeled targets = 5555

    def forward(self, inputs, targets, pad_ratios, part_idx):
        
        n_part = 7
        vis_part = torch.ceil(n_part*(1-pad_ratios))
        invis = part_idx > vis_part

        
        
        unlab = targets < 0
        targets[unlab] =  5555 #5555


        targets[invis] = 7777

        """
        if targets.size(0) == 0:
            return torch.tensor(0, dtype=torch.float32).to(inputs.device), inputs
        """
        inputs = oim(inputs, targets, self.lut, self.cq,
                     self.header_cq, momentum=self.momentum)
        inputs *= self.scalar

        
        new_targets = targets.clone().detach()
        new_targets[invis] = 5555
        new_targets[unlab] = 5555

        loss = F.cross_entropy(inputs, new_targets, weight=self.weight,
                               size_average=self.size_average,
                               ignore_index=5555)

        self.header_cq = ((self.header_cq + (new_targets >= len(self.lut)
                                             ).long().sum().item()) % self.cq.size(0))
        return loss, inputs
        
