# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import torch
from torch import nn
import torch.nn.functional as F


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor_part")
class FastRCNNPredictor_part(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor_part, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        self.num_parts = config.MODEL.REID.NUM_PARTS
        self.feat_dim = config.MODEL.REID.FEAT_DIM

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.ModuleList()
        self.fc.extend([nn.Linear(2048, self.feat_dim) for _ in range(self.num_parts)])
        self.fc_global = nn.Linear(2048, self.feat_dim)
        

        
    def forward(self, x, query=False):

        n, c, h, w = x.shape

        part_feat = []
        for j in range(self.num_parts):
            f = self.avgpool(x[:, :, j:(j+1), :]).view(n, -1)
            f = self.fc[j](f)
            f = f.div(f.norm(p=2, dim=1, keepdim=True).expand_as(f))
            part_feat.append(f)

        fg = self.fc_global(self.avgpool(x).view(n, -1))
        fg = fg.div(fg.norm(p=2, dim=1, keepdim=True).expand_as(fg))
        part_feat.append(fg)


        return part_feat



def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PART_PREDICTOR]
    return func(cfg, in_channels)













