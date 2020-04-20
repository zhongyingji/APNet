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
        self.down_dim = nn.Conv2d(2048, self.num_parts*self.feat_dim, 1, 1)

        
    def forward(self, x, query=False):
        n, c, h, w = x.shape

        down_x = self.down_dim(x)
        # n, 7*256, h, w
        x1 = down_x[:, :1*self.feat_dim, 0, :].view(n, self.feat_dim, 1, w)
        x2 = down_x[:, 1*self.feat_dim:2*self.feat_dim, 1, :].view(n, self.feat_dim, 1, w)
        x3 = down_x[:, 2*self.feat_dim:3*self.feat_dim, 2, :].view(n, self.feat_dim, 1, w)
        x4 = down_x[:, 3*self.feat_dim:4*self.feat_dim, 3, :].view(n, self.feat_dim, 1, w)
        x5 = down_x[:, 4*self.feat_dim:5*self.feat_dim, 4, :].view(n, self.feat_dim, 1, w)
        x6 = down_x[:, 5*self.feat_dim:6*self.feat_dim, 5, :].view(n, self.feat_dim, 1, w)
        x7 = down_x[:, 6*self.feat_dim:7*self.feat_dim, 6, :].view(n, self.feat_dim, 1, w)


        part_feat = []
        f1 = F.avg_pool2d(x1, x1.size()[2:]).view(x1.size(0), -1)
        f2 = F.avg_pool2d(x2, x2.size()[2:]).view(x2.size(0), -1)
        f3 = F.avg_pool2d(x3, x3.size()[2:]).view(x3.size(0), -1)
        f4 = F.avg_pool2d(x4, x4.size()[2:]).view(x4.size(0), -1)
        f5 = F.avg_pool2d(x5, x5.size()[2:]).view(x5.size(0), -1)
        f6 = F.avg_pool2d(x6, x6.size()[2:]).view(x6.size(0), -1)
        f7 = F.avg_pool2d(x7, x7.size()[2:]).view(x7.size(0), -1)

        f1_norm = f1.div(f1.norm(p=2, dim=1, keepdim=True).expand_as(f1))
        f2_norm = f2.div(f2.norm(p=2, dim=1, keepdim=True).expand_as(f2))
        f3_norm = f3.div(f3.norm(p=2, dim=1, keepdim=True).expand_as(f3))
        f4_norm = f4.div(f4.norm(p=2, dim=1, keepdim=True).expand_as(f4))
        f5_norm = f5.div(f5.norm(p=2, dim=1, keepdim=True).expand_as(f5))
        f6_norm = f6.div(f6.norm(p=2, dim=1, keepdim=True).expand_as(f6))
        f7_norm = f7.div(f7.norm(p=2, dim=1, keepdim=True).expand_as(f7))


        part_feat.append(f1_norm)
        part_feat.append(f2_norm)
        part_feat.append(f3_norm)
        part_feat.append(f4_norm)
        part_feat.append(f5_norm)
        part_feat.append(f6_norm)
        part_feat.append(f7_norm)



        x_reasmb = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2)
        # n, 256, h, w

        f = F.avg_pool2d(x_reasmb, x_reasmb.size()[2:]).view(x_reasmb.size(0), -1)
        # n, 256

        f_global_norm = f.div(f.norm(p=2, dim=1, keepdim=True).expand_as(f))


        part_feat.append(f_global_norm)


        return part_feat



def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PART_PREDICTOR]
    return func(cfg, in_channels)













