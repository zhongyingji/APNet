# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


class PostProcessor(nn.Module):

    def __init__(self):
        super(PostProcessor, self).__init__()

    def forward(self, part_feat, boxes):
        # part_feat [N*256, N*256, ...]
        # vis_conf N*n_part

        n_part = len(part_feat)
        feat = torch.cat(part_feat, dim=1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        embed_norm = feat.split(boxes_per_image, dim=0)
        
        results = []
        for embed, box in zip(
            embed_norm, boxes
        ):
        # id
            box.add_field("part_embeds", embed)    
            results.append(box)

        return results




def make_roi_box_post_processor(cfg):
    return PostProcessor()
    
