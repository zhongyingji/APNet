# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

class PostProcessor(nn.Module):

    def __init__(self):
        super(PostProcessor, self).__init__()


    def forward(self, reg_vals, boxes):

        boxes_per_image = [len(box) for box in boxes]
        reg_val = reg_vals.split(boxes_per_image, dim=0)

        results = []
        for reg, box in zip(
            reg_val, boxes
        ):
            box.add_field("reg_vals", reg)
            results.append(box)

        return results
      


def make_roi_box_post_processor(cfg):
    return PostProcessor()

