# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .box_head.box_head_part import build_roi_box_head_part
from .box_head.box_head_padreg import build_roi_box_head_padreg
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, query=False):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        
        if not self.training or not (self.cfg.MODEL.REID.TRAIN_PART or self.cfg.MODEL.REID.TRAIN_PADREG):
            x, detections, loss_box = self.box(features, proposals, targets, query)
            losses.update(loss_box)
       

        if self.cfg.MODEL.BOX_PADREG_ON:
            loss_box_pad_reg = {}
            if self.training and self.cfg.MODEL.REID.TRAIN_PADREG:
                x, detections, loss_box_pad_reg = self.box_padreg(features, proposals, targets)
            elif not self.training:
                x, detections, loss_box_pad_reg = self.box_padreg(features, detections, targets, query)
            losses.update(loss_box_pad_reg)

        

        if self.cfg.MODEL.BOX_PART_ON:
            loss_box_part = {}
            if self.training and self.cfg.MODEL.REID.TRAIN_PART:
                x, detections, loss_box_part = self.box_part(features, proposals, targets)
                # proposals = targets in box_head_part.py
                # train the head with gt bboxes
            elif not self.training:
                # if testing
                # for query, use targets
                # for gallery, use detections
                x, detections, loss_box_part = self.box_part(features, detections, targets, query)
            losses.update(loss_box_part)


        
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
   
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                #pass
                keypoint_features = x
           
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            #print('keypoint proposals')
            #print(detections)
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        
        
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))

    if cfg.MODEL.BOX_PART_ON:
        roi_heads.append(("box_part", build_roi_box_head_part(cfg, in_channels)))
    if cfg.MODEL.BOX_PADREG_ON:
        roi_heads.append(["box_padreg", build_roi_box_head_padreg(cfg, in_channels)])

    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
