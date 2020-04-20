# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from .roi_box_feature_extractors_padreg import make_roi_box_feature_extractor
from .roi_box_predictors_padreg import make_roi_box_predictor
from .inference_padreg import make_roi_box_post_processor
from .loss_padreg import make_roi_box_loss_evaluator
from .completeness_reg import Ratio

from maskrcnn_benchmark.layers import EstLoss


class ROIBoxHead_padreg(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead_padreg, self).__init__()

        self.regcrit = EstLoss(loss="smooth_l1")
        
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg, self.regcrit)

        self.ratio_estimator = Ratio(cfg.MODEL.REID.PADREG_RAND_CUT, cfg.MODEL.REID.PADREG_AUG_PER)

    def forward(self, features, proposals, targets=None, query=False):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subs ampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            targets = self.ratio_estimator.get_ratio(targets, self.training)
            proposals = targets

            # for query
            # in generalized_rcnn.py
            # proposals is targets_cpy
        elif query:
            for target, proposal in zip(targets, proposals):
                target.add_field("embeds", proposal.get_field("embeds"))
            proposals = targets

            # ! exception
            # this time the gallery proposals are ground truth
            # copying the keypoints

            # proposals = self.ratio_estimator.get_ratio(proposals, self.training)


        # for query
        # both "reg_vals" and "pad_ratio"

        # for gallery
        # only "reg_vals"

        

        x = self.feature_extractor(features, proposals)
       

        reg_vals = self.predictor(x)
        

        if not self.training:

            # when no training
            # for query, proposals are ground truth
            # for gallery, proposals are results, just add part_embeds on it
            
            result = self.post_processor(reg_vals, proposals)
            return x, result, {}


          
        loss_reg = self.loss_evaluator(reg_vals, targets)

        
        return (
            x,
            proposals,
            dict(loss_reg=loss_reg) ,
        )
        
        


def build_roi_box_head_padreg(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead_padreg(cfg, in_channels)
