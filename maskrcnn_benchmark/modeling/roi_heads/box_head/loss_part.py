# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import numpy as np

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        regionsensitive,
        oim
    ):  
        self.rsfe = regionsensitive
        self.oimloss = oim

        self.w_part_feat = [0.1] * (len(self.oimloss)-1)
        self.w_part_feat.append(1)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets

        #target = target.copy_with_fields("labels")

        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        # including, label, id
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        person_ids = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            ids_per_image = matched_targets.get_field("ids")
            ids_per_image = ids_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            ids_per_image[bg_inds] = -3
            ids_per_image[ignore_inds] = -3
            # -2 with no id

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            person_ids.append(ids_per_image)

        return labels, regression_targets, person_ids

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.
        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, person_ids = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)


        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, person_ids_per_image, proposals_per_image in zip(
            labels, regression_targets, person_ids, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field("ids", person_ids_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image



        self._proposals = proposals
        return proposals


    def __call__(self, part_feat, targets):


        # N*n_part*7*7

        assert len(part_feat) == len(self.oimloss)


        boxes_per_img = [len(target) for target in targets]
        image_shapes = [target.size for target in targets]

        ids = cat([target.get_field("ids") for target in targets], dim=0)

        ids = ids.long()

        pad_ratio = cat([target.get_field("pad_ratio") for target in targets], dim=0)

        part_losses = []
        for j in range(len(part_feat)-1):
            part_losses.append(self.w_part_feat[j]*(self.oimloss[j](part_feat[j], ids.clone().detach(), pad_ratio.detach(), j+1))[0])
        part_losses.append(self.w_part_feat[-1]*(self.oimloss[-1](part_feat[-1], ids.clone().detach()))[0])


        return part_losses


def make_roi_box_loss_evaluator(cfg, oim):
    regionsensitive = cfg.MODEL.REID.RSFE
    loss_evaluator = FastRCNNLossComputation(
        regionsensitive,
        oim
    )

    return loss_evaluator
