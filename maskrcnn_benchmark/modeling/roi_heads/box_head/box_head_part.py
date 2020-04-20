# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from .roi_box_feature_extractors_part import make_roi_box_feature_extractor
from .roi_box_predictors_part import make_roi_box_predictor
from .inference_part import make_roi_box_post_processor
from .loss_part import make_roi_box_loss_evaluator
from .completeness_part import Ratio

from maskrcnn_benchmark.layers import OIMLoss
from maskrcnn_benchmark.layers import OIMLoss_Part
from maskrcnn_benchmark.layers.estimator_loss import est_decode
from maskrcnn_benchmark.structures.bounding_box import BoxList



class ROIBoxHead_part(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead_part, self).__init__()


        num_part = cfg.MODEL.REID.NUM_PARTS
        
        self.oimloss = nn.ModuleList()
        for j in range(num_part):
            self.oimloss.append(OIMLoss_Part(cfg.MODEL.REID.FEAT_DIM, cfg.MODEL.REID.NUM_IDS))
        
        if cfg.MODEL.REID.RSFE:
            self.oimloss.append(OIMLoss(cfg.MODEL.REID.FEAT_DIM, cfg.MODEL.REID.NUM_IDS))

        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg, self.oimloss)

        self.ratio_estimator = Ratio(cfg.MODEL.REID.DISCARD_NOKP)
        self.query_by_gt = cfg.MODEL.REID.PADREG_QUERY_GT

        self.padreg = cfg.MODEL.BOX_PADREG_ON

    def forward(self, features_, proposals, targets=None, query=False):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        tmpp = [feature.shape[2:] for feature in features_]
        features = [F.pad(feature, (0, 0, 0, size[0]), "constant", value=0.0) for size, feature in zip(tmpp, features_)]
        
        if self.training:
            targets = self.ratio_estimator.get_ratio(targets, self.training)
            proposals = targets

        elif query:
            for target, proposal in zip(targets, proposals):
                target.add_field("embeds", proposal.get_field("embeds"))

            if self.padreg:
                for target, proposal in zip(targets, proposals):
                    target.add_field("reg_vals", proposal.get_field("reg_vals"))
                proposals = targets
                
                query_get_ratio = self.ratio_estimator.get_ratio \
                                if self.query_by_gt else self.ratio_estimator.get_ratio_by_est
                proposals = query_get_ratio(proposals, self.training)
            else:
                proposals = targets
            

        elif self.padreg:
            old_proposals = proposals
            device_ = proposals[0].bbox.device
            return_bboxlist = []

            for proposal in proposals:
                p_bbox = 1.0*proposal.bbox
                new_bbox = []
                n_proposal = p_bbox.shape[0]

                regvals = proposal.get_field("reg_vals")
                reg_vals = est_decode(regvals)
                # reg_valss = (reg_vals+1)/2

                for j in range(n_proposal):
                    bbox = p_bbox[j, :]
                    h = bbox[3] - bbox[1]
                    new_h = h*(1./(1.-reg_vals[j]))
                    bbox[3] = bbox[1] + new_h
                    new_bbox.append(bbox.tolist())
                if n_proposal == 0:
                    new_bbox = torch.tensor([]).view(0, 4)
                new_bboxlist = BoxList(new_bbox, proposal.size, mode="xyxy")
                new_bboxlist._copy_extra_fields(proposal)
                return_bboxlist.append(new_bboxlist)

            return_bboxlist = [return_box.to(device_) for return_box in return_bboxlist]
            proposals = return_bboxlist

        else:
            pass
            # keep the proposals
           

        x = self.feature_extractor(features, proposals)

        # final classifier that converts the features into predictions
        
        part_feat = self.predictor(x)

        if not self.training:

            # when no training
            # for query, proposals are ground truth
            # for gallery, proposals are results, just add part_embeds on it
            
            if not query and self.padreg:
                proposals = self.exchange_box(old_proposals, proposals)

            result = self.post_processor(part_feat, proposals)

            return x, result, {}


        loss_part_oim = self.loss_evaluator(part_feat, targets)

        loss_dict = dict(zip(["loss_reid_p"+str(i) for i in range(1, len(loss_part_oim)+1)], loss_part_oim))

        
        return (
            x,
            proposals,
            loss_dict,
        )

    def exchange_box(self, boxlists1, boxlists2):
        result = []
        for boxlist1, boxlist2 in zip(boxlists1, boxlists2):
            result_box = BoxList(boxlist1.bbox, boxlist1.size, mode="xyxy")
            result_box._copy_extra_fields(boxlist2)
            result.append(result_box)
        return result
        
        


def build_roi_box_head_part(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead_part(cfg, in_channels)
