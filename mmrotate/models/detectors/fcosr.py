# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class FCOSR(RotatedSingleStageDetector):
    """Implementation of `FCOSR: A Simple Anchor-free Rotated Detector for Aerial Object Detection`__
    
    https://arxiv.org/abs/2111.10780 
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FCOSR, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, gt_bboxes, gt_labels)
        return losses