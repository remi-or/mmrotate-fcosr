# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from itertools import repeat
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import nms_rotated
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from torch import Tensor

from mmrotate.core import build_assigner, multiclass_nms_rotated
from mmrotate.core.bbox.iou_calculators import build_iou_calculator
from mmrotate.core.bbox.transforms import obb2poly, poly2obb
from ..builder import ROTATED_HEADS, build_loss

INF = 100000000

# TODO : add keep scores and fix the rotated_test
# TODO : it seems that the predictions that aren't detections aren't
#        utilized for the loss computation. Then, why include them in
#        the final loss computation? Instead, using the regressed_mask, we
#        can filter out position on detection criteria and have them flattened.


def multiclass_nms_rotated_extra(
    multi_bboxes: Tensor,
    multi_scores: Tensor,
    score_thr: Tensor,
    nms: float,
    extra_nms: Optional[float] = None,
    max_num: int = -1,
    score_factors: Optional[Tensor] = None,
    return_inds: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """NMS for multi-class bboxes with an extra class-agnostic NMS on top.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        nms (float, optional): Config of the extra class-agnostic NMS. If left
            as None, no extra NMS is performed after the first multiclass NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        dets (Tensor): of shape (nb_dets, 6) where detections are boxes
            with score
        labels (Tensor): of shape (nb_dets) with the detections labels.
            Labels are 0-based.
        indices (Tensor | None): of shape (nb_dets) with the indices of
            detections if the return_inds flag is True, otherwise None
    """
    # apply the first multiclass NMS with no max_num
    first_nms_output = multiclass_nms_rotated(multi_bboxes, multi_scores,
                                              score_thr, nms, -1,
                                              score_factors, return_inds)
    dets, labels = first_nms_output[:2]
    indices = first_nms_output[2] if return_inds else None
    # apply the eventual second NMS still with no max_num
    if extra_nms is not None:
        # TODO: check necessity of .detach().clone()
        kept_indices = nms_rotated(dets[:, :-1].detach().clone(),
                                   dets[:, -1].detach().clone(), extra_nms)[1]
        dets, labels = dets[kept_indices], labels[kept_indices]
        indices = indices[kept_indices] if return_inds else None
    # enforce max_num
    if dets.size(0) > max_num:
        dets, labels = dets[:max_num], labels[:max_num]
        indices = indices[:max_num] if return_inds else None
    # return statement varies upon return_inds flag
    return dets, labels, indices


def discrete_rot90_rbboxes(num_rot: int, rbboxes: Tensor, w: int,
                           h: int) -> Tensor:
    """Rotates the given rbboxes by a multiple of 90 degrees.

    Args:
        num_rot (int): the number of times the rbboxes should be rotated
            by 90  degrees.
        rbboxes (Tensor): the rbboxes to rotate in the (cx, cy, w, h, angle)
            format. Shape: (nb_rbboxes, 5)
        w (int): the width of the image on which the rbboxes were predicted.
        h (int): the height of the image on which the rbboxes were predicted.

    Returns:
        (Tensor) the rotated rbboxes in the same format and same shape.
    """
    # since this function is for 90° rotations, force n to be in [0, 3]
    n = num_rot % 4
    if n == 0:
        return rbboxes
    polys = obb2poly(rbboxes)
    if n == 1:
        rotated_polys = torch.zeros_like(polys)
        rotated_polys[:, 0::2] = -polys[:, 1::2]
        rotated_polys[:, 1::2] = polys[:, 0::2]
        rotated_polys += torch.tensor([(w + h) / 2,
                                       (h - w) / 2]).repeat(4)[None]
    elif n == 2:
        rotated_polys = -polys
        rotated_polys += torch.tensor([w, h]).repeat(4)[None]
    elif n == 3:
        rotated_polys = torch.zeros_like(polys)
        rotated_polys[:, 0::2] = polys[:, 1::2]
        rotated_polys[:, 1::2] = -polys[:, 0::2]
        rotated_polys += torch.tensor([(w - h) / 2,
                                       (h + w) / 2]).repeat(4)[None]
    else:
        raise ValueError(f'Incompatible number of rotations {num_rot}')
    return poly2obb(rotated_polys)


@ROTATED_HEADS.register_module()
class FCOSRHead(BaseDenseHead):
    r"""A dense head used in `FCOSR
    <https://arxiv.org/ftp/arxiv/papers/2111/2111.10780.pdf>`_.

    The head contains two subnetworks, one for classification and one for
    regression of bounding boxes.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of feature channels.
        stacked_convs (int, optional): Number of stacked convolutions.
        strides (tuple[int], optional): Stride of each feature level.
        regress_ranges (tuple[int], optional): Regress ranges of each feature
            level.
        conv_cfg (dict, optional): Config dict for convolution layer.
        norm_cfg (dict, optional): Config dict for normalization layer.
        assigner (dict, optional): Config dict for the bbox assigner.
        cls_loss (dict, optional): Config dict for classification loss.
        cls_scores (str, optional): What score to use for classification loss.
        reg_loss (dict, optional): Config dict for regression loss.
        reg_weights (str, optional): What weights to use for regression loss.
        train_cfg (dict | None, optional): The training config dict.
        test_cfg (dict | None, optional): The training config dict.
        init_cfg (dict, optional): Initialization config dict.
    """

    # use sim ota, dcn on last conv, drop_positive_sample, use_qfl
    # TODO : support multiple loss functions
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512,
                                                                      INF)),
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        assigner=dict(
            type='GaussianAssigner',
            gauss_factor=12.0,
            inside_ellipsis_thresh=0.23,
            epsilon=1e-6),
        cls_loss=dict(
            type='QualityFocalLoss',
            beta=2.0,
            use_sigmoid=True,
            loss_weight=1.0),
        cls_scores='gauss',
        reg_loss=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        reg_weights='iou',
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='cls_logits_conv',
                std=0.01,
                bias_prob=0.01)),
    ) -> None:
        super(FCOSRHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.strides = nn.Parameter(
            torch.tensor(strides, dtype=torch.float32), requires_grad=False)
        self.regress_ranges = regress_ranges
        self.build_classification_subnetwork(stacked_convs, conv_cfg, norm_cfg)
        self.build_regression_subnetwork(stacked_convs, conv_cfg, norm_cfg)
        self.cls_loss_fn = build_loss(cls_loss)
        self.cls_scores = cls_scores
        self.reg_loss_fn = build_loss(reg_loss)
        self.reg_weights = reg_weights
        self.use_cls_scores = cls_loss['type'] == 'QualityFocalLoss'
        self.use_iou = (self.use_cls_scores and self.cls_scores
                        == 'iou') or self.reg_weights == 'iou'
        if train_cfg:
            self.assigner = build_assigner(assigner)
            self.iou_calculator = build_iou_calculator(
                dict(type='RBboxOverlaps2D'))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def device(self) -> torch.device:
        """Returns the device on which the head currently is."""
        return self.strides.device

    def build_classification_subnetwork(self, stacked_convs: int,
                                        conv_cfg: dict,
                                        norm_cfg: dict) -> None:
        """Builds the classification subnetwork of the FCOSR head.

        Args:
            stacked_convs (int): Number of stacked convolutions.
            conv_cfg (dict, optional): Config dict for convolution layer.
            norm_cfg (dict, optional): Config dict for normalization layer.

        Returns:
            None.
        """
        # accumulator for classification convolutions
        cls_convs = []
        for i in range(stacked_convs):
            cls_convs.append(
                ConvModule(
                    in_channels=(self.feat_channels
                                 if i else self.in_channels),
                    out_channels=self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        # stacked classification convolutions
        self.cls_convs = nn.Sequential(*cls_convs)
        # classification subnetwork output
        self.cls_logits_conv = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.num_classes,
            kernel_size=3,
            padding=1)

    def build_regression_subnetwork(self, stacked_convs: int, conv_cfg: dict,
                                    norm_cfg: dict) -> None:
        """Builds the regression subnetwork of the FCOSR head.

        Args:
            stacked_convs (int): Number of stacked convolutions.
            conv_cfg (dict, optional): Config dict for convolution layer.
            norm_cfg (dict, optional): Config dict for normalization layer.

        Returns:
            None.
        """
        # accumulator for regression convolutions
        reg_convs = []
        for i in range(stacked_convs):
            reg_convs.append(
                ConvModule(
                    in_channels=(self.feat_channels
                                 if i else self.in_channels),
                    out_channels=self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        # stacked regression convolutions
        self.reg_convs = nn.Sequential(*reg_convs)
        # regression subnetwork outputs
        self.reg_xy_conv = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=2,
            kernel_size=3,
            padding=1)
        self.reg_wh_conv = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=2,
            kernel_size=3,
            padding=1)
        self.reg_theta_conv = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=1,
            kernel_size=3,
            padding=1)
        # level wise scale layers for the regression subnetwork
        self.reg_scales = nn.ModuleList([Scale() for _ in self.strides])

    def forward_single_lvl(self, feature_map: Tensor, stride: Tensor,
                           scale: Scale) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass on a single feature level.

        Args:
            feature_map (Tensor): the level feature map of shape
                (batch_size, in_channels, fm_height, fm_width)
            stride (Tensor): the level's stride
            scale (Scale): the level's Scale layer

        Returns:
            cls_logits (Tensor): classification logits for this level
                of shape (batch_size, num_classes, fm_height, fm_width)
            reg_bboxes (Tensor): bounding box predictions for this level
                of shape (batch_size, 5, fm_height, fm_width)
        """
        # dissociate the classification and regression features
        cls_features = feature_map
        reg_features = feature_map
        # apply the stacked convolutions
        cls_features = self.cls_convs(cls_features)
        reg_features = self.reg_convs(reg_features)
        # compute classification logits
        cls_logits = self.cls_logits_conv(cls_features)
        # compute regression logits
        reg_xy = stride * scale(self.reg_xy_conv(reg_features))
        reg_wh = stride * (1.0 + F.elu(scale(self.reg_wh_conv(reg_features))))
        reg_theta = torch.fmod(self.reg_theta_conv(reg_features), torch.pi / 2)
        return cls_logits, torch.cat((reg_xy, reg_wh, reg_theta), dim=1)

    def forward_multi_lvl(
            self, feature_maps: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass on a all feature levels.

        Args:
            feature_maps (tuple[Tensor, ...]): the feature maps for all levels

        Returns:
            cls_logits (Tensor): concatenated classification logits for all
                levels of shape (batch_size, nb_preds, num_classes)
            reg_bboxes (Tensor): concatenated bounding box predictions for all
                lvls of shape (batch_size, nb_preds, 5)
        """
        list_cls_logits, list_rbox_preds = multi_apply(
            self.forward_single_lvl,
            feature_maps,
            self.strides,
            self.reg_scales,
        )
        all_cls_logits = torch.cat(
            [cls_logits.flatten(2) for cls_logits in list_cls_logits], dim=-1)
        all_rbox_preds = torch.cat(
            [rbox_preds.flatten(2) for rbox_preds in list_rbox_preds], dim=-1)
        return all_cls_logits.transpose(1, 2), all_rbox_preds.transpose(1, 2)

    forward = forward_multi_lvl

    def compute_auxiliaries(
        self,
        fm_shapes: List[Tuple[int, int]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes the array of initial positions, strides and regress ranges
        for all feature levels.

        Args:
            feature_maps_wh (list[tuple[int, int]]): the feature maps height
                and width for all levels

        Returns:
            positions (Tensor): the tensor containing the initial positions
                on the input images. Shape: (sum_feature_maps_area, 2)
            strides (Tensor) : the tensor containing the strides of the feature
                map for each position. Shape: (sum_feature_maps_area, )
            regression_ranges (Tensor): the tensor containing the regression
                ranges of the feature map for each position. Shape:
                (sum_feature_maps_area, 2)
        """
        positions_acc, stride_acc, regress_ranges_acc = [], [], []
        for (width,
             height), stride, regress_ranges in zip(fm_shapes, self.strides,
                                                    self.regress_ranges):
            offset, step = torch.floor(stride / 2), stride
            positions = torch.stack(torch.meshgrid(
                torch.arange(offset, offset + width *
                             step, step),  # type: ignore
                torch.arange(offset, offset + height * \
                             step, step),  # type: ignore
                indexing='ij'))
            positions = positions.flatten(1).transpose(0, 1)  # P, 2
            positions_acc.append(positions)
            stride_acc.append(stride.repeat(positions.size(0)))
            regress_ranges_acc.append(
                torch.tensor(regress_ranges).repeat((positions.size(0), 1)))
        return (torch.vstack(positions_acc).to(self.device),
                torch.hstack(stride_acc).to(self.device),
                torch.vstack(regress_ranges_acc).to(self.device))

    def run_assignment(
        self,
        gt_boxes: Tensor,
        points: Tensor,
        strides: Tensor,
        regranges: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # run the assigner
        outputs = self.assigner.assign(gt_boxes, points, strides,
                                       regranges).info
        # retrieve gt_inds
        gt_inds: Tensor = outputs['gt_inds']  # P
        # get scores
        if 'scores' in outputs:
            ngd_scores: Tensor = outputs['scores']  # P
        elif self.use_cls_scores:
            warnings.warn(
                'Assigner provided no cls scores, fixing scores to 1.0')
            ngd_scores = torch.ones(
                gt_boxes.size(0), device=points.device)  # P
        else:
            ngd_scores = torch.zeros(0, device=points.device)  # 0
        # compute assigned_mask
        assigned_mask = gt_inds != 0  # P
        return gt_inds, ngd_scores, assigned_mask

    def compute_target_single_image(
            self,
            gt_boxes: Tensor,  # N, 5
            gt_labels: Tensor,  # N
            pred_boxes: Tensor,  # P, 5
            points: Tensor,  # P, 2
            strides: Tensor,  # P
            regranges: Tensor,  # P, 2
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Computes the FCOSR targets for one image in the batch.

        Args:
            gt_rbboxes (Tensor): the ground truth rbboxes in the
                (cx, cy, w, h, angle) format. Shape: (nb_gts, 5)
            gt_labels (Tensor): the ground truth labels with
                shape (nb_gts, 5)
            pred_boxes (Tensor): the predicted rbboxes in the same
                format as gt_rbboxes. Shape: (nb_preds, 5)
            points (Tensor): the positions of the predictions in
                the (x, y) format with shape (nb_preds, 2)
            strides (Tensor): the strides of the feature maps from
                which each prediction came from. Shape: (nb_preds,)
            regranges (Tensor): the regress ranges of the feature
                maps from which each prediction came from. Shape:
                (nb_preds, 2)

        Returns:
            cls_targets (Tensor): of shape (nb_preds, ) the classification
                targets with BG label = num_classes
            cls_scores (Tensor): the classification scores if they are
                needed with shape (nb_preds,)
            reg_targets (Tensor): of shape (nb_dets, 5) containing the
                regression targets in the (cx, cy, w, h, theta) format
            reg_weights (Tensor): of shape (nb_dets,) containing the
                weights of the regression loss
        """
        gt_inds, ngd_scores, assigned = self.run_assignment(
            gt_boxes, points, strides, regranges)
        if assigned.sum() == 0:
            return (
                torch.full_like(gt_inds, self.num_classes),
                torch.zeros_like(strides, dtype=torch.float32),
                torch.zeros((0, 5), device=strides.device),
                torch.zeros((0, ), device=strides.device),
            )
        cls_targets = F.pad(
            gt_labels, (1, 0),
            value=self.num_classes).index_select(0, gt_inds)
        reg_targets = F.pad(gt_boxes,
                            (0, 0, 1, 0)).index_select(0, gt_inds[assigned])
        reg_targets[:, :2] -= points[assigned]
        if self.use_iou:
            iou = self.iou_calculator(
                bboxes1=reg_targets,
                bboxes2=pred_boxes[assigned],
                is_aligned=True)
        # compute the classification scores if need be
        if self.use_cls_scores:
            cls_scores = torch.zeros_like(ngd_scores)
            if self.cls_scores == 'gauss':
                cls_scores[assigned] = ngd_scores[assigned]
            elif self.cls_scores == 'iou':
                cls_scores[assigned] = iou
            elif self.cls_scores == 'none':
                cls_scores[assigned] = 1.0
            else:
                raise ValueError(f'Unknown cls_scores: {self.cls_scores}')
        else:
            cls_scores = torch.zeros((0), device=points.device)
        # compute the regression weights
        reg_weights = torch.zeros_like(reg_targets[:, 0])
        if self.reg_weights == 'iou':
            reg_weights = iou
        elif self.reg_weights['type'] == 'mean':
            reg_weights[:] = 1.0
        elif self.reg_weights['type'] in ('gauss', 'centerness'):
            reg_weights = ngd_scores[assigned]
        else:
            raise ValueError(f'Unknown reg_weights: {self.reg_weights}')
        return cls_targets, cls_scores, reg_targets, reg_weights

    def compute_targets(
        self,
        gt_boxes: List[Tensor],
        gt_labels: List[Tensor],
        pred_boxes: Tensor,
        fm_shapes: List[Tuple[int, int]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        points, strides, regranges = self.compute_auxiliaries(fm_shapes)
        cls_targets, cls_scores, reg_targets, reg_weights = multi_apply(
            self.compute_target_single_image, gt_boxes, gt_labels, pred_boxes,
            repeat(points), repeat(strides), repeat(regranges))
        return (torch.cat(cls_targets), torch.cat(cls_scores),
                torch.cat(reg_targets), torch.cat(reg_weights))

    def forward_train(self, feature_maps: Tuple[Tensor,
                                                ...], gt_boxes: List[Tensor],
                      gt_labels: List[Tensor]) -> Dict[str, Tensor]:
        # head forward pass
        cls_logits, pred_boxes = self.forward(feature_maps)
        # compute targets for all images
        cls_targets, cls_scores, reg_targets, reg_wghts = self.compute_targets(
            gt_boxes, gt_labels, pred_boxes,
            [(fm.size(2), fm.size(3)) for fm in feature_maps])
        # format predictions
        cls_logits = torch.cat([x for x in cls_logits])  # B, P, C -> BP, C
        pred_boxes = torch.cat([x for x in pred_boxes])  # B, P, 5 -> BP, 5
        # filter regression predictions on successful assignment
        pred_boxes = pred_boxes[cls_targets != self.num_classes]
        # compute losses
        losses = self.loss(cls_logits, cls_targets, cls_scores, pred_boxes,
                           reg_targets, reg_wghts)
        return losses

    def loss(self, cls_logits: Tensor, cls_targets: Tensor, cls_scores: Tensor,
             rbox_preds: Tensor, reg_targets: Tensor,
             reg_weights: Tensor) -> Dict[str, Tensor]:
        # compute classification loss
        avg_factor = reg_weights.sum() if (
            reg_targets.size(0) != 0) else cls_targets.size(0)
        if self.use_cls_scores:
            cls_loss = self.cls_loss_fn(
                cls_logits, (cls_targets, cls_scores), avg_factor=avg_factor)
        else:
            cls_loss = self.cls_loss_fn(
                cls_logits, cls_targets, avg_factor=avg_factor)
        loss = {'cls_loss': cls_loss}
        # compute regression loss
        if reg_targets.size(0) != 0:
            reg_loss = self.reg_loss_fn(
                rbox_preds, reg_targets, reg_weights, avg_factor=avg_factor)
        else:
            reg_loss = torch.tensor([0.0])
        loss['reg_loss'] = reg_loss
        # TODO : add iou_mean
        return loss

    def get_bboxes_single(
            self,
            cls_logits: Tensor,  # P, C
            rbox_preds: Tensor,  # P, 5
            positions: Tensor,  # P, 2
            strides: Tensor,  # P
            img_meta: dict,
            cfg: Config,
            rescale: bool = False) -> Tuple[Tensor, Tensor]:
        """Gather the predicted bounding boxes for one image
        by following these steps:
        1. discard boxes that are too small (default: min_size = 0.0)
        2. discard boxes that have less than 2 points outside the
            original image
        3. apply top-k filtering on boxes level-wise (default: k = 0)
        4. rescale boxes if rescale flag is set to True
        5. apply either class-wise and maybe followed by an extra
            class-agnostic NMS (default: simple class-wise NMS)

        Args:
            cls_logits (Tensor): tensor of shape (nb_positions, nb_classes)
                where each column corresponds to a position and each line
                to a class
            rbox_preds (Tensor): tensor of shape (nb_positions, 5) where
                each column is a predicted bounding box at the given position.
            positions (Tensor): tensor of shape (nb_positions, 2) where
                each column corresponds to a position (x, y)
            strides (Tensor): tensor of shape (nb_position) where each element
                corresponds to a position and is the stride of the feature map
                from which that position came from
            img_meta (dict): dict containing at least 'scale_factor' and
                'img_shape'
            cfg (dict): the config for this phase (training or test)
            rescale (bool): whether or not to rescale the detections with the
                images' scale factors

        Returns
            dets (Tensor): of shape (nb_dets, 6) where each detections is
                in the (cx, cy, w, h, theta, score) format.
            labels (Tensor): of shape (nb_dets, ) containing the detections'
                labels
        """
        # turn logits to scores and add an extra column for background class
        cls_scores = torch.hstack((cls_logits.sigmoid(),
                                   torch.zeros((cls_logits.size(0), 1),
                                               device=cls_logits.device)))
        # filter on size
        min_bbox_size = cfg.get('min_bbox_size', 0)
        if min_bbox_size > 0:
            big_enough = (rbox_preds[:, 2] >= min_bbox_size) & \
                         (rbox_preds[:, 3] >= min_bbox_size)
            if not big_enough.all():
                cls_scores, rbox_preds = cls_scores[big_enough], rbox_preds[
                    big_enough]
                positions, strides = positions[big_enough], strides[big_enough]
        # filter having two points inside the original image
        # TODO : add two point poly cut
        rbox_preds[:, :2] += positions
        corners: Tensor = obb2poly(rbox_preds)
        img_width, img_height = int(img_meta['img_shape'][0]), int(
            img_meta['img_shape'][1])
        points_inside = (
            (corners[:, 0::2] < img_width) & (corners[:, 0::2] >= 0)) & \
            ((corners[:, 1::2] < img_height) & (corners[:, 1::2] >= 0))
        two_points_in = points_inside.sum(dim=1) > 1
        if not two_points_in.all():
            cls_scores, rbox_preds = cls_scores[two_points_in], rbox_preds[
                two_points_in]
            positions, strides = positions[two_points_in], strides[
                two_points_in]
        # level-wise top-k filtering
        nms_pre = cfg.get('nms_pre', 0)
        if nms_pre > 0:
            kept = torch.zeros_like(positions, dtype=torch.bool)
            for lvl_stride in self.strides:
                lvl_mask = lvl_stride == strides
                if lvl_mask.sum() > nms_pre:
                    lvl_max_scores = cls_scores[lvl_mask].max(dim=1)[0]
                    lvl_topk_inds = lvl_max_scores.topk(nms_pre)[1]
                    kept[lvl_mask[lvl_topk_inds]] = 1
                else:
                    kept[lvl_mask] = 1
            cls_scores, rbox_preds = cls_scores[kept], rbox_preds[kept]
            positions, strides = positions[kept], strides[kept]
        # eventual rescaling
        if rescale and (scale_factor := img_meta['scale_factor']) != 1:
            rbox_preds[:, :4] /= scale_factor
        # apply NMS
        dets, labels, _ = multiclass_nms_rotated_extra(
            multi_bboxes=rbox_preds,
            multi_scores=cls_scores,
            score_thr=cfg.get('score_thr', 0.05),
            nms=cfg.get('nms', 0.1),
            extra_nms=cfg.get('extra_nms'),
            max_num=cfg.get('max_per_img', 2000))
        # format return
        return dets, labels

    @force_fp32(apply_to=('cls_logits', 'rbboxes_pred'))
    def get_bboxes(self,
                   cls_logits: Tensor,
                   rbboxes_pred: Tensor,
                   positions: Tensor,
                   strides: Tensor,
                   img_metas: List[dict],
                   cfg: dict,
                   rescale: bool = None) -> List[Tuple[Tensor, Tensor]]:
        """Gather the predicted bounding boxes for all images. If the batch is
        a test rotation (one image being rotated multiple times) then collate
        the rbboxes and performs NMS.

        Args:
            cls_logits (Tensor): containing the logits of class probailities
                with shape (btch_size, nb_predictions, nb_classes).
            rbboxes_pred (Tensor): of shape (batch_size, nb_predictions, 5)
                containing the predicted rbboxes in the (cx, cy, w, h, theta)
            positions (Tensor): the tensor containing the initial positions
                on the input images. Shape: (sum_feature_maps_area, 2)
            strides (Tensor) : the tensor containing the strides of the feature
                map for each position. Shape: (sum_feature_maps_area, )
            regression_ranges (Tensor): the tensor containing the regression
                ranges of the feature map for each position. Shape:
                (sum_feature_maps_area, 2)
            img_meta (list[dict]): dict containing at least 'scale_factor' and
                'img_shape'
            cfg (dict): the config for this phase (training or test)
            rescale (bool): whether or not to rescale the detections with the
                images' scale factors

        Returns
            dets (Tensor): of shape (nb_dets, 6) where each detections is
                in the (cx, cy, w, h, theta, score) format.
            labels (Tensor): of shape (nb_dets, ) containing the detections'
                labels
        """
        # get the detections from each image
        dets_list, labels_list = multi_apply(self.get_bboxes_single,
                                             cls_logits, rbboxes_pred,
                                             repeat(positions),
                                             repeat(strides), img_metas,
                                             repeat(cfg), repeat(rescale))
        # check if the batch is a rotation test
        rotations = cfg.get('test_rotations', [])
        # if it's not (rotations is an empty list) return the list of dets
        if not rotations:
            return [x for x in zip(dets_list, labels_list)]  # type: ignore
        # if it is rotate the preds & apply same NMS as in get_rboxes_single
        all_rbox_preds: List[Tensor] = []
        all_cls_scores: List[Tensor] = []
        img_width, img_height = img_metas[0]['img_shape']
        for dets, labels, rotation in zip(dets_list, labels_list, rotations):
            # rotate the predictions and accumulate them
            all_rbox_preds.append(
                discrete_rot90_rbboxes(rotation, dets[:, :-1], img_width,
                                       img_height))
            # compute the cls_scores and accumulate them
            # (not as in the original FCOSR)
            all_cls_scores.append(
                dets[:, -1] * F.one_hot(labels.long(), self.num_classes + 1))
        # apply the NMS
        dets, labels, _ = multiclass_nms_rotated_extra(
            multi_bboxes=torch.cat(all_rbox_preds, dim=0),
            multi_scores=torch.cat(all_cls_scores, dim=0),
            score_thr=cfg.get('score_thr', 0.05),
            nms=cfg.get('nms', 0.1),
            extra_nms=cfg.get('extra_nms'),
            max_num=cfg.get('max_per_img', 2000))
        return [(dets, labels)]
