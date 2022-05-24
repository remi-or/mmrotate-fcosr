# Copyright (c) OpenMMLab. All rights reserved.
from itertools import count
from typing import Tuple

import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from torch import Tensor

from mmrotate.core.bbox.transforms import obb2hbb
from ..builder import ROTATED_BBOX_ASSIGNERS

PI = 3.1415926535898


def normalized_gaussian_distance_scores(
        gt_bboxes: Tensor, points: Tensor,
        gauss_factor: float) -> Tuple[Tensor, Tensor]:
    """Computes the normalized gaussian distance scores for each position and
    each bounding box. Also returns the determinant of each covariant matrix
    used (one per bbox).

    Args:
        gt_bboxes (Tensor): tensor of shape (nb_bboxes, 5) where
            each line is a bbox as (cx, cy, w, h, theta)
        points (Tensor): tensor of shape (nb_points, 2)
            containing all points on which bbox are regressed
        gauss_factor (float): the gauss factor used for computation

    Returns:
        scores (Tensor): the tensor of shape (nb_bboxes, nb_points)
            containing the normalized gaussian scores
        cm_dets (Tensor): the tensor containing the determinants of
            the cm (covariant matrices) for each gt_bbox
    """
    cm_dets = torch.zeros(gt_bboxes.size(0), device=points.device)
    scores = torch.zeros((gt_bboxes.size(0), points.size(0)),
                         device=points.device)
    for i, gt_bbox in enumerate(gt_bboxes):
        # unpack
        cx, cy, w, h, theta = gt_bbox
        cos, sin = torch.cos(theta), torch.sin(theta)
        # compute shrunk width (sw) and shrunk height (sh)
        shrink_factor = min(w, h) / gauss_factor
        sw, sh = w * shrink_factor, h * shrink_factor
        # compute covariant matrix (cm)
        cm_11 = sw * cos**2 + sh * sin**2
        cm_12_21 = (sw - sh) * sin * cos
        cm_22 = sw * sin**2 + sh * cos**2
        # compute the cm det
        cm_dets[i] = cm_11 * cm_22 - cm_12_21 * cm_12_21
        # compute covariant matrix inverse (cmi)
        inv_det = 1.0 / cm_dets[i]
        cmi_11 = cm_22 * inv_det
        cmi_12_21 = -cm_12_21 * inv_det
        cmi_22 = cm_11 * inv_det
        # compute offset between gt_bbox center and the points
        offset = points - torch.tensor(data=[[cx, cy]], device=points.device)
        # compute ngds and accumulate
        scores[i] = torch.exp((-0.5 * cmi_11) * offset[:, 0] * offset[:, 0] +
                              (-cmi_12_21) * offset[:, 0] * offset[:, 1] +
                              (-0.5 * cmi_22) * offset[:, 1] * offset[:, 1])
    return scores, cm_dets


def gaussian_distance_scores(gt_bboxes: Tensor, points: Tensor,
                             gauss_factor: float,
                             epsilon: float) -> Tuple[Tensor, Tensor]:
    """Computes the normalized and refined gaussian distance scores for each
    position and each bounding box. Normalized scores are used for
    classification weights and refined scores to determine which bbox to
    regress to.

    Args:
        gt_bboxes (Tensor): tensor of shape (nb_bboxes, 5) where
            each line is a bbox as (cx, cy, w, h, theta)
        points (Tensor): tensor of shape (nb_points, 2)
            containing all points on which bbox are regressed
        gauss_factor (float): the gauss factor used for computation
        epsilon (float): a small float used to avoid division by 0

    Returns:
        nscores (Tensor): tensor of shape (nb_bboxes, nb_points)
            containing the normalized gaussian scores
        rscores (Tensor): tensor of shape (nb_bboxes, nb_points)
            containing the refined gaussian scores
    """
    nscores, cm_dets = normalized_gaussian_distance_scores(
        gt_bboxes, points, gauss_factor)

    rscores = nscores.detach().clone()
    for i, gt_bbox in enumerate(gt_bboxes):
        w, h = gt_bbox[2:4]
        sqrt_wh = torch.sqrt(w * h)
        short_edge = torch.min(w, h)
        # n2r stands for normalized to refined
        n2r_factor = torch.sqrt(sqrt_wh * short_edge) / (
            2.0 * PI * torch.sqrt(cm_dets[i]) + epsilon)
        rscores[i, :] *= n2r_factor
    return nscores, rscores


def add_regress_range_to_mask(gt_bboxes: Tensor, points: Tensor,
                              inside_mask: Tensor, strides: Tensor,
                              regress_ranges: Tensor) -> Tensor:
    """Uses FCOSR assignment rules to modify inplace a mask representing which
    position is each bounding box.

    Args:
        gt_bboxes (Tensor): tensor of shape (nb_bboxes, 5) where
            each line is a bbox as (cx, cy, w, h, theta).
        points (Tensor): tensor of shape (nb_points, 2)
            containing all points on which bbox are regressed.
        inside_mask (Tensor): a mask of shape (nb_bboxes, nb_points)
            where each line corresponds to a bbox describing which
            position is inside or not.
        strides (Tensor) : tensor of shape (nb_points, ) containing
            the strides of the level for each position.
        regression_ranges (Tensor): tensor of shape (nb_points, 2)
            containing the regression rangesof the level for each position.
    """
    for i, encapsulating_hbb, gt_bbox in zip(count(), obb2hbb(gt_bboxes),
                                             gt_bboxes):
        # unpack the hbb
        center, wh = encapsulating_hbb[:2], encapsulating_hbb[2:4]
        top_left = points[inside_mask[i]] - (center - wh / 2)
        bottom_right = (center + wh / 2) - points[inside_mask[i]]
        max_size = torch.max(top_left, bottom_right).max(dim=1)[0]
        min_rr_mask = regress_ranges[inside_mask[i], 0] <= max_size
        max_rr_mask = max_size <= regress_ranges[inside_mask[i], 1]
        mlvl_mask = (gt_bbox[2:4].min() / 2.0) < strides[inside_mask[i]]
        assigned_mask = (min_rr_mask & max_rr_mask) | (~max_rr_mask
                                                       & mlvl_mask)
        inside_mask[i, inside_mask[i].nonzero()[~assigned_mask]] = False
    return inside_mask


@ROTATED_BBOX_ASSIGNERS.register_module()
class GaussianAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox. Each
    proposals will be assigned with 0 or a positive integer indicating the
    ground truth index. Does this using elliptical sampling described in the
    FCOSR paper: <https://arxiv.org/ftp/arxiv/papers/2111/2111.10780.pdf>.

    Args:
        gauss_factor (float, optional): Gauss factor for elliptical sampling.
        inside_ellipsis_thresh (float, optional): Threshold that controls
            how tight the elliptical sampling hugs the bounding box.
        epsilon (float, optional): Small float used to avoid division by 0.
    """

    def __init__(self,
                 gauss_factor: float = 12.0,
                 inside_ellipsis_thresh: float = 0.23,
                 epsilon: float = 1e-9) -> None:
        self.gauss_factor = gauss_factor
        self.inside_ellipsis_thresh = inside_ellipsis_thresh
        self.epsilon = epsilon

    def assign(
            self,
            gt_rboxes: Tensor,  # N, 5
            points: Tensor,  # P, 2
            strides: Tensor,  # P
            regress_ranges: Tensor,  # P, 2
    ) -> AssignResult:
        # compute gaussian distance scores (normalized and refined)
        nscores, rscores = gaussian_distance_scores(gt_rboxes, points,
                                                    self.gauss_factor,
                                                    self.epsilon)  # N, P
        # get assigned ground truth indices for each position
        inside_mask = (nscores >= self.inside_ellipsis_thresh)  # N, P
        inside_mask = add_regress_range_to_mask(gt_rboxes, points, inside_mask,
                                                strides, regress_ranges)
        rscores[~inside_mask] = -1.0
        max_rscores, gt_inds = rscores.max(dim=0)  # P
        assigned_mask = max_rscores > 0.0
        # get normalized gaussian distance scores (ngd_score)
        ngd_score = nscores[gt_inds, range(nscores.size(1))]  # P
        ngd_score[~assigned_mask] = 0.0
        # offset the gt_inds to 1-start
        gt_inds[assigned_mask] += 1
        gt_inds[~assigned_mask] = 0
        # build the result and add ngd scores (used for classification)
        result = AssignResult(
            num_gts=gt_rboxes.size(0),
            gt_inds=gt_inds,
            max_overlaps=None,
            labels=None,
        )
        result.set_extra_property('scores', ngd_score)
        return result
