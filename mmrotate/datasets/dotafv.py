# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Tuple

import torch
from mmcv.ops import box_iou_rotated
from torch import Tensor

from mmrotate.core import eval_rbbox_map
from .builder import ROTATED_DATASETS
from .dota import DOTADataset


def format_dets(det_results) -> List[Tensor]:
    acc = []
    for image_dets in det_results:
        image_acc = []
        for label, bboxes in enumerate(image_dets):
            image_acc.append(
                torch.hstack((
                    torch.tensor(bboxes[:, :-1]),
                    torch.full((bboxes.shape[0], 1), label),
                )))
        acc.append(torch.vstack(image_acc).float())
    return acc


def format_gts(annotations) -> List[Tensor]:
    acc = []
    for image_gts in annotations:
        gts = torch.hstack((
            torch.tensor(image_gts['bboxes']),
            torch.tensor(image_gts['labels']).reshape((-1, 1)),
        ))
        acc.append(gts.float())
    return acc


def eval_matches_single(dets, gts, iou_thr) -> Tuple[int, ...]:
    true_matches, false_matches, false_dets = 0, 0, 0
    missed_gts = set(range(len(gts)))
    ious: Tensor = box_iou_rotated(dets[:, :-1], gts[:, :-1])
    for i_det, i_gt in enumerate(ious.argmax(1).tolist()):
        if ious[i_det, i_gt] >= iou_thr:
            if i_gt in missed_gts:
                missed_gts.remove(i_gt)
            if dets[i_det, -1] == gts[i_gt, -1]:
                true_matches += 1
            else:
                false_matches += 1
        else:
            false_dets += 1
    return true_matches, false_matches, false_dets, len(missed_gts)


def eval_matches(det_results, annotations, iou_thr=0.5) -> List[float]:
    accumulator = torch.zeros(
        (4, ))  # true_matches, false_matches, false_dets, missed_gts
    for dets, gts in zip(format_dets(det_results), format_gts(annotations)):
        accumulator += torch.tensor(eval_matches_single(dets, gts, iou_thr))
    return (accumulator / accumulator.sum()).tolist()


def print_matches(matches: List[float]) -> None:
    print('+--------------+---------------+------------+------------+')
    print('| true_matches | false_matches | false_dets | missed_gts |')
    print('+--------------+---------------+------------+------------+')
    print(f"|{' '*8}{matches[0]:.3f} |{' '*9}{matches[1]:.3f}",
          f"|{' '*6}{matches[2]:.3f} |{' '*6}{matches[3]:.3f} |")
    print('+--------------+---------------+------------+------------+')


def eval_iou_single(dets, gts, iou_thr) -> Tuple[float, float, int]:
    ca_iou, tm_iou = 0., 0.
    ious: Tensor = box_iou_rotated(dets[:, :-1], gts[:, :-1])
    for i_det, i_gt in enumerate(ious.argmax(1).tolist()):
        if ious[i_det, i_gt] >= iou_thr:
            ca_iou += ious[i_det, i_gt].item()
            if dets[i_det, -1] == gts[i_gt, -1]:
                tm_iou += ious[i_det, i_gt].item()
    return ca_iou, tm_iou, dets.size(0)


def eval_ious(det_results, annotations, iou_thr=0.5) -> Tuple[float, float]:
    accumulator = torch.zeros((3, ))  # class_agnostic iou, true_match iou
    for dets, gts in zip(format_dets(det_results), format_gts(annotations)):
        accumulator += torch.tensor(eval_iou_single(dets, gts, iou_thr))
    ca_iou, tm_iou, dets = accumulator.tolist()
    return ca_iou / dets, tm_iou / dets


def print_ious(ious: Tuple[float, float]) -> None:
    print('+--------------------+----------------+')
    print('| class_agnostic_IoU | true_match_IoU |')
    print('+--------------------+----------------+')
    print(f"|{' '*14}{ious[0]:.3f} |{' '*10}{ious[1]:.3f} |")
    print('+--------------------+----------------+')


@ROTATED_DATASETS.register_module()
class DOTAFullValDataset(DOTADataset):

    def evaluate(self,
                 results,
                 logger=None,
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        # mAP
        assert isinstance(iou_thr, float)
        mean_ap, _ = eval_rbbox_map(
            results,
            annotations,
            scale_ranges=scale_ranges,
            iou_thr=iou_thr,
            dataset=self.CLASSES,
            logger=logger,
            nproc=nproc)
        eval_results['mAP'] = mean_ap
        # matches
        matches = eval_matches(results, annotations, iou_thr)
        eval_results['true_matches'] = matches[0]
        eval_results['false_matches'] = matches[1]
        eval_results['false_dets'] = matches[2]
        eval_results['missed_dets'] = matches[3]
        print()
        print_matches(matches)
        # iou
        ious = eval_ious(results, annotations, iou_thr)
        eval_results['class_agnostic_IoU'] = ious[0]
        eval_results['true_match_IoU'] = ious[1]
        print()
        print_ious(ious)
        return eval_results
