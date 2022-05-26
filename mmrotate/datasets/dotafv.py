import os
from itertools import chain, repeat
import numpy as np

from mmcv.ops import box_iou_rotated
from mmrotate.core import eval_rbbox_map
from .builder import ROTATED_DATASETS
from .dota import DOTADataset


def format_dets(det_results):
    return [
        np.vstack([
            np.hstack((bboxes, np.full(bboxes.shape[:, 0].shape, label)))
            for label, bboxes in enumerate(image_dets)
        ])
        for image_dets in det_results
    ]

def format_gts(annotations):
    return [
        np.hstack((image_gts['bboxes'], image_gts['labels'].reshape(-1, 1)))
        for image_gts in annotations
    ]

def eval_matches_single(dets, gts, iou_thr):
    true_matches, false_matches, false_dets = 0, 0, 0
    missed_gts = set(range(len(gts)))
    ious = box_iou_rotated(dets[:, :-1], gts[:, :-1])
    for i_det, i_gt in enumerate(np.argmax(ious, 1)):
        if ious[i_det, i_gt] >= iou_thr:
            if i_gt in missed_gts:
                missed_gts.remove(i_gt)
            if dets[i_det, -1] == gts[i_gt, -1]:
                true_matches += 1
            else:
                false_matches += 1
        else:
            false_dets += 1
    return true_matches, false_matches, false_dets, missed_gts

def eval_matches(det_results, annotations, iou_thr=0.5):
    accumulator = np.zeros((4,)) # true_matches, false_matches, false_dets, missed_gts
    for dets, gts in zip(format_dets(det_results), format_gts(annotations)):
        accumulator += np.array(eval_matches_single(dets, gts, iou_thr))
    return accumulator / np.sum(accumulator)


@ROTATED_DATASETS.register_module()
class DOTAFullValDataset(DOTADataset):

    def evaluate(self,
                 results,
                 logger=None,
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
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
        eval_results['matches'] = eval_matches(results, 
                                               annotations, 
                                               iou_thr)
        return eval_results
