
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
# from mmengine.evaluator import BaseMetric
from mmpretrain.evaluation.metrics.multi_label import MultiLabelMetric
# from mmengine.logging import MMLogger
from mmpretrain.structures.utils import format_label

from mmpretrain.registry import METRICS
from mmpretrain.structures import label_to_onehot
from mmpretrain.evaluation.metrics.single_label import to_tensor
# from mmpretrain.evaluation.metrics.multi_label import _average_precision

from mmpretrain.evaluation.metrics.single_label import _precision_recall_f1_support

def _average_precision(pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    r"""Calculate the average precision for a single class.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        pred (torch.Tensor): The model prediction with shape
            ``(N, 90, num_classes)``.
        target (torch.Tensor): The target of predictions with shape
            ``(N, 90, num_classes)``.

    Returns:
        torch.Tensor: average precision result.
    """
    # 如果 shape 为 (N, 90, num_classes)，则将其调整为 (N*90, num_classes)
    if pred.shape[1] == 90 and target.shape[1] == 90:
        pred = pred.view(-1, pred.shape[-1])  # 将形状调整为 (N*90, num_classes)
        target = target.view(-1, target.shape[-1])  # 将形状调整为 (N*90, num_classes)

    assert pred.shape == target.shape, \
        f"The size of pred ({pred.shape}) doesn't match "\
        f'the target ({target.shape}).'

    # a small value for division by zero errors
    eps = torch.finfo(torch.float32).eps

    # get rid of -1 target such as difficult sample
    # that is not wanted in evaluation results.
    valid_index = target > -1
    pred = pred[valid_index]
    target = target[valid_index]

    # sort examples
    sorted_pred_inds = torch.argsort(pred, dim=0, descending=True)
    sorted_target = target[sorted_pred_inds]

    # get indexes when gt_true is positive
    pos_inds = sorted_target == 1

    # Calculate cumulative tp case numbers
    tps = torch.cumsum(pos_inds, 0)
    total_pos = tps[-1].item()  # the last of tensor may change later

    # Calculate cumulative tp&fp(pred_poss) case numbers
    pred_pos_nums = torch.arange(1, len(sorted_target) + 1).to(pred.device)
    pred_pos_nums[pred_pos_nums < eps] = eps

    tps[torch.logical_not(pos_inds)] = 0
    precision = tps / pred_pos_nums.float()
    ap = torch.sum(precision, 0) / max(total_pos, eps)
    return ap


@METRICS.register_module()
class CChessMultiLabelMetric(MultiLabelMetric):

    def __init__(self,
                 filter_gt_labels: List[int] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.filter_gt_labels = filter_gt_labels

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()

            result['pred_score'] = data_sample['pred_score'].clone()
            num_classes = result['pred_score'].size()[-1]

            if 'gt_score' in data_sample:
                result['gt_score'] = data_sample['gt_score'].clone()
            else:
                label = format_label(data_sample['gt_label'])
                sparse_onehot = F.one_hot(label, num_classes)
                result['gt_score'] = sparse_onehot

            # Save the result to `self.results`.
            self.results.append(result)


    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        metrics = {}

        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        if self.filter_gt_labels is not None:
            target = target[:, :, self.filter_gt_labels]
            pred = pred[:, :, self.filter_gt_labels]

        metric_res = self.calculate(
            pred,
            target,
            pred_indices=False,
            target_indices=False,
            average=self.average,
            thr=self.thr,
            topk=self.topk)

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics

        if self.thr:
            suffix = '' if self.thr == 0.5 else f'_thr-{self.thr:.2f}'
            for k, v in pack_results(*metric_res).items():
                metrics[k + suffix] = v
        else:
            for k, v in pack_results(*metric_res).items():
                metrics[k + f'_top{self.topk}'] = v

        result_metrics = dict()
        for k, v in metrics.items():
            if self.average is None:
                result_metrics[k + '_classwise'] = v.detach().cpu().tolist()
            elif self.average == 'macro':
                result_metrics[k] = v.item()
            else:
                result_metrics[k + f'_{self.average}'] = v.item()
        return result_metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        pred_indices: bool = False,
        target_indices: bool = False,
        average: Optional[str] = 'macro',
        thr: Optional[float] = None,
        topk: Optional[int] = None,
        num_classes: Optional[int] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the precision, recall, f1-score.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, num_classes)`` or a sequence of index/onehot
                format labels.
            target (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, num_classes)`` or a sequence of index/onehot
                format labels.
            pred_indices (bool): Whether the ``pred`` is a sequence of
                category index labels. If True, ``num_classes`` must be set.
                Defaults to False.
            target_indices (bool): Whether the ``target`` is a sequence of
                category index labels. If True, ``num_classes`` must be set.
                Defaults to False.
            average (str | None): How to calculate the final metrics from
                the confusion matrix of every category. It supports three
                modes:

                - `"macro"`: Calculate metrics for each category, and calculate
                  the mean value over all categories.
                - `"micro"`: Average the confusion matrix over all categories
                  and calculate metrics on the mean confusion matrix.
                - `None`: Calculate metrics of every category and output
                  directly.

                Defaults to "macro".
            thr (float, optional): Predictions with scores under the thresholds
                are considered as negative. Defaults to None.
            topk (int, optional): Predictions with the k-th highest scores are
                considered as positive. Defaults to None.
            num_classes (Optional, int): The number of classes. If the ``pred``
                is indices instead of onehot, this argument is required.
                Defaults to None.

        Returns:
            Tuple: The tuple contains precision, recall and f1-score.
            And the type of each item is:

            - torch.Tensor: A tensor for each metric. The shape is (1, ) if
              ``average`` is not None, and (C, ) if ``average`` is None.

        Notes:
            If both ``thr`` and ``topk`` are set, use ``thr` to determine
            positive predictions. If neither is set, use ``thr=0.5`` as
            default.
        """
        average_options = ['micro', 'macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specicy from {average_options}.'

        def _format_label(label, is_indices):
            """format various label to torch.Tensor."""
            if isinstance(label, np.ndarray):
                assert label.ndim == 2, 'The shape `pred` and `target` ' \
                    'array must be (N, num_classes).'
                label = torch.from_numpy(label)
            elif isinstance(label, torch.Tensor):
                assert label.ndim == 2, 'The shape `pred` and `target` ' \
                    'tensor must be (N, num_classes).'
            elif isinstance(label, Sequence):
                if is_indices:
                    assert num_classes is not None, 'For index-type labels, ' \
                        'please specify `num_classes`.'
                    # TODO: support 90-dim label
                    # label = torch.stack([
                    #     label_to_onehot(indices, num_classes)
                    #     for indices in label
                    # ])
                    raise NotImplementedError('90-dim label is not supported')
                else:
                    label = torch.stack(
                        [to_tensor(onehot) for onehot in label])
            else:
                raise TypeError(
                    'The `pred` and `target` must be type of torch.tensor or '
                    f'np.ndarray or sequence but get {type(label)}.')
            return label

        """
        reshape pred and target to (N*90, num_classes)
        """
        pred = pred.view(-1, pred.shape[-1])
        target = target.view(-1, target.shape[-1])

        pred = _format_label(pred, pred_indices)
        target = _format_label(target, target_indices).long()
       
        assert pred.shape == target.shape, \
            f"The size of pred ({pred.shape}) doesn't match "\
            f'the target ({target.shape}).'

        if num_classes is not None:
            assert pred.size(1) == num_classes, \
                f'The shape of `pred` ({pred.shape}) '\
                f"doesn't match the num_classes ({num_classes})."
        num_classes = pred.size(1)

        thr = 0.5 if (thr is None and topk is None) else thr

        if thr is not None:
            # a label is predicted positive if larger than thr
            pos_inds = (pred >= thr).long()
        else:
            # top-k labels will be predicted positive for any example
            _, topk_indices = pred.topk(topk)
            pos_inds = torch.zeros_like(pred).scatter_(1, topk_indices, 1)
            pos_inds = pos_inds.long()

        return _precision_recall_f1_support(pos_inds, target, average)

