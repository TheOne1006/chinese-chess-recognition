from typing import Optional, Union, Sequence, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmpretrain.evaluation.metrics.multi_label import AveragePrecision, _average_precision
from mmpretrain.structures.utils import format_label
from mmpretrain.evaluation.metrics.single_label import to_tensor


dict_cate_names = {
    'point': '.',
    'other': 'x',
    'red_king': 'K',
    'red_advisor': 'A',
    'red_bishop': 'B',
    'red_knight': 'N',
    'red_rook': 'R',
    'red_cannon': 'C',
    'red_pawn': 'P',
    'black_king': 'k',
    'black_advisor': 'a',
    'black_bishop': 'b',
    'black_knight': 'n',
    'black_rook': 'r',
    'black_cannon': 'c',
    'black_pawn': 'p',
}

_cate_names = list(dict_cate_names.values())

_cate_keys = list(dict_cate_names.keys())


@METRICS.register_module()
class CChessPrecisionWith16Class(BaseMetric):
    default_prefix: Optional[str] = 'cchess-16-class'

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
                # sparse_onehot = F.one_hot(label, num_classes)
                result['gt_score'] = F.one_hot(label, num_classes)

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

        # concat
        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        # default
        allow_shape = (90, 16)
        cate_keys = _cate_keys
        cate_names = _cate_names

        if self.filter_gt_labels is not None:
            target = target[:, :, self.filter_gt_labels]
            pred = pred[:, :, self.filter_gt_labels]
            allow_shape = (90, len(self.filter_gt_labels))
            cate_keys = [cate_keys[i] for i in self.filter_gt_labels]
            cate_names = [cate_names[i] for i in self.filter_gt_labels]

        ap = self.calculate(pred, target, allow_shape, cate_names)

        result_metrics = dict()

        for k, v in zip(cate_keys, ap):
            result_metrics[k] = v

        return result_metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        allow_shape: Tuple[int, int] = (90, 16),
        cate_names: List[str] = _cate_names
    ) -> torch.Tensor:


        pred = to_tensor(pred)
        target = to_tensor(target)


        assert pred.ndim == 3 and target.ndim == 3, \
            'Both `pred` and `target` should have shape `(N, 90, 16)`.'
        

        # check shape
        assert pred.shape[1:] == allow_shape, \
            f'`pred` should have shape `(N, {allow_shape[0]}, {allow_shape[1]})`.'
        assert target.shape[1:] == allow_shape, \
            f'`target` should have shape `(N, {allow_shape[0]}, {allow_shape[1]})`.'
        

        # (N, 90, 16)
        ap = pred.new_zeros(len(cate_names))
        for k in range(len(cate_names)):
            ap[k] = _average_precision(pred[:, :, k], target[:, :, k])

        return ap * 100.0
