from typing import Optional, Union, Sequence, List
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmpretrain.evaluation.metrics.multi_label import AveragePrecision, _average_precision
from mmpretrain.structures.utils import format_label
from mmpretrain.evaluation.metrics.single_label import to_tensor

@METRICS.register_module()
class CChessPrecisionWithLayout(BaseMetric):
    default_prefix: Optional[str] = 'cchess-layout'

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

        ap = self.calculate(pred, target)

        result_metrics = dict()
        # A0 - J8
        for i in range(10):
            for j in range(9):
                result_metrics[f'AP_{chr(65 + i)}{j}'] = ap[i * 9 + j]

        return result_metrics

    @staticmethod
    def calculate(pred: Union[torch.Tensor, np.ndarray],
                  target: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:


        pred = to_tensor(pred)
        target = to_tensor(target)


        assert pred.ndim == 3 and target.ndim == 3, \
            'Both `pred` and `target` should have shape `(N, 90, 16)`.'
        
        # check shape
        assert pred.shape[1] == 90 and pred.shape[2] == 16, \
            '`pred` should have shape `(N, 90, 16)`.'
        assert target.shape[1] == 90 and target.shape[2] == 16, \
            '`target` should have shape `(N, 90, 16)`.'
        

        # (N, 90, 16)
        ap = pred.new_zeros(90)
        for k in range(90):
            ap[k] = _average_precision(pred[:, k, :], target[:, k, :])

        return ap * 100.0
