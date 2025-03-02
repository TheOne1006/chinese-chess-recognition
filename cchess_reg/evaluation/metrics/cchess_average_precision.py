from typing import Optional, Union, Sequence, List
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import METRICS
from mmpretrain.evaluation.metrics.multi_label import AveragePrecision, _average_precision
from mmpretrain.structures.utils import format_label
from mmpretrain.evaluation.metrics.single_label import to_tensor

@METRICS.register_module()
class CChessAveragePrecision(AveragePrecision):

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


    @staticmethod
    def calculate(pred: Union[torch.Tensor, np.ndarray],
                  target: Union[torch.Tensor, np.ndarray],
                  average: Optional[str] = 'macro') -> torch.Tensor:
        r"""Calculate the average precision for a single class.

        Args:
            pred (torch.Tensor | np.ndarray): The model predictions with
                shape ``(N, num_classes)``.
            target (torch.Tensor | np.ndarray): The target of predictions
                with shape ``(N, num_classes)``.
            average (str | None): The average method. It supports two modes:

                - `"macro"`: Calculate metrics for each category, and calculate
                  the mean value over all categories. The result of this mode
                  is also called mAP.
                - `None`: Calculate metrics of every category and output
                  directly.

                Defaults to "macro".

        Returns:
            torch.Tensor: the average precision of all classes.
        """
        average_options = ['macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specicy from {average_options}.'



        pred = to_tensor(pred)
        target = to_tensor(target)


        pred = pred.view(-1, pred.shape[-1])
        target = target.view(-1, target.shape[-1])
        assert pred.ndim == 2 and pred.shape == target.shape, \
            'Both `pred` and `target` should have shape `(N, num_classes)`.'

        num_classes = pred.shape[1]
        ap = pred.new_zeros(num_classes)
        for k in range(num_classes):
            ap[k] = _average_precision(pred[:, k], target[:, k])
        if average == 'macro':
            return ap.mean() * 100.0
        else:
            return ap * 100

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

        if self.filter_gt_labels is not None:
            target = target[:, :, self.filter_gt_labels]
            pred = pred[:, :, self.filter_gt_labels]

        ap = self.calculate(pred, target, self.average)

        result_metrics = dict()

        if self.average is None:
            result_metrics['AP_classwise'] = ap.detach().cpu().tolist()
        else:
            result_metrics['mAP'] = ap.item()

        return result_metrics
