# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Sequence, Union

import torch.nn.functional as F
import torch
from mmengine.structures import BaseDataElement
from mmdeploy.utils import (Backend, get_root_logger)
from mmdeploy.codebase.mmpretrain.deploy.classification_model import __BACKEND_MODEL, End2EndModel


@__BACKEND_MODEL.register_module('end2end', force=True)
class End2EndModelExt(End2EndModel):
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict') -> Any:
        """Run forward inference.

        Args:
            inputs (torch.Tensor): The input tensors
            data_samples (List[BaseDataElement], optional): The data samples.
                Defaults to None.
            mode (str, optional): forward mode, only support `predict`.

        Returns:
            Any: Model output.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        if inputs.device != torch.device(self.device):
            get_root_logger().warning(f'expect input device {self.device}'
                                      f' but get {inputs.device}.')
        inputs = inputs.to(self.device)
        cls_score = self.wrapper({self.input_name:
                                  inputs})[self.output_names[0]]

        from mmpretrain.models.heads import MultiLabelClsHead
        from mmpretrain.structures import DataSample
        pred_scores = cls_score

        if self.head is None or not isinstance(self.head, MultiLabelClsHead):
            pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

            if data_samples is not None:
                for data_sample, score, label in zip(data_samples, pred_scores,
                                                     pred_labels):
                    data_sample.set_pred_score(score).set_pred_label(label)
            else:
                data_samples = []
                for score, label in zip(pred_scores, pred_labels):
                    data_samples.append(DataSample().set_pred_score(
                        score).set_pred_label(label))
        else:
            if data_samples is None:
                data_samples = [DataSample() for _ in range(cls_score.size(0))]
            
            # fix to support cchess header
            pred_labels = cls_score.argmax(dim=-1, keepdim=True).detach()
            pred_labels = pred_labels.squeeze(-1)

            for data_sample, score, label in zip(data_samples, cls_score, pred_labels):
                data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples
