# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch.nn.functional as F
import torch
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement
from torch import nn

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_root_logger)
from mmdeploy.codebase.mmpretrain.deploy.classification_model import __BACKEND_MODEL

# __BACKEND_MODEL = Registry('backend_classifiers')


@__BACKEND_MODEL.register_module('end2end', force=True)
class End2EndModel(BaseBackendModel):
    """End to end model for inference of classification.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (str | Config): Deployment config file or loaded Config
            object.
        data_preprocessor (BaseDataPreprocessor): The data preprocessor
                of the model. Default to `None`.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: Union[str, Config] = None,
                 deploy_cfg: Union[str, Config] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self._init_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            **kwargs)
        self.model_cfg = model_cfg
        self.head = None
        if model_cfg is not None:
            self.head = self._get_head()
        self.device = device

    def _get_head(self):
        from mmpretrain.models import build_head
        head_config = self.model_cfg['model']['head']
        head = build_head(head_config)
        return head

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str, **kwargs):
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg,
            **kwargs)

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
            pred_scores = F.softmax(cls_score, dim=-1)
            pred_labels = pred_scores.argmax(dim=-1, keepdim=True).detach()
            pred_labels = pred_labels.squeeze(-1)

            for data_sample, score, label in zip(data_samples, pred_scores, pred_labels):
                data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples
