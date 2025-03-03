# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.structures import BaseDataElement
from torch import Tensor
from torch.nn import functional as F

from mmdeploy.core import FUNCTION_REWRITER

from mmpretrain.structures import DataSample, MultiTaskDataSample
from cchess_reg.models.headers.cchess_table_head import CChessTableHead

#  MultiTaskDataSample 强制增加 set_pred_score
MultiTaskDataSample.set_pred_score = DataSample.set_pred_score
MultiTaskDataSample.set_pred_label = DataSample.set_pred_label

@FUNCTION_REWRITER.register_rewriter(
    'mmpretrain.models.classifiers.ImageClassifier.forward', backend='default')
# @FUNCTION_REWRITER.register_rewriter(
#     'mmpretrain.models.classifiers.BaseClassifier.forward', backend='default')
def multiheader_classifier__forward(
        self,
        batch_inputs: Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
        mode: str = 'predict'):
    """Rewrite `forward` of Classification for default backend.

    Args:
        batch_inputs (torch.Tensor): The input tensor with shape
            (N, C, ...) in general.
        data_samples (List[BaseDataElement], optional): The annotation
            data of every samples. It's required if ``mode="loss"``.
            Defaults to None.
        mode (str): Return what kind of value. Defaults to 'predict'.

    Returns:
        return a list of :obj:`mmengine.BaseDataElement`.
    """
    output = self.extract_feat(batch_inputs)
    if self.head is not None:
        output = self.head(output)

    from mmpretrain.models.heads import ConformerHead, MultiLabelClsHead, MultiTaskHead

    if isinstance(self.head, CChessTableHead):
        output = F.softmax(output, dim=-1)
    elif isinstance(self.head, MultiLabelClsHead):
        output = torch.sigmoid(output)
    elif isinstance(self.head, ConformerHead):
        output = F.softmax(torch.add(output[0], output[1]), dim=1)
    elif isinstance(self.head, MultiTaskHead):

        # todo，处理第一个
        # output is dict

        output = output['main_cls']
        output = F.softmax(output, dim=1)

    else:
        output = F.softmax(output, dim=1)
    return output
