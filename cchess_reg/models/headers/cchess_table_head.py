from typing import Tuple, List, Union, Sequence, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.structures.utils import format_label
from mmpretrain.registry import MODELS
from mmpretrain.models.heads.multi_label_cls_head import MultiLabelClsHead
from mmpretrain.evaluation.metrics import Accuracy
from cchess_reg.structures import CChessDataSample

@MODELS.register_module()
class CChessTableHead(MultiLabelClsHead):
    """表格检测头部模块
    
    Args:
        num_classes (int): 输出通道数，例如16
        loss (dict): 损失函数配置
    """
    def __init__(self,
                 num_classes: int,
                 loss: Dict = dict(type='CrossEntropyLoss', use_sigmoid=False),
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 cal_acc: bool = True,
                 init_cfg: Optional[dict] = None,
                ):
        super().__init__(loss=loss, init_cfg=init_cfg, thr=thr, topk=topk)

        self.num_classes = num_classes
        self.cal_acc = cal_acc

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        # 保持 统一
        if isinstance(feats, tuple):
            return feats[-1]
        return feats
        
    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """前向传播过程"""
        logits = self.pre_logits(feats)
        # 校验 logits.shape

        assert logits.ndim == 4, 'logits.ndim must be 4'
        assert logits.shape[1:] == (16, 10, 9), 'logits.shape must be [BS, 16, 10, 9]'
        # 转换成 [BS, Cls, H, W] -> [BS, H, W, Cls]
        logits = logits.permute(0, 2, 3, 1)
        # 转换成 [BS, H, W, Cls] -> [BS, H * W, Cls]
        """
        A0 ... A8 B0 ... B8 C0 ...
        """
        logits = logits.view(logits.size(0), -1, logits.size(3))
        return logits

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[CChessDataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[CChessDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses


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

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[CChessDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            # torch.Size([BS * 90]) 个 cls 标签
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        cls_score = cls_score.reshape(-1, cls_score.size(-1))
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:

            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses
    
    def _get_predictions(self, cls_score: torch.Tensor,
                         data_samples: List[CChessDataSample]):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        #cls_score.shape = [BS * 90, Cls]
        # cls_score = cls_score.reshape(-1, cls_score.size(-1))
        # 计算 softmax
        pred_scores = F.softmax(cls_score, dim=-1)
        # 计算 argmax [BS, 90]
        pred_labels = pred_scores.argmax(dim=-1, keepdim=True).detach()

        # # torch.Size([BS, 90]) 提取 最大值,  并移除最后一个维度
        # pred_score_with_softmax = pred_scores.gather(dim=-1, index=pred_labels)

        # 移除最后一个维度
        pred_labels = pred_labels.squeeze(-1)
        # pred_score_with_softmax = pred_score_with_softmax.squeeze(-1)

        if data_samples is None:
            data_samples = [CChessDataSample() for _ in range(cls_score.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores, pred_labels):
            # score fix 提取 最大的 1 个
            # pred_score_with_softmax  
            data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples
    
    def _single_get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        # cls_score.shape = [BS * 90, Cls]
        cls_score = cls_score.reshape(-1, cls_score.size(-1))
        # 计算 softmax  is 
        pred_scores = F.softmax(cls_score, dim=1)
        # 计算 argmax
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = CChessDataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples
