from typing import Optional, Sequence, Tuple, Union

import torch
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
# from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
# from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList)
from mmpose.models.heads.coord_cls_heads import RTMCCHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMCCHead2(RTMCCHead):

    def __init__(
        self,
        loss2: ConfigType = dict(
            type='MSELoss', 
            use_target_weight=True),
        **kwarg,
    ):
        super().__init__(**kwarg)
        self.loss2_name = loss2.get('type')
        self.loss_module2 = MODELS.build(loss2)
 
    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)
        # [BS, keypoint_num, hidden_dims * 2]
        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        
        # [BS, keypoint_num, hidden_dims * 2]
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)

        # 添加 bone loss
        # 将simcc预测转换为坐标点
        pred_coords = self.simcc_to_coords(pred_x, pred_y)  # [BS, keypoint_num, 2] 
        gt_coords = self.simcc_to_coords(gt_x, gt_y)      # [BS, keypoint_num, 2]

        # print("---------")
        # print(pred_coords[0])
        # print(gt_coords[0])
        
        # 计算 mse loss
        # print("pred_coords.shape", pred_coords.shape)
        # print("gt_coords.shape", gt_coords.shape)
        # print("keypoint_weights.shape", keypoint_weights.shape)
        
        # target_weight (torch.Tensor[N, K, 2]):
        keypoint_weights_2d = keypoint_weights.unsqueeze(-1).repeat(1, 1, 2)
        loss2_loss = self.loss_module2(pred_coords, gt_coords, keypoint_weights_2d)

        # 添加到losses字典 self.loss2_name
        losses.update(loss_smooth=loss2_loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses
    

    def simcc_to_coords(self, simcc_x, simcc_y):
        """将SimCC预测转换为坐标点, 保留梯度"""
        # 对预测结果进行 softmax，确保概率和为1
        simcc_x = torch.softmax(simcc_x, dim=2)
        simcc_y = torch.softmax(simcc_y, dim=2)

        # 创建坐标索引
        x_indices = torch.arange(simcc_x.size(2), dtype=torch.float32, device=simcc_x.device)
        y_indices = torch.arange(simcc_y.size(2), dtype=torch.float32, device=simcc_y.device)

        # 计算期望值（加权平均）得到坐标
        x_coords = torch.sum(simcc_x * x_indices, dim=2) / self.simcc_split_ratio
        y_coords = torch.sum(simcc_y * y_indices, dim=2) / self.simcc_split_ratio

        return torch.stack([x_coords, y_coords], dim=-1)  # [B, K, 2]
