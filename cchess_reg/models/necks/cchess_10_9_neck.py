# import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
# from mmcv.cnn import build_activation_layer, build_norm_layer

from mmpretrain.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class CChess10_9Neck(BaseModule):
    """10_9 表格检测头部模块
    
    Args:
        in_channels (int): 输入通道数，通常是2048
        mid_channels (List[int]): 中间通道数，通常是[512, 256, 128]
        num_classes (int): 输出类别数，通常是16
    """
    def __init__(self,
                 in_channels: int = 2048,
                 mid_channels: List[int] = [512, 256, 128],
                 num_classes: int = 16,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        
        conv_layers = []
        for i in range(len(mid_channels)):
            if i == 0:
                conv_layers.append(nn.Conv2d(in_channels, mid_channels[i], kernel_size=1, stride=1, padding=0))
            else:
                conv_layers.append(nn.Conv2d(mid_channels[i - 1], mid_channels[i], kernel_size=3, stride=1, padding=1))
                
            # 增加一个 ReLU 激活函数
            conv_layers.append(nn.ReLU(inplace=True))

        final_conv = nn.Conv2d(mid_channels[-1], num_classes, kernel_size=1, stride=1, padding=0)

        # 增加一个平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d((10, 9))

        self.conv = nn.Sequential(*conv_layers)

        self.final_conv = final_conv

    
    
    def forward(self, inputs: Union[Tuple,
                                    torch.Tensor]) -> Tuple[torch.Tensor]:
        """forward function.

        Args:
            inputs (Union[Tuple, torch.Tensor]): The features extracted from
                the backbone. Multiple stage inputs are acceptable but only
                the last stage will be used.

        Returns:
            Tuple[torch.Tensor]: A tuple of output features.
        """
        assert isinstance(inputs, (tuple, torch.Tensor)), (
            'The inputs of `LinearNeck` must be tuple or `torch.Tensor`, '
            f'but get {type(inputs)}.')
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        x = self.conv(inputs)
        x = self.avg_pool(x)
        x = self.final_conv(x)

        return x

