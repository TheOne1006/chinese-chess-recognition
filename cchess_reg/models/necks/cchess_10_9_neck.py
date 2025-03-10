# import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
# from mmcv.cnn import build_activation_layer, build_norm_layer

from mmpretrain.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class CChess10_9Neck(BaseModule):
    """10_9 表格检测头部模块，增强版本包含上下错落的特征提取
    
    该模块专为中国象棋棋盘识别设计，采用多种先进特征提取技术：
    1. 空洞卷积：使用不同膨胀率捕获多尺度特征，扩大感受野
    2. 特征金字塔：自顶向下传递语义信息，增强低层特征表达
    3. 残差连接：保留原始特征，缓解梯度消失问题
    4. 多尺度特征融合：综合利用各层级特征，提高识别精度
    
    上下错落的设计特别适合棋盘识别：
    - 棋盘具有固定的10×9结构，需要精确定位
    - 棋子之间存在上下文关系，位置相互影响
    - 不同尺度特征对识别不同大小和形状的棋子很重要
    
    Args:
        in_channels (int): 输入通道数，
        mid_channels (List[int]): 中间通道数，通常是[256, 128]
        num_classes (int): 输出类别数，通常是16
        use_residual (bool): 是否使用残差连接
        use_dilated (bool): 是否使用空洞卷积
        use_fpn (bool): 是否使用特征金字塔结构
    """
    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: List[int] = [256, 128],
                 num_classes: int = 16,
                 use_residual: bool = True,
                 use_dilated: bool = True,
                 use_fpn: bool = True,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.use_residual = use_residual
        self.use_dilated = use_dilated
        self.use_fpn = use_fpn
        
        # 初始卷积层，降低通道数
        # 使用1×1卷积减少计算量，同时保留空间信息
        self.init_conv = nn.Conv2d(in_channels, mid_channels[0], kernel_size=1, stride=1, padding=0)
        self.init_relu = nn.ReLU(inplace=True)
        
        # 主干特征提取网络
        # 包含常规卷积和空洞卷积两种分支，形成上下错落的特征提取结构
        self.main_layers = nn.ModuleList()
        for i in range(len(mid_channels) - 1):
            # 创建主卷积层 - 常规3×3卷积捕获局部特征
            main_conv = nn.Conv2d(
                mid_channels[i], 
                mid_channels[i+1], 
                kernel_size=3, 
                stride=1, 
                padding=1
            )
            self.main_layers.append(nn.Sequential(main_conv, nn.ReLU(inplace=True)))
            
            # 如果使用空洞卷积，添加平行的空洞卷积分支
            # 空洞卷积扩大感受野，捕获更大范围的上下文信息
            if use_dilated:
                # 不同层使用不同的空洞率，形成多尺度特征提取
                dilation_rate = 2 if i == 0 else 4  # 第一层使用较小空洞率，深层使用较大空洞率
                dilated_conv = nn.Conv2d(
                    mid_channels[i], 
                    mid_channels[i+1], 
                    kernel_size=3, 
                    stride=1, 
                    padding=dilation_rate,
                    dilation=dilation_rate
                )
                self.main_layers.append(nn.Sequential(dilated_conv, nn.ReLU(inplace=True)))
        
        # 如果使用FPN，创建自顶向下的路径
        # FPN结构使高层语义特征能够指导低层特征的学习
        if use_fpn:
            self.fpn_layers = nn.ModuleList()
            for i in range(len(mid_channels) - 1, 0, -1):
                # 使用1×1卷积调整通道数，便于特征融合
                fpn_conv = nn.Conv2d(
                    mid_channels[i], 
                    mid_channels[i-1], 
                    kernel_size=1, 
                    stride=1
                )
                self.fpn_layers.append(nn.Sequential(fpn_conv, nn.ReLU(inplace=True)))
        
        # 创建特征融合层
        # 当同时使用FPN和残差连接时，需要融合所有层级的特征
        if use_fpn and use_residual:
            # 计算融合后的通道数：所有中间层特征的通道数之和
            total_channels = sum(mid_channels)
            # 使用1×1卷积融合多层特征
            self.fusion_conv = nn.Conv2d(total_channels, mid_channels[-1], kernel_size=1)
        
        # 最终输出层
        # 将特征映射到类别空间
        self.final_conv = nn.Conv2d(mid_channels[-1], num_classes, kernel_size=1, stride=1, padding=0)
        
        # 适应棋盘大小的池化层
        # 中国象棋棋盘为10×9，使用自适应池化确保输出尺寸匹配
        self.avg_pool = nn.AdaptiveAvgPool2d((10, 9))
    
    def forward(self, inputs: Union[Tuple, torch.Tensor]) -> Tuple[torch.Tensor]:
        """前向传播函数
        
        实现了上下错落的特征提取流程：
        1. 初始特征提取
        2. 多分支并行处理（常规卷积+空洞卷积）
        3. 特征金字塔自顶向下传递
        4. 多尺度特征融合
        5. 最终分类输出

        Args:
            inputs (Union[Tuple, torch.Tensor]): 从骨干网络提取的特征
                支持多阶段输入，但只会使用最后一个阶段的特征

        Returns:
            Tuple[torch.Tensor]: 输出特征元组
        """
        assert isinstance(inputs, (tuple, torch.Tensor)), (
            'The inputs of `CChess10_9Neck` must be tuple or `torch.Tensor`, '
            f'but get {type(inputs)}.')
        if isinstance(inputs, tuple):
            inputs = inputs[-1]  # 使用最后一层特征

        # 初始卷积 - 降低通道数，保留空间信息
        x = self.init_conv(inputs)
        x = self.init_relu(x)
        
        # 存储中间特征用于残差连接和FPN
        # 这些特征将在后续的上下错落结构中被重复使用
        features = [x]
        
        # 主干特征提取 - 实现上下错落的核心部分
        if self.use_dilated:
            # 使用空洞卷积时，每个层级有两个分支（常规卷积+空洞卷积）
            # 这种并行结构能够同时捕获不同尺度的特征
            for i in range(0, len(self.main_layers), 2):
                main_out = self.main_layers[i](features[-1])  # 常规卷积分支
                dilated_out = self.main_layers[i+1](features[-1])  # 空洞卷积分支
                # 融合两个分支的特征 - 结合不同感受野的信息
                fused = main_out + dilated_out
                features.append(fused)
        else:
            # 不使用空洞卷积时，直接顺序处理
            for layer in self.main_layers:
                out = layer(features[-1])
                features.append(out)
        
        # 如果使用FPN，添加自顶向下的路径
        # 这部分实现了特征金字塔的上下错落信息流
        if self.use_fpn:
            for i, fpn_layer in enumerate(self.fpn_layers):
                idx = len(features) - i - 1
                # 自顶向下传递特征
                top_down = fpn_layer(features[idx])
                # 将自顶向下的特征与相应层级的特征融合
                if self.use_residual:
                    # 使用残差连接保留原始特征信息
                    features[idx-1] = features[idx-1] + top_down
                else:
                    # 直接替换，完全依赖高层特征指导
                    features[idx-1] = top_down
        
        # 使用最终特征 - 多尺度特征融合
        if self.use_fpn and self.use_residual:
            # 如果同时使用FPN和残差连接，融合所有层级的特征
            # 这种多尺度融合能够综合利用不同层级的特征信息
            pooled_features = [self.avg_pool(feat) for feat in features]
            x = torch.cat(pooled_features, dim=1)  # 在通道维度拼接
            x = self.fusion_conv(x)  # 使用1×1卷积融合
        else:
            # 否则只使用最后一层特征
            x = self.avg_pool(features[-1])
        
        # 最终输出 - 映射到类别空间
        x = self.final_conv(x)

        return x

