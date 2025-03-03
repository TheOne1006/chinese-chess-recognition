from mmcv.transforms.processing import RandomFlip
import mmcv
from mmpretrain.registry import TRANSFORMS
import numpy as np
from typing import Tuple

@TRANSFORMS.register_module()
class CChessRandomFlip(RandomFlip):
    """中国象棋数据随机翻转类。
    
    继承自 mmcv 的 RandomFlip 类，用于对中国象棋图像和标签进行随机翻转增强。
    除了翻转图像外，还会相应地调整标签数据，保持图像和标签的一致性。
    
    支持水平翻转、垂直翻转和对角线翻转三种方式。
    """

    def _flip_label(self, label: np.ndarray,
                   direction: str) -> np.ndarray:   
        """翻转标签数据。
        
        根据指定的翻转方向，对中国象棋标签数据进行相应的翻转操作。
        标签被视为10行9列的棋盘布局，根据翻转方向调整位置。
        
        Args:
            label (np.ndarray): 原始标签数据
            direction (str): 翻转方向，可选 'horizontal'、'vertical' 或 'diagonal'
            
        Returns:
            list: 翻转后的标签数据列表
        """
        # 判断 label 只是 list 类型
        if isinstance(label, list):
            label = np.array(label)
        # label
        label_10_9 = np.array(label).reshape(-1, 9)
        if direction == 'horizontal':
            # 水平翻转：每行元素顺序反转
            label_10_9 = label_10_9[:, ::-1]
        elif direction == 'vertical':
            # 垂直翻转：行的顺序反转
            label_10_9 = label_10_9[::-1, :]
        elif direction == 'diagonal':
            # 对角线翻转：先水平后垂直（或先垂直后水平）
            label_10_9 = label_10_9[:, ::-1]
            label_10_9 = label_10_9[::-1, :]
        return label_10_9.reshape(-1).tolist()


    
    def _flip(self, results: dict) -> None:
        """翻转图像、边界框、语义分割图和关键点。
        
        重写父类的 _flip 方法，在翻转图像的同时，也对中国象棋特有的标签数据进行翻转。
        
        Args:
            results (dict): 包含图像和标签数据的字典
        """
        # 翻转图像
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        # 翻转标签
        if results.get('gt_label', None) is not None:
            results['gt_label'] = self._flip_label(
                results['gt_label'],
                results['flip_direction'])

