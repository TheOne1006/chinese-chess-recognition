from mmcv.transforms.processing import RandomFlip
import mmcv
from mmpretrain.registry import TRANSFORMS
import numpy as np
from typing import Tuple

@TRANSFORMS.register_module()
class CChessRandomFlip(RandomFlip):


    def _flip_label(self, label: np.ndarray, img_shape: Tuple[int, int],
                   direction: str) -> np.ndarray:   
        # 判断 label 只是 list 类型
        if isinstance(label, list):
            label = np.array(label)
        # label
        label_10_9 = np.array(label).reshape(-1, 9)
        if direction == 'horizontal':
            label_10_9 = label_10_9[:, ::-1]
        elif direction == 'vertical':
            label_10_9 = label_10_9[::-1, :]
        elif direction == 'diagonal':
            label_10_9 = label_10_9[:, ::-1]
            label_10_9 = label_10_9[::-1, :]
        return label_10_9.reshape(-1).tolist()


    
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_label', None) is not None:
            results['gt_label'] = self._flip_label(results['gt_label'],
                                                   img_shape,
                                                   results['flip_direction'])

