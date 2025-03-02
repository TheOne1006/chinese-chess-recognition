from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS
import numpy as np
import random

@TRANSFORMS.register_module()
class CChessHalfFlip(BaseTransform):
    """中国象棋半边镜像数据增强。

    支持以下镜像模式:
    - 左右半边镜像
    - 上下半边镜像

    Args:
        flip_mode (str): 镜像模式，可选 'horizontal' 或 'vertical'。默认为 'horizontal'
        prob (float): 应用此转换的概率。默认为 0.5
    """

    def __init__(self, 
                 flip_mode: str = 'horizontal',
                 prob: float = 0.5) -> None:
        assert flip_mode in ['horizontal', 'vertical']
        assert 0 <= prob <= 1.0

        self.flip_mode = flip_mode
        self.prob = prob

    def transform(self, results: dict) -> dict:
        """执行半边镜像转换。

        Args:
            results (dict): 包含图像和标签的结果字典

        Returns:
            dict: 更新后的结果字典
        """
        if random.random() > self.prob:
            return results

        img = results['img']
        h, w = img.shape[:2]

        if self.flip_mode == 'horizontal':
            # 确保宽度是偶数，如果是奇数则向下取整
            # 向下取整
            cell_w = w // 9
            mid_w = cell_w * 4
            

            if random.random() < 0.5:
                # 镜像图片
                flip_half_img = np.fliplr(img[:, :mid_w])
                # 根据 flip_half_img 的宽度进行裁剪
                source_mid_w = w - flip_half_img.shape[1]
                # 左半边镜像到右边
                img[:, source_mid_w:] = flip_half_img
                if results.get('gt_label', None) is not None:
                    label = np.array(results['gt_label']).reshape(10, 9)
                    label[:, 4:] = np.fliplr(label[:, :5])
                    results['gt_label'] = label.reshape(-1).tolist()
            else:

                # 镜像图片
                mid_w = cell_w * 5
                flip_half_img = np.fliplr(img[:, mid_w:])
                source_mid_w = flip_half_img.shape[1]

                # 右半边镜像到左边
                img[:, :source_mid_w] = flip_half_img
                if results.get('gt_label', None) is not None:
                    label = np.array(results['gt_label']).reshape(10, 9)
                    label[:, :5] = np.fliplr(label[:, 4:])
                    results['gt_label'] = label.reshape(-1).tolist()
        else:
            # 确保高度是偶数，如果是奇数则向下取整
            mid_h = h // 2
            
            if random.random() < 0.5:
                # 镜像图片
                flip_half_img = np.flipud(img[:mid_h, :])

                # 根据 flip_half_img 的高度进行裁剪
                source_mid_h = h - flip_half_img.shape[0]

                # 上半部分镜像到下面
                img[source_mid_h:, :] = flip_half_img
                if results.get('gt_label', None) is not None:
                    label = np.array(results['gt_label']).reshape(10, 9)
                    label[5:, :] = np.flipud(label[:5, :])
                    results['gt_label'] = label.reshape(-1).tolist()
            else:
                # 镜像图片
                flip_half_img = np.flipud(img[mid_h:, :])

                 # 根据 flip_half_img 的高度进行裁剪
                source_mid_h = flip_half_img.shape[0]

                # 下半部分镜像到上面
                img[:source_mid_h, :] = flip_half_img
                if results.get('gt_label', None) is not None:
                    label = np.array(results['gt_label']).reshape(10, 9)
                    label[:5, :] = np.flipud(label[5:, :])
                    results['gt_label'] = label.reshape(-1).tolist()

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(flip_mode={self.flip_mode}, '
        repr_str += f'prob={self.prob})'
        return repr_str 
