import numpy as np
# import random
from typing import List, Optional, Tuple, Dict
# from mmpose.utils import cache_randomness
from mmcv.transforms.utils import cache_randomness
from mmpose.datasets.transforms.common_transforms import BaseTransform
from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomUseFullImg(BaseTransform):
    """Data augmentation with random use full img as bbox

    Required Keys:
        - ori_shape

    Modified Keys:

        - bbox
        - bbox_center
        - bbox_scale

    Args:
        prob (float): The probability to use full img as bbox
    """

    def __init__(self,
                 prob: float = 0.3,
                 ) -> None:
        super().__init__()
        self.prob = prob


    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = results['ori_shape']


        if np.random.rand() > self.prob:
            results['bbox'][0] = [0, 0, w, h]

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(min_total_keypoints={self.min_total_keypoints}, '
        repr_str += f'padding={self.padding}, '
        repr_str += f'prob={self.prob}, '
        return repr_str
