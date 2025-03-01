import warnings
import numpy as np
from mmpose.structures.bbox import bbox_xyxy2cs
from mmpose.datasets.transforms.common_transforms import BaseTransform
from mmpose.registry import TRANSFORMS
from mmengine.dist import get_dist_info
from typing import List, Optional, Dict

@TRANSFORMS.register_module()
class RandomGetBBoxCenterScale(BaseTransform):
    """
    随机获取 bbox 的中心和缩放
    """
    def __init__(self, paddings: List[float] = [1.1, 1.3]) -> None:
        super().__init__()
        self.paddings = paddings


    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        if 'bbox_center' in results and 'bbox_scale' in results:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('Use the existing "bbox_center" and "bbox_scale"'
                              '. The padding will still be applied.')
            results['bbox_scale'] = results['bbox_scale'] * self.padding

        else:
            bbox = results['bbox']
            padding = np.random.choice(self.paddings)
            center, scale = bbox_xyxy2cs(bbox, padding=padding)

            results['bbox_center'] = center
            results['bbox_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(padding={self.padding})'
        return repr_str


