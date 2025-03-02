import numpy as np
import random
from typing import List, Optional, Tuple, Dict
# from mmpose.utils import cache_randomness
from mmcv.transforms.utils import cache_randomness
from mmpose.datasets.transforms.common_transforms import BaseTransform
from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomHalfCChess4(BaseTransform):
    """Data augmentation with half-body transform that keeps only the upper or
    lower body at random.

    Required Keys:

        - keypoints
        - keypoints_visible
        - upper_body_ids
        - lower_body_ids

    Modified Keys:

        - bbox
        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.5
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3
    """

    def __init__(self,
                 padding: float = 1.5,
                 prob: float = 0.3) -> None:
        super().__init__()
        
        self.padding = padding
        self.prob = prob

        self.min_total_keypoints = 2

        # A0, A8, J0, J8, 0,1,2,3
        self.match_group_ids = [
            [0, 1], # A0, A1
            [0, 2], # A0, J0
            [2, 3], # J0, J8
            [1, 3], # A8, J8
        ]

    def _get_half_body_bbox(self, keypoints: np.ndarray,
                            half_body_ids: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Get half-body bbox center and scale of a single instance.

        Args:
            keypoints (np.ndarray): Keypoints in shape (K, D)
            upper_body_ids (list): The list of half-body keypont indices

        Returns:
            tuple: A tuple containing half-body bbox center and scale
            - center: Center (x, y) of the bbox
            - scale: Scale (w, h) of the bbox
        """

        selected_keypoints = keypoints[half_body_ids]
        center = selected_keypoints.mean(axis=0)[:2]

        x1, y1 = selected_keypoints.min(axis=0)
        x2, y2 = selected_keypoints.max(axis=0)
        w = x2 - x1
        h = y2 - y1
        scale = np.array([w, h], dtype=center.dtype) * self.padding

        return center, scale

    @cache_randomness
    def _random_select_half_body(self, keypoints_visible: np.ndarray) -> List[Optional[List[int]]]:
        """Randomly determine whether applying half-body transform and get the
        half-body keyponit indices of each instances.

        Args:
            keypoints_visible (np.ndarray, optional): The visibility of
                keypoints in shape (N, K, 1) or (N, K, 2).

        Returns:
            list[list[int] | None]: The selected half-body keypoint indices
            of each instance. ``None`` means not applying half-body transform.
        """

        if keypoints_visible.ndim == 3:
            keypoints_visible = keypoints_visible[..., 0]

        half_body_ids = []
        allow_ids = [
            0, 1, 2, 3
        ]

        for visible in keypoints_visible:
            if visible.sum() < self.min_total_keypoints:
                indices = None
            elif np.random.rand() > self.prob:
                indices = None
            else:
                valid_ids = [i for i in allow_ids if visible[i] > 0]

                set_valid_ids = set(valid_ids)

                # 随机选择一个 group
                # match_group_ids 需要满足 valid_ids
                allow_group_ids = [ids for ids in self.match_group_ids if set(ids).issubset(set_valid_ids)]

                if len(allow_group_ids) == 0:
                    indices = None
                else:
                    
                    # 随机选择一个 group
                    indices = random.choice(allow_group_ids)
                    # print("执行 random half cchess 4 with group ids: ", indices)

            half_body_ids.append(indices)

        return half_body_ids

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        half_body_ids = self._random_select_half_body(
            keypoints_visible=results['keypoints_visible'])

        bbox_center = []
        bbox_scale = []

        for i, indices in enumerate(half_body_ids):
            if indices is None:
                bbox_center.append(results['bbox_center'][i])
                bbox_scale.append(results['bbox_scale'][i])
            else:
                _center, _scale = self._get_half_body_bbox(
                    results['keypoints'][i], indices)
                bbox_center.append(_center)
                bbox_scale.append(_scale)

        results['bbox_center'] = np.stack(bbox_center)
        results['bbox_scale'] = np.stack(bbox_scale)
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
