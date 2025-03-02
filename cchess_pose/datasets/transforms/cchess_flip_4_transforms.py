from typing import List,Union

import numpy as np
# from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
# from mmengine import is_list_of
from mmcv.image import imflip
from mmpose.structures.bbox import flip_bbox
from mmpose.structures.keypoint import flip_keypoints

# from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

from mmpose.datasets.transforms.common_transforms import RandomFlip



@TRANSFORMS.register_module()
class CChessRandomFlip4(RandomFlip):
    """
    A0, A8
    J0, J8
    """

    base_keypoint = [
        "A0", "A8",
        "J0", "J8",
    ]

    # 水平（左右）翻转 用于 上下分布
    flip_idx_horizontal_with_top_down_keypoints = [
        "A8", "A0",
        "J8", "J0",
    ]
    # 上下翻转 用于 上下分布
    flip_idx_vertical_with_top_down_keypoints = [
        "A8", "A0",
        "J8", "J0",
    ]

    # 水平（左右）翻转 用于 左右分布
    flip_idx_horizontal_with_left_right_keypoints = [
        "A8", "A0",
        "J8", "J0",
    ]

    # 上下翻转 用于 左右分布
    flip_idx_vertical_with_left_right_keypoints = [
        "A8", "A0",
        "J8", "J0",
    ]

    flip_idx_diagonal_keypoints = [
        "J8", "J0",
        "A8", "A0",
    ]

    def __init__(self,
                prob: Union[float, List[float]] = 0.5,
                direction: Union[str, List[str]] = 'horizontal') -> None:
        super().__init__(prob, direction)
    
        # 水平（左右）翻转 用于 上下分布, 根据 self.flip_idx_horizontal_with_top_down_keypoints 生成
        self.flip_idx_horizontal_with_top_down = [
            self.base_keypoint.index(keypoint) for keypoint in self.flip_idx_horizontal_with_top_down_keypoints
        ]
        # 上下翻转 用于 上下分布
        self.flip_idx_vertical_with_top_down = [
            self.base_keypoint.index(keypoint) for keypoint in self.flip_idx_vertical_with_top_down_keypoints
        ]
        # 水平（左右）翻转 用于 左右分布
        self.flip_idx_horizontal_with_left_right = [
            self.base_keypoint.index(keypoint) for keypoint in self.flip_idx_horizontal_with_left_right_keypoints
        ]
        # 上下翻转 用于 左右分布
        self.flip_idx_vertical_with_left_right = [
            self.base_keypoint.index(keypoint) for keypoint in self.flip_idx_vertical_with_left_right_keypoints
        ]

        # 对角线翻转
        self.flip_idx_diagonal = [
            self.base_keypoint.index(keypoint) for keypoint in self.flip_idx_diagonal_keypoints
        ]
    
    def is_top_down(self, results: dict) -> bool:

        original_keypoints = results['keypoints'].copy()

        assert original_keypoints.shape == (1, 4, 2)

        A0_idx = self.base_keypoint.index("A0")
        J0_idx = self.base_keypoint.index("J0")
        A0_keypoints = original_keypoints[:, A0_idx, :]
        J0_keypoints = original_keypoints[:, J0_idx, :]

        # 上下分布
        top_down_flag = abs(A0_keypoints[0, 1] - J0_keypoints[0, 1]) > abs(A0_keypoints[0, 0] - J0_keypoints[0, 0])

        return top_down_flag

    def transform(self, results: dict) -> dict:
        flip_dir = self._choose_direction()
         
        if flip_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = flip_dir

            h, w = results.get('input_size', results['img_shape'])
            # flip image and mask
            if isinstance(results['img'], list):
                results['img'] = [
                    imflip(img, direction=flip_dir) for img in results['img']
                ]
            else:
                results['img'] = imflip(results['img'], direction=flip_dir)

            if 'img_mask' in results:
                results['img_mask'] = imflip(
                    results['img_mask'], direction=flip_dir)

            # flip bboxes
            if results.get('bbox', None) is not None:
                results['bbox'] = flip_bbox(
                    results['bbox'],
                    image_size=(w, h),
                    bbox_format='xyxy',
                    direction=flip_dir)
            if results.get('bbox_center', None) is not None:
                results['bbox_center'] = flip_bbox(
                    results['bbox_center'],
                    image_size=(w, h),
                    bbox_format='center',
                    direction=flip_dir)

            # flip keypoints
            if results.get('keypoints', None) is not None:

                flip_indices = [
                    # 0, 1, 2, ..., 5
                    i for i in range(6)
                ]

                if flip_dir == "vertical" or flip_dir == "horizontal":
                    is_top_down = self.is_top_down(results)

                    if flip_dir == "vertical" and is_top_down:
                        # print("上下分布 时 上下翻转")
                        flip_indices = self.flip_idx_vertical_with_top_down
                    elif flip_dir == "vertical" and not is_top_down:
                        # print("左右分布 时 上下翻转")
                        flip_indices = self.flip_idx_vertical_with_left_right
                    elif flip_dir == "horizontal" and is_top_down:
                        # print("上下分布 时 左右翻转")
                        flip_indices = self.flip_idx_horizontal_with_top_down
                    elif flip_dir == "horizontal" and not is_top_down:
                        # print("左右分布 时 左右翻转")
                        flip_indices = self.flip_idx_horizontal_with_left_right
                # elif flip_dir == "diagonal":
                #     flip_indices = self.flip_idx_diagonal

                if flip_dir == "diagonal":
                    keypoints, keypoints_visible = flip_keypoints(
                        results['keypoints'],
                        results.get('keypoints_visible', None),
                        image_size=(w, h),
                        flip_indices=self.flip_idx_horizontal_with_left_right,
                        direction='horizontal')
                    
                    keypoints, keypoints_visible = flip_keypoints(
                        keypoints,
                        keypoints_visible,
                        image_size=(w, h),
                        flip_indices=self.flip_idx_vertical_with_left_right,
                        direction='vertical')

                else:
                    keypoints, keypoints_visible = flip_keypoints(
                        results['keypoints'],
                        results.get('keypoints_visible', None),
                        image_size=(w, h),
                        flip_indices=flip_indices,
                        direction=flip_dir)
                    
                    results['flip_indices'] = flip_indices
                    
                results['keypoints'] = keypoints
                results['keypoints_visible'] = keypoints_visible
                results['transformed_keypoints'] = keypoints.copy()

        return results
