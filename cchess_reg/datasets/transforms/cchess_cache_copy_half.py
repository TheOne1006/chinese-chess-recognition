from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS
# from typing import Tuple, List
import numpy as np
import random
import copy
# import mmcv
from mmcv.transforms.utils import cache_randomness


@TRANSFORMS.register_module()
class CChessCachedCopyHalf(BaseTransform):
    """CChessCachedCopyHalf data augmentation.

    .. code:: text

                           cchess copy half transform
            +-----------------------------------------------+
            |                                               |
            |                  img1                         |
            |                                               |
            |-----------+------------+-----------+----------|
            |                                               |
            |                                               |
            |                    img2                       |
            +-----------------------------------------------+
                                  Image
    
    Required Keys:

        - img
        - gt_label (np.int64)
    
    Modified Keys:

        - img
        - gt_label


    Args:
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 cache_size: int = 100,
                 prob: float = .5) -> None:
        self.prob = prob
        self.results_cache = []
        self.cache_size = cache_size
        
    @cache_randomness
    def get_cache_item(self):

        cache = self.results_cache

        if len(cache) == 0:
            return None
        
        return random.choice(cache)
    

    def _after_transform(self, results: dict):
        # 如果 cache 超过 cache_size，则随机删除一个
        if len(self.results_cache) > self.cache_size:
            self.results_cache.pop(np.random.randint(0, len(self.results_cache)))

        self.results_cache.append(copy.deepcopy(results))




    def transform(self, results: dict) -> dict:
        
        origin_results = copy.deepcopy(results)

        cache_item = self.get_cache_item()

        if cache_item is None:
            self._after_transform(origin_results)
            return results
        

        # 上半部分图片
        img_h, _ = results['img'].shape[:2]
        half_img_h = img_h // 2

        # 百分之五十 覆盖
        if np.random.rand() > 0.5:
            # 上半部分图片 和 gt
            results['img'][:half_img_h, :, :] = cache_item['img'][:half_img_h, :, :]
            results['gt_label'][:45] = cache_item['gt_label'][:45]
        else:
          results['img'][half_img_h:, :, :] = cache_item['img'][half_img_h:, :, :]
          results['gt_label'][45:] = cache_item['gt_label'][45:]

   
        self._after_transform(origin_results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(cache_size={self.cache_size}, '
        repr_str += f'prob={self.prob})'
        return repr_str



