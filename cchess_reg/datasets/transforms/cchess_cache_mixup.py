from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS
from typing import Tuple, List
import numpy as np
import random
import copy
import mmcv
from mmcv.transforms.utils import cache_randomness



# Deprecated, 使用 `CChessMixSinglePngCls` 替代
@TRANSFORMS.register_module()
class CChessCachedMixUp(BaseTransform):
    """CChessCached mixup data augmentation.

    .. code:: text

                           cchess mixup transform
            +-----------------------------------------------+
            |   img1 A0 |  img2 A1   |  ...An    | imgN A9  |
            |-----------+------------+-----------+----------|
            |   imgN B0 |            |           |          |
            |-----------+------------+-----------+----------|
            |   .       |            |           |          |
            |   .       |            |           |          |
            |   .       |            |           |          |
            |   .       |            |           |          |
            |-----------+------------+-----------+----------|
            |   J0      |            |           |          |
            +-----------------------------------------------+
                                  Image

    The cached mixup transform steps are as follows:
        1. 将最后一次转换的结果附加到缓存中。
        2. 随机从缓存中抽取 N 张图像，
        3. 随机镶嵌到 A0 - J8 对应位置上，以及修改对应 label
    
    Required Keys:

        - img
        - gt_label (np.int64)
        - mix_results (List[dict])
        - img_path (str)
    
    Modified Keys:

        - img
        - gt_label


    Args:
        img_scale (Tuple[int, int]): Image output size after mixup pipeline. 
            The shape order should be (width, height). Defaults to (640, 640).
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        max_mix_cells (int): The maximum number of cells to be mixed up.
            Defaults to 10.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        rotate_angle (Tuple[int, int]): The range of the rotation angle.
            Defaults to (-180, 180).
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (400, 450),
                 max_cached_images: int = 20,
                 max_mix_cells: int = 10,
                 random_pop: bool = True,
                 rotate_angle: Tuple[int, int] = (-180, 180),
                 unmatch_file_name_startwith: str = "js",
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple)
        assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                       f'but got {max_cached_images}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        

        self.img_scale = img_scale
        self.max_mix_cells = max_mix_cells
        self.rotate_angle = rotate_angle
        self.results_cache = []
        self.unmatch_file_name_startwith = unmatch_file_name_startwith
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = prob

        cchess_table_10x9 = np.zeros((10, 9, 2), dtype=np.float32)

        item_cell_width = img_scale[0] / 9
        item_cell_height = img_scale[1] / 10
        self.item_cell_width = item_cell_width
        self.item_cell_height = item_cell_height

        # 初始化 棋盘 位置
        for row in range(10):
            for col in range(9):
                cchess_table_10x9[row, col] = [
                    col * item_cell_width,
                    row * item_cell_height
                ]
        
        self.cchess_table_10x9 = cchess_table_10x9
        
    @cache_randomness
    def get_indexes(self, cache: list, select_images_num: int) -> List[int]:
        """Call function to collect indexes.

        Args:
            cache (list): The result cache.
            select_images_num (int): The number of images to be selected from the cache.

        Returns:
            List[int]: indexes.
        """

        # 从 cache 中随机选择 select_images_num 个图像

        indexes = random.sample(range(len(cache)), select_images_num)

        return indexes


    def get_cell_xywh(self, cell_index: int) -> np.ndarray:
        """Get the x, y of a cell.
        """
        row = cell_index // 9
        col = cell_index % 9

        return self.cchess_table_10x9[row, col]
    
    def crop_cell_img(self, cell_index: int, img: np.ndarray) -> np.ndarray:
        """Crop the cell image from the original image.
        """
        x, y = self.get_cell_xywh(cell_index)
        w = self.item_cell_width
        h = self.item_cell_height

        # to int
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # hwc
        crop_img = img[y:y+h, x:x+w]
        # 旋转
        if self.rotate_angle[0] != 0 or self.rotate_angle[1] != 0:

            # 随机选择旋转角度
            random_angle = random.uniform(
                self.rotate_angle[0], 
                self.rotate_angle[1]
            )
            # 旋转图像
            crop_img = mmcv.imrotate(crop_img, random_angle)

        return crop_img
    
    def paste_cell_img(self, cell_index: int, img: np.ndarray, cell_img: np.ndarray) -> np.ndarray:
        """Paste the cell image to the original image.
        """
        x, y = self.get_cell_xywh(cell_index)

        w = cell_img.shape[1]
        h = cell_img.shape[0]
    
        # to int
        half_w = int(w // 2)
        half_h = int(h // 2)
        quarter_w = int(w // 4)
        quarter_h = int(h // 4)

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # .shape = (h, w)
        # cell_mask = (cell_img == [0, 0, 0]).all(axis=-1).astype(np.uint8) 

        # 创建 cell_mask (45, 44, 3) 
        cell_mask = (cell_img == [0, 0, 0]).all(axis=-1).astype(np.uint8)

        # 增加一个维度
        cell_mask = cell_mask[..., np.newaxis]
        cell_safe_x = int(half_w - quarter_w)
        cell_safe_y = int(half_h - quarter_h)

        cell_mask[cell_safe_y:cell_safe_y+half_h, cell_safe_x:cell_safe_x+half_w] = 0

        # 粘贴 with mask
        origin_img_part = img[y:y+h, x:x+w]
        img[y:y+h, x:x+w] = cell_img * (1 - cell_mask) + origin_img_part *  cell_mask
        return img

    def transform(self, results: dict) -> dict:
        """CChessCachedMixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        # 1. 结果的 文件名 以 unmatch_startwith 开头，则不加入缓存
        # 2. 结果的 label 存在 1(other)，则不加入缓存

        file_name = results['img_path'].split('/')[-1]
        if not file_name.startswith(self.unmatch_file_name_startwith) and 1 not in results['gt_label']:
            self.results_cache.append(copy.deepcopy(results))

        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 1:
            return results

        if random.uniform(0, 1) > self.prob:
            return results
        
        select_index = random.randint(0, len(self.results_cache) - 1)

        # 从 cache 中获取 select_images_num 个图像
        retrieve_result = copy.deepcopy(self.results_cache[select_index])

        # 1. 随机选择 mix_cells 个 cells

        # 覆盖的 cells 数量
        cover_cells_num = random.randint(max(1, self.max_mix_cells//2), self.max_mix_cells)
        # 随机获取 贴图的 cells 索引
        retrieve_result_not_point_indexes = [
            index for index, label in enumerate(retrieve_result['gt_label']) if label > 0
        ]
        crop_cell_indexes = random.sample(retrieve_result_not_point_indexes, 
                                          min(cover_cells_num, len(retrieve_result_not_point_indexes)))
        # 随机获取 贴图的 cells 图像
        crop_cell_imgs = [self.crop_cell_img(index, retrieve_result['img']) for index in crop_cell_indexes]


        # 个数 与 有效 复制 cells 同步
        paste_cell_indexes = random.sample(range(90), len(crop_cell_indexes))
        # 随机获取 粘贴图的 cells 图像
        for paste_cell_index, crop_cell_img, crop_cell_index in zip(paste_cell_indexes, crop_cell_imgs, crop_cell_indexes):
            # 修改 图片
            results['img'] = self.paste_cell_img(paste_cell_index, results['img'], crop_cell_img)
            # 修改 label
            results['gt_label'][paste_cell_index] = retrieve_result['gt_label'][crop_cell_index]
   


        # results['img'] = mixup_img.astype(np.uint8)
        # results['img_shape'] = mixup_img.shape[:2]
   
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_iters={self.max_iters}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop}, '
        repr_str += f'prob={self.prob})'
        return repr_str



