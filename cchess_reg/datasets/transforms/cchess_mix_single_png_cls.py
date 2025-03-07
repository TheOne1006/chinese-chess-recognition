from mmcv.transforms import BaseTransform
from dataclasses import dataclass
from mmpretrain.registry import TRANSFORMS
from typing import Tuple, List
import numpy as np
import random
import copy
from PIL import Image
import mmcv
import cv2
import os
from mmcv.transforms.utils import cache_randomness

@dataclass
class CacheItem:
    img_rgb: np.ndarray
    mask: np.ndarray
    label: int


@TRANSFORMS.register_module()
class CChessMixSinglePngCls(BaseTransform):

    cache_items: List[CacheItem] = []
    
    """CChess mixup single png cls data augmentation.

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
            Defaults to (-15, 15).
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    dict_cate_names = {
        'point': '.',
        'other': 'x',
        'red_king': 'K',
        'red_advisor': 'A',
        'red_bishop': 'B',
        'red_knight': 'N',
        'red_rook': 'R',
        'red_cannon': 'C',
        'red_pawn': 'P',
        'black_king': 'k',
        'black_advisor': 'a',
        'black_bishop': 'b',
        'black_knight': 'n',
        'black_rook': 'r',
        'black_cannon': 'c',
        'black_pawn': 'p',
    }


    def __init__(self,
                 img_scale: Tuple[int, int] = (400, 450),
                 cell_scale: Tuple[float, float] = (1.0, 1.5),
                 rotate_angle: Tuple[int, int] = (-180, 180),
                 png_resources_path: str = "",
                 max_mix_cells: int = 10,
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple)
        # assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
        #                                f'but got {max_cached_images}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        
        self.png_resources_path = png_resources_path
        self.img_scale = img_scale
        self.rotate_angle = rotate_angle
        self.max_mix_cells = max_mix_cells
        self.cell_scale = cell_scale

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

        self.init_png_resources()

    def init_png_resources(self):

        cates = self.dict_cate_names.keys()
        
        
        cache_dict_num = {}
        

        for index, cate in enumerate(cates):
            if cate in ['other']:
                continue

            cate_png_dir = os.path.join(self.png_resources_path, cate)
            
            # 如果 目录不存在, 则 跳过
            if not os.path.exists(cate_png_dir):
                continue
            
            cate_png_files = os.listdir(cate_png_dir)

            # filter png
            cate_png_files = [file for file in cate_png_files if file.endswith('.png')]
            
            
            cache_dict_num[cate] = len(cate_png_files)

            for file in cate_png_files:
                file_path = os.path.join(cate_png_dir, file)
                # 读取 rgba 图像
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) 

                try:
                    # bgra 转换为 rgba
                    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                except:
                    raise ValueError(f'The image {file_path} not rgba.')

                # 判断必须 是 rgba
                assert img_rgba.shape[2] == 4, f'The image {file_path} not rgba.'

                # 读取 rgba 中的 alpha 通道, 未做 mask
                alpha_channel = img_rgba[:, :, 3]
                # 将 alpha 通道转换为 0 和 1
                # mask = (alpha_channel > 122).astype(bool)
                mask = alpha_channel
                img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGB)

                cache_item = CacheItem(
                    img_rgb=img_rgb,
                    mask=mask,
                    label=index
                )

                self.cache_items.append(cache_item)
                

        for label, num in cache_dict_num.items():
            print(f'The label {label} has {num} images.')
        
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
    
    # def crop_cell_img(self, cell_index: int, img: np.ndarray) -> np.ndarray:
    #     """Crop the cell image from the original image.
    #     """
    #     x, y = self.get_cell_xywh(cell_index)
    #     w = self.item_cell_width
    #     h = self.item_cell_height

    #     # to int
    #     x = int(x)
    #     y = int(y)
    #     w = int(w)
    #     h = int(h)

    #     # hwc
    #     crop_img = img[y:y+h, x:x+w]
    #     # 旋转
    #     if self.rotate_angle[0] != 0 or self.rotate_angle[1] != 0:

    #         # 随机选择旋转角度
    #         random_angle = random.uniform(
    #             self.rotate_angle[0], 
    #             self.rotate_angle[1]
    #         )
    #         # 旋转图像
    #         crop_img = mmcv.imrotate(crop_img, random_angle)

    #     return crop_img
    
    def paste_cell_img(self, cell_index: int, img: np.ndarray, cache_item: CacheItem) -> np.ndarray:
        """Paste the cell image to the original image.
        """
        x, y = self.get_cell_xywh(cell_index)

        w = self.item_cell_width
        h = self.item_cell_height

        # 缩放
        cell_scale = random.uniform(self.cell_scale[0], self.cell_scale[1])
        w = w * cell_scale
        h = h * cell_scale

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # resize
        cell_img = cv2.resize(cache_item.img_rgb, (w, h))
        mask = cv2.resize(cache_item.mask, (w, h))

        # 旋转
        if self.rotate_angle[0] != 0 or self.rotate_angle[1] != 0:

            # 随机选择旋转角度
            random_angle = random.uniform(
                self.rotate_angle[0], 
                self.rotate_angle[1]
            )
            # 旋转图像
            cell_img = mmcv.imrotate(cell_img, random_angle)
            mask = mmcv.imrotate(mask, random_angle)

        # 上下翻转
        if random.uniform(0, 1) > 0.5:
            cell_img = cv2.flip(cell_img, 0)
            mask = cv2.flip(mask, 0)

        # 左右翻转
        if random.uniform(0, 1) > 0.5:
            cell_img = cv2.flip(cell_img, 1)
            mask = cv2.flip(mask, 1)

        # 修改这部分
        origin_img_part = img[y:y+h, x:x+w]
        

        if origin_img_part.shape[0] != h or origin_img_part.shape[1] != w:
            w = origin_img_part.shape[1]
            h = origin_img_part.shape[0]
            # 重新修正 mask 和 cell_img
            mask = cv2.resize(mask, (w, h))
            cell_img = cv2.resize(cell_img, (w, h))

        mask = (mask > 122).astype(np.uint8)
        mask = mask[..., np.newaxis] 

        try:
            img[y:y+h, x:x+w] = cell_img * mask + origin_img_part * (1 - mask)
        except Exception as e:
            raise e

    def transform(self, results: dict) -> dict:
        """CChessCachedMixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        if random.uniform(0, 1) < self.prob:
            return results
        
        gt_label: list[int] = results['gt_label']
        # 随机从 gt_label 提取 元素为 0 的 索引
        gt_label_point_indexes = [index for index, label in enumerate(gt_label) if label == 0]
        
        # 覆盖的 cells 数量
        cover_cells_num = random.randint(max(1, self.max_mix_cells//2), self.max_mix_cells)

        cover_cells_num = min(cover_cells_num, len(gt_label_point_indexes))


        # 随机获取 贴图的 cells 索引
        retrieve_result_indexes = self.get_indexes(self.cache_items, cover_cells_num)

        # 获取 贴图的 cells 图像
        cache_items: List[CacheItem] = [self.cache_items[index] for index in retrieve_result_indexes]

        # 从 gt_label_point_indexes 中随机获取 索引
        gt_label_point_indexes = random.sample(gt_label_point_indexes, cover_cells_num)


        assert len(gt_label_point_indexes) == len(cache_items), 'The number of gt_label_point_indexes and cache_items must be the same.'

    

        # 随机获取 粘贴图的 cells 图像
        for paste_cell_index, cache_item in zip(gt_label_point_indexes, cache_items):
            # 修改 图片
            self.paste_cell_img(paste_cell_index, results['img'], cache_item)
            # 修改 label
            results['gt_label'][paste_cell_index] = cache_item.label
   


        # results['img'] = mixup_img.astype(np.uint8)
        # results['img_shape'] = mixup_img.shape[:2]
   
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'cell_scale={self.cell_scale}, '
        repr_str += f'rotate_angle={self.rotate_angle}, '
        repr_str += f'png_resources_path={self.png_resources_path}, '
        repr_str += f'prob={self.prob})'
        return repr_str



