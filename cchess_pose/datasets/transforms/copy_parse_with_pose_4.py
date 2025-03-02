import cv2
import numpy as np
from mmpose.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import copy

# from mmpose.datasets.transforms.base_transform import BaseTransform

@TRANSFORMS.register_module()
class CopyParseWithPose4(BaseTransform):
    """CopyParse data augmentation.

    CopyParse Transform Steps:

        1. 从缓存中随机选择一张图片，将缓存的图片，根据 四个角点，进行仿射变换，映射到 当前图片上
        2. 图片放入缓存
        3. 超出限制移除缓存

    .. code:: text

                     copyParse transform
                +------------------------------+
                |                              |
                |      +-----------------+     |
                |      |                 |     |
                |      |                 |     |
                |      |                 |     |
                |      |   other layout  |     |
                |      |                 |     |
                |      |                 |     |
                |      +-----------------+     |
                |         background           |
                +------------------------------+

    Required Keys:

    - img
    - keypoints
    - keypoints_visible

    Modified Keys:

    - img
        
    """
    def __init__(self,
                 cache_size=100,
                 prob=0.5):
        super().__init__()
        self.cache_size = cache_size
        self.prob = prob
        self.cache = []


    def _get_cache_item(self):
        """
            随机选择缓存中的一张图片
        """
        cache_idx = np.random.randint(0, len(self.cache))
        return self.cache[cache_idx]


    def _after_transform(self, results):
        """
            超出限制移除缓存, 
            1. 随机移除
            2. 当前图片, 
                keypoints !== 0, keypoints_visible!==0， 则加入缓存
        """
        if len(self.cache) > self.cache_size:
            self.cache.pop(np.random.randint(0, len(self.cache)))

        
        # 加入缓存
        cache_item = {
            'img': results['img'],
            'keypoints': results['keypoints'],
            'keypoints_visible': results['keypoints_visible']
        }

        self.cache.append(cache_item)

    def _is_allowed(self, results):
        """
            判断当前图片是否允许加入缓存
        """
        if (results['keypoints'] == 0).any() or (results['keypoints_visible'] == 0).any():
            return False
        return True
    
    # 计算每个点相对于中心点的角度
    @staticmethod
    def compute_angle(point, center):
        return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        
    def transform(self, results):
        """
            Apply transform to results.

            1. 判断 cache 是否为空
            2. 当前图片 的 visibility 是否为 0
        
        
        """
        if np.random.rand() > self.prob:
            return results

        if not self._is_allowed(results):
            return results
            
        img = results['img']
        keypoints = results.get('keypoints', None)
        # keypoints_visible = results.get('keypoints_visible', None)

        origin_result = copy.deepcopy(results)
            
        # 如果缓存为空，直接返回
        if not self.cache:
            self._after_transform(origin_result)
            return results
        
            
        h, w = img.shape[:2]
        
        # 随机选择缓存中的一张图片
        cache_item = self._get_cache_item()
        cache_img = cache_item['img']
        cache_keypoints = cache_item['keypoints']
        # cache_keypoints_visible = cache_item['keypoints_visible']

        # 提取 cahce_img 中的 keypoints 四个角的图片 透视变化到
        # 当前图片中，即 img 中的四个 keypoints 位置

        # 确保 src_points 和 dst_points 是正确的格式：4个点，每个点是(x,y)坐标
        # 检查形状是否为 (1, 4, 2)
        if cache_keypoints.shape != (1, 4, 2) or keypoints.shape != (1, 4, 2):
            raise ValueError("keypoints 的形状必须是 (1, 4, 2)")
        
        # 从形状 (1, 4, 2) 提取正确的点用于透视变换
        cache_points = cache_keypoints[0].astype(np.float32)
        dst_points = keypoints[0].astype(np.float32)
        
        # 增加一些 padding, 约为 棋谱的 4%
        # 1. 分别计算中心点
        src_center = np.mean(cache_points, axis=0)
        dst_center = np.mean(dst_points, axis=0)
        
        # 2. 根据各自中心点，每个点向外延伸 4%
        for i in range(len(cache_points)):
            # 计算从中心点到当前点的向量
            vector = cache_points[i] - src_center
            # 将向量延长8%
            cache_points[i] = src_center + vector * 1.06
        
        for i in range(len(dst_points)):
            # 计算从中心点到当前点的向量
            vector = dst_points[i] - dst_center
            # 将向量延长8%
            dst_points[i] = dst_center + vector * 1.06
            
        
        # # 如果越界，则不进行透视变换
        # if (cache_points < 0).any() or (dst_points < 0).any():
        #     return results
        # # 宽高需根据自身 图片进行判断
        # cache_h, cache_w = cache_img.shape[:2]
        # if (cache_points[:, 1] > cache_h).any() or (dst_points[:, 0] > cache_w).any():
        #     return results
        
        # dist_h, dist_w = dst_points.shape[:2]
        # if (dst_points[:, 1] > dist_h).any() or (dst_points[:, 0] > dist_w).any():
        #     return results
        
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(cache_points, dst_points)

        # 进行透视变换
        warped_img = cv2.warpPerspective(cache_img, M, (w, h))

        # 创建mask用于图像融合 - 确保只在四个点之间生成掩码
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 正确处理点数组，确保它是一个形状为 (4, 2) 的数组
        polygon_points = dst_points.reshape(-1, 2).astype(np.int32)
        
        # polygon_points 应该重新排序，确保是顺时针方向
        
         # 首先计算多边形的中心点
        center = np.mean(polygon_points, axis=0)
        
    
        # 根据角度排序点（顺时针方向）
        sorted_indices = sorted(range(len(polygon_points)), 
                               key=lambda i: -CopyParseWithPose4.compute_angle(polygon_points[i], center))
        polygon_points = polygon_points[sorted_indices]
        
        # 填充多边形区域 - 使用一个连续的多边形
        cv2.fillPoly(mask, [polygon_points], 255)

        # 将mask转换为与图像相同的通道数
        if len(img.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        else:
            mask = mask / 255.0
        
        # 图像融合 - 确保结果是有效的图像
        blended_img = img * (1 - mask) + warped_img * mask
        
        # 确保图像值在有效范围内
        blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
        
        results['img'] = blended_img

        # 更新缓存
        self._after_transform(origin_result)
        
        return results
