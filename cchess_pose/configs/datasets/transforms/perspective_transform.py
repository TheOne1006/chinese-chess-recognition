import cv2
import numpy as np
from mmpose.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

# from mmpose.datasets.transforms.base_transform import BaseTransform

@TRANSFORMS.register_module()
class RandomPerspectiveTransform(BaseTransform):
    """随机透视变换。

    Args:
        prob (float): 执行概率, 默认0.5
        scale (tuple[float]): 透视变换的强度范围，默认(0.05, 0.1)
    """

    def __init__(self, prob=0.5, scale=(0.05, 0.1)):
        super().__init__()
        self.prob = prob
        self.scale_range = scale

    def transform(self, results: dict) -> dict:
        """执行透视变换。"""
        if np.random.random() > self.prob:
            return results
            
        img = results['img']
        height, width = img.shape[:2]
        
        # 随机生成透视变换的强度
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        # 定义四个角点
        pts1 = np.array([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ], dtype=np.float32)

        # 随机扰动四个角点
        pts2 = np.array([
            [0 + np.random.uniform(-scale * width, scale * width), 0 + np.random.uniform(-scale * width, scale * width)],
            [width + np.random.uniform(-scale * width, scale * width), 0 + np.random.uniform(-scale * width, scale * width)],
            [0 + np.random.uniform(-scale * width, scale * width), height + np.random.uniform(-scale * width, scale * width)],
            [width + np.random.uniform(-scale * width, scale * width), height + np.random.uniform(-scale * width, scale * width)]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 应用透视变换到图像
        img_warped = cv2.warpPerspective(img, M, (width, height))
        results['img'] = img_warped
        
        # 如果有关键点，也需要转换关键点坐标
        if 'keypoints' in results:
            keypoints = results['keypoints']
            if len(keypoints) > 0:
                # 将关键点坐标转换为齐次坐标 (N, 3)
                points = keypoints[0]  # 只取x,y坐标
                ones = np.ones(len(points))[:, None]  # (N, 1)
                points_homogeneous = np.hstack([points, ones])  # (N, 3)
                
                # 应用透视变换 (3x3) @ (3xN) -> (3xN)
                transformed_points = M @ points_homogeneous.T
                
                # 转回笛卡尔坐标 (N, 2)
                transformed_points = transformed_points.T
                transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]
                
                # 更新关键点坐标，保持其他信息不变
                # keypoints_new = keypoints.copy()
                # keypoints_new[:, :2] = transformed_points
                results['keypoints'] = np.array([transformed_points])
                
        return results
