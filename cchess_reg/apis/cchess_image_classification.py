from mmpretrain.apis.image_classification import ImageClassificationInferencer
# import torch
import numpy as np
from typing import List, Optional, Union, Sequence
from mmpretrain.structures import DataSample
# from mmpretrain.utils import get_file_backend
# from mmengine.fileio import get_file_backend, list_dir_or_file
# import os.path as osp
# from mmpretrain.apis.base import BaseInferencer, InputType, ModelType
# from cchess_reg.structures import CChessDataSample
from pathlib import Path
# import torch
# import cv2
# from mmengine.fileio import imread


class CChessImageClassificationInferencer(ImageClassificationInferencer):
    """自定义的中国象棋棋盘识别推理类"""
    
    # def __init__(self, **kwargs):
    #     # 设置strict=False以允许缺少某些键
    #     kwargs['show'] = kwargs.get('show', False)
    #     kwargs['device'] = kwargs.get('device', 'cuda:0')
    #     self._strict_load = False
    #     super().__init__(**kwargs)
    
    def postprocess(self,
                    preds: List[DataSample],
                    visualization: List[np.ndarray],
                    return_datasamples=False) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            # torch.Size([90, 16])
            pred_scores = data_sample.pred_score

            pred_label = pred_scores.argmax(dim=-1, keepdim=True).detach()
            pred_score = pred_scores.gather(dim=-1, index=pred_label)
            # pred_score = float(torch.max(pred_scores).item())
            # pred_label = torch.argmax(pred_scores).item()
            result = {
                'pred_scores': pred_scores.detach().cpu().numpy(),
                'pred_label': pred_label.squeeze(-1).detach().cpu().numpy(),
                'pred_score': pred_score.squeeze(-1).detach().cpu().numpy(),
            }

            if self.classes is not None:
                result['pred_class'] = [
                    self.classes[x] for x in result['pred_label']
                ]
            results.append(result)

        return results

    # def visualize(self,
    #               ori_inputs: List[InputType],
    #               preds: List[CChessDataSample],
    #               show: bool = False,
    #               wait_time: int = 0,
    #               resize: Optional[int] = None,
    #               rescale_factor: Optional[float] = None,
    #               draw_score=True,
    #               show_dir=None):
    #     """可视化推理结果。

    #     Args:
    #         images: 输入图像，可以是路径或者图像数组
    #         preds: 模型预测结果
    #         show: 是否显示可视化结果
    #         wait_time: 显示等待时间
    #         resize: 调整图像大小
    #         rescale_factor: 图像缩放因子
    #         draw_score: 是否绘制得分
    #         show_dir: 保存可视化结果的目录

    #     Returns:
    #         List[np.ndarray]: 可视化结果图像列表
    #     """
    #     if not show and show_dir is None:
    #         return None

    #     if self.visualizer is None:
    #         from mmpretrain.visualization import UniversalVisualizer
    #         self.visualizer = UniversalVisualizer()

    #     # 处理不同类型的输入
    #     if isinstance(images, (str, np.ndarray)):
    #         images = [images]
        
    #     visualization = []
    #     for i, (image, data_sample) in enumerate(zip(images, preds)):
    #         if isinstance(image, str):
    #             # 从路径加载图像
    #             image_array = imread(image)
    #             # BGR图像需要转换为RGB
    #             image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    #             name = Path(image).stem
    #         else:
    #             # 直接使用图像数组
    #             image_array = image
    #             name = str(i)

    #         # 设置输出文件路径
    #         if show_dir is not None:
    #             show_dir = Path(show_dir)
    #             show_dir.mkdir(exist_ok=True)
    #             out_file = str((show_dir / name).with_suffix('.png'))
    #         else:
    #             out_file = None

    #         # 使用可视化器绘制结果
    #         self.visualizer.visualize_cls(
    #             image_array,
    #             data_sample,
    #             classes=self.classes,
    #             resize=resize,
    #             show=show,
    #             wait_time=wait_time,
    #             rescale_factor=rescale_factor,
    #             draw_gt=False,
    #             draw_pred=True,
    #             draw_score=draw_score,
    #             name=name,
    #             out_file=out_file)
            
    #         visualization.append(self.visualizer.get_image())
        
    #     if show:
    #         self.visualizer.close()
        
    #     return visualization
