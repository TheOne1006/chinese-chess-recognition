import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mmcv
from typing import Optional, Sequence
from mmengine.visualization import Visualizer
from mmengine.registry import VISUALIZERS


from mmpretrain.visualization.utils import get_adaptive_scale
from cchess_reg.structures import CChessDataSample

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
classes_labels = list(dict_cate_names.values())

@VISUALIZERS.register_module()
class CChessVisualizer(Visualizer):
    """自定义中国象棋棋盘可视化器，与inference_demo.py保持一致"""
    DEFAULT_TEXT_CFG = {
        'family': 'monospace',
        'color': 'white',
        'bbox': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
        'verticalalignment': 'top',
        'horizontalalignment': 'left',
    }
    # def __init__(self, name='visualizer', **kwargs):
    #     super().__init__(name=name, **kwargs)
        
    
    def visualize_cls(self,
                      image: np.ndarray,
                      data_sample: CChessDataSample,
                      classes: Optional[Sequence[str]] = None,
                      draw_gt: bool = True,
                      draw_pred: bool = True,
                      draw_score: bool = True,
                      resize: Optional[int] = None,
                      rescale_factor: Optional[float] = None,
                      text_cfg: dict = dict(),
                      show: bool = False,
                      wait_time: float = 0,
                      out_file: Optional[str] = None,
                      name: str = '',
                      step: int = 0) -> None:
        """可视化棋盘识别结果。
        
        Args:
            image (np.ndarray): 输入图像
            data_sample (CChessDataSample): 数据样本，包含预测结果
            classes (Optional[Sequence[str]], optional): 类别名称列表
            draw_gt (bool, optional): 是否绘制真实标签. 默认True
            draw_pred (bool, optional): 是否绘制预测结果. 默认True
            draw_score (bool, optional): 是否显示分数. 默认True
            resize (Optional[int], optional): 调整图像大小. 默认None
            rescale_factor (Optional[float], optional): 缩放因子. 默认None
            text_cfg (dict, optional): 文本配置. 默认dict()
            show (bool, optional): 是否显示图像. 默认False
            wait_time (float, optional): 显示等待时间. 默认0
            out_file (Optional[str], optional): 输出文件路径. 默认None
            name (str, optional): 窗口名称. 默认''
            step (int, optional): 步骤编号. 默认0
        """
        # 尝试多种方式获取类别名称
        if classes is None:
            if self.dataset_meta is not None:
                classes = self.dataset_meta.get('classes', None)
                
            # 如果仍未获取到类别名称，使用中国象棋默认类别
            if classes is None:
                classes = classes_labels
        
        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image = mmcv.imresize(image, (resize, resize * h // w))
            else:
                image = mmcv.imresize(image, (resize * w // h, resize))
        elif rescale_factor is not None:
            image = mmcv.imrescale(image, rescale_factor)

        self.set_image(image)
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 棋盘有10行9列
        rows, cols = 10, 9
        
        # 设置棋盘边距
        offset_x, offset_y = 50, 50
        
        # 计算每个格子的大小
        cell_h, cell_w = (h - offset_y * 2) / (rows - 1), (w - offset_x * 2) / (cols - 1)
        
        # 解析预测结果
        if draw_pred and 'pred_label' in data_sample:
            # 预测标签 pred_labels = []
            pred_labels = data_sample.pred_label.tolist()
            # 预测得分
            pred_scores = data_sample.pred_score.tolist() if 'pred_score' in data_sample else None
            
            # 绘制棋盘格子和标签
            for i in range(rows * cols):
                row, col = i // cols, i % cols
                
                # 计算格子左上角坐标
                x, y = col * cell_w + offset_x - cell_w/2, row * cell_h + offset_y - cell_h/2
                
                # 绘制矩形
                self.draw_bboxes(
                    np.array([[x, y, x + cell_w, y + cell_h]]),
                    edge_colors='r',
                    alpha=0.5,
                    line_widths=1
                )
                
                # 获取类别和置信度
                class_idx = pred_labels[i]
                label = classes[class_idx] if classes else str(class_idx)
                conf = pred_scores[i]
                
                # 确保conf是单个浮点数而不是列表
                if isinstance(conf, list):
                    conf = conf[class_idx] if len(conf) > class_idx else 0.0
                    
                # 根据标签是否为大写选择不同的颜色
                if label.isupper():
                    facecolor = 'red'
                else:
                    facecolor = 'green'
                    
                if label == 'x':
                    facecolor = 'black'
                elif label == '.':
                    facecolor = 'white'
                
                # 在格子中心绘制类别和置信度
                self.draw_texts(
                    f'{label}\n{conf:.2f}' if draw_score else label,
                    np.array([[x + cell_w/2, y + cell_h/2]]),
                    colors='white',
                    font_sizes=int(min(cell_w, cell_h) / 4),
                    bboxes=[{
                        'facecolor': facecolor,
                        'alpha': 0.5,
                        'pad': 2,
                    }],
                    horizontal_alignments='center',
                    vertical_alignments='center'
                )

        # 添加标题
        img_scale = get_adaptive_scale(image.shape[:2])
        text_cfg = {
            'size': int(img_scale * 10),
            **self.DEFAULT_TEXT_CFG,
            **text_cfg,
        }
        self.ax_save.text(
            w/2,
            20,
            "cchess reg",
            **text_cfg,
            ha='center',
            va='top',
        )
        
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # 保存图像到指定文件而不是vis_backends
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img