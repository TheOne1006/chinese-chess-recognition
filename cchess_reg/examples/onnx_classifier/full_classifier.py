import onnxruntime
import numpy as np
import cv2
from typing import Tuple, List, Union
from .base_onnx import BaseONNX


def center_crop(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop the image to the target size.

    Args:
        image (np.ndarray): The input image.
        target_size (Tuple[int, int]): The desired output size (height, width).

    Returns:
        np.ndarray: The cropped image.
    """
    h, w, _ = image.shape
    target_w, target_h = target_size

    center_x = w // 2
    center_y = h // 2

    start_x = int(center_x - target_w // 2)
    start_y = int(center_y - target_h // 2)

    cropped_image = image[start_y:start_y + target_h, start_x:start_x + target_w]

    return cropped_image


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

class FULL_CLASSIFIER_ONNX(BaseONNX):

    label_2_short = dict_cate_names

    classes_labels = list(dict_cate_names.keys())

    def __init__(self, 
                 model_path,
                 # 输入图片大小
                 input_size=(280, 315), # (w, h)
                 # 图片裁剪大小
                 crop_size=(400, 450), # (w, h)
                 # 是否使用cuda
                 enable_cuda: bool = True
                 ):
        super().__init__(model_path, input_size, enable_cuda)

        self.crop_size = crop_size


    def preprocess_image(self, img_bgr: cv2.UMat, is_rgb: bool = True):

        if not is_rgb:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr

        if img_rgb.shape[:2] != self.crop_size:
            # 调整图片大小 执行 center crop
            img_rgb = center_crop(img_rgb, self.crop_size) # dst_size = (w, h)

        # resize 到 input_size
        img_rgb = cv2.resize(img_rgb, self.input_size)

        # normalize mean and std
        img = (img_rgb - np.array([ 123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])

        img = img.astype(np.float32)
        # 转换为浮点型并归一化
        # img = img.astype(np.float32) / 255.0
        
        # 调整维度顺序 (H,W,C) -> (C,H,W)
        img = np.transpose(img, (2, 0, 1))
        
        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)

        return img
    

    def run_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on the image.

        Args:
            image (np.ndarray): The image to run inference on.

        Returns:
            tuple: A tuple containing the detection results and labels.
        """
        # 运行推理
        outputs, = self.session.run(None, {self.input_name: image})
        
        return outputs

    def pred(self, image: List[Union[cv2.UMat, str]], is_rgb: bool = True) -> Tuple[List[List[str]], List[List[str]], List[List[float]], str]:
        """
        Predict the detection results of the image.

        Args:
            image (cv2.UMat, str): The image to predict.

        Returns:
          
        """
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            is_rgb = False
        else:
            img_bgr = image.copy()

        image = self.preprocess_image(img_bgr, is_rgb)

        labels = self.run_inference(image)

        # 校验 labels 的 shape
        assert labels.shape[1:] == (90, 16)

        # shape (90, 16)
        first_batch_labels = labels[0]

        # 获取置信度最高的标签
        # list[int]
        label_indexes = np.argmax(first_batch_labels, axis=-1).tolist()

        # 将标签索引转换为标签
        # list[str]
        label_names = [self.classes_labels[index] for index in label_indexes]

        # list[str]
        label_short = [self.label_2_short[name] for name in label_names]

        # 获取置信度, 根据  first_batch_labels 和 label_indexes
        confidence = first_batch_labels[np.arange(first_batch_labels.shape[0]), label_indexes]

        label_names_10x9 = [label_names[i*9:(i+1)*9] for i in range(10)]
        label_short_10x9 = [label_short[i*9:(i+1)*9] for i in range(10)]
        confidence_10x9 = [confidence[i*9:(i+1)*9] for i in range(10)]


        layout_str = "\n".join(["".join(row) for row in label_short_10x9])

        return label_names_10x9, label_short_10x9, confidence_10x9, layout_str
    
    def draw_pred(self, image: cv2.UMat, label_index: int, label_name: str, label_short: str, confidence: float) -> cv2.UMat:

        # 在图像上绘制预测结果
        cv2.putText(image, f"{label_short} {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return image
    

    def draw_pred_with_result(self, image: cv2.UMat, results: List[Tuple[int, str, str, float]], cells_xyxy: np.ndarray, is_rgb: bool = True) -> cv2.UMat:

        assert len(results) == cells_xyxy.shape[0]

        if not is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for i, (label_index, label_name, label_short, confidence) in enumerate(results):
            # 确保坐标是整数类型
            x1, y1, x2, y2 = map(int, cells_xyxy[i])

            if label_name.startswith('red'):
                color = (180, 105, 255) # 粉红色
            elif label_name.startswith('black'):
                color = (0, 100, 50) # 黑色
            else:
                color = (0, 0, 255) # 蓝色

            
            if confidence < 0.5:
                # yellow 
                color = (255, 255, 0)

            # confidence:.2f 仅保留两位小数 移除

            label_str = f"{label_short} {confidence:.2f}" if confidence < 0.9 else f"{label_short}"

            cv2.putText(image, label_str, (x1 + 8, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        return image


