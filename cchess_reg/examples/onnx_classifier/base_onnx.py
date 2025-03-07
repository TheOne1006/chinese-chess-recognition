import onnxruntime
import numpy as np
import torch
import cv2
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, List

class BaseONNX(ABC):
    def __init__(self, model_path: str, input_size: Tuple[int, int], enable_cuda: bool = True):
        """初始化ONNX模型基类

        Args:
            model_path (str): ONNX模型路径
            input_size (tuple): 模型输入尺寸 (width, height)
        """
        allow_cuda = torch.cuda.is_available()
        if enable_cuda and allow_cuda:
            providers = [
                'CUDAExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def load_image(self, image: Union[cv2.UMat, str]) -> cv2.UMat:
        """加载图像

        Args:
            image (Union[cv2.UMat, str]): 图像路径或cv2图像对象

        Returns:
            cv2.UMat: 加载的图像
        """
        if isinstance(image, str):
            return cv2.imread(image)
        return image.copy()

    @abstractmethod
    def preprocess_image(self, img_bgr: cv2.UMat, *args, **kwargs) -> np.ndarray:
        """图像预处理抽象方法

        Args:
            img_bgr (cv2.UMat): BGR格式的输入图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        pass

    @abstractmethod
    def run_inference(self, image: np.ndarray) -> Any:
        """运行推理的抽象方法

        Args:
            image (np.ndarray): 预处理后的输入图像

        Returns:
            Any: 模型输出结果
        """
        pass

    @abstractmethod
    def pred(self, image: Union[cv2.UMat, str], *args, **kwargs) -> Any:
        """预测的抽象方法

        Args:
            image (Union[cv2.UMat, str]): 输入图像或图像路径

        Returns:
            Any: 预测结果
        """
        pass

    @abstractmethod
    def draw_pred(self, img: cv2.UMat, *args, **kwargs) -> cv2.UMat:
        """绘制预测结果的抽象方法

        Args:
            img (cv2.UMat): 要绘制的图像

        Returns:
            cv2.UMat: 绘制结果后的图像
        """
        pass

    
    def check_images_list(self, images: List[Union[cv2.UMat, str, np.ndarray]]):
        """
        检查图像列表是否有效
        """
        for image in images:
            if not isinstance(image, cv2.UMat) and not isinstance(image, str) and not isinstance(image, np.ndarray):
                raise ValueError("The images must be a list of cv2.UMat or str or np.ndarray.")
 