from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import Backend, get_backend, get_input_shape, load_config
from mmdeploy.utils.timer import TimeCounter

class CChessClassificationModel(BaseBackendModel):
    """CChess Classification Model for deployment.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        deploy_cfg (str | mmengine.Config): Deployment config file or loaded Config
            object.
        model_cfg (str | mmengine.Config): Model config file or loaded Config
            object.
    """

    def __init__(self, model_cfg, deploy_cfg, device):
        super().__init__(deploy_cfg)
        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg
        self.device = device

    def forward(self, img, *args, **kwargs):
        """Run forward inference.

        Args:
            img (torch.Tensor | np.ndarray): Input image.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        # 确保输入是4D张量 [N,C,H,W]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        elif img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs = self.wrapper({self.input_name: img})
            score = outputs[self.output_names[0]]

            # 使用 argmax 而不是 topk
            pred_label = score.argmax(dim=-1)
            pred_score = torch.softmax(score, dim=-1)

            result = {
                'pred_scores': pred_score.cpu().numpy(),
                'pred_label': pred_label.cpu().numpy(),
            }

            return [result]

    def show_result(self,
                   img: np.ndarray,
                   result: list,
                   win_name: str = '',
                   show: bool = True,
                   **kwargs):
        """Show predictions of classification.
        """
        return self._visualize(img, result, show=show, win_name=win_name)

    def _visualize(self, img: np.ndarray, result: list, show: bool,
                  win_name: str) -> np.ndarray:
        """Visualize predictions.
        """
        # 这里可以自定义可视化逻辑
        return img 