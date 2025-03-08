from typing import Optional, Union, Sequence, List
import torch
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric

@METRICS.register_module()
class CChessFullAccuracy(BaseMetric):
    """
    计算中国象棋局面识别的全局准确率
    
    衡量模型对整个棋盘局面的识别能力：
    - 一个完整的棋盘有90个可能的位置（9行10列）
    - 默认情况下，只有当所有位置都预测正确时，才算作一个正确的样本
    - 通过errK参数，可以容忍一定数量的错误位置
    """
    
    default_prefix: Optional[str] = 'cchess-full'
    
    def __init__(self,
                errK: Union[int, Sequence[int]] = (1, ),
                **kwargs) -> None:
        """
        初始化评估指标
        
        Args:
            errK: 允许的错误位置数量
                - 当为单个整数时，表示最多允许errK个位置错误仍算正确
                - 当为序列时，会分别计算每个errK对应的准确率
                - 例如 errK = 1 表示一局游戏中最多1个位置错误仍视为正确
                - 例如 errK = (1, 3, 5) 会计算分别允许1、3、5个错误时的准确率
            **kwargs: 基类参数
        """
        super().__init__(**kwargs)

        # 确保errK是序列类型
        if isinstance(errK, int):
            self.errK = (errK, )
        else:
            self.errK = tuple(sorted(errK))  # 排序以确保顺序一致


    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # 假设预测结果和真实标签都是列表或张量
            pred_score = data_sample['pred_score']
            # 计算每个位置的预测概率
            pred_probs = torch.softmax(pred_score, dim=-1)
            # 计算每个位置的预测标签
            pred_labels = torch.argmax(pred_probs, dim=-1)
            gt_labels = data_sample['gt_label']
            
            # pred_labels = preds['pred_labels']
            # gt_labels = preds['gt_labels']

            result = {
                'pred_labels': pred_labels.to('cpu'),
                'gt_labels': gt_labels.to('cpu'),
            }

            self.results.append(result)

    def compute_metrics(self, results: List):
        """计算所有样本的评估指标。
        
        基于收集的预测结果和真实标签，计算以下指标：
        - all_correct_accuracy: 所有位置都正确预测的样本比例
        - err_k_accuracy: 最多有k个错误位置的样本比例（k为初始化时指定的errK值）
        
        Args:
            results (List): 所有批次处理后的结果列表，每项包含'pred_labels'和'gt_labels'
        
        Returns:
            Dict: 计算得到的评估指标字典，包含：
                - all_correct_accuracy: 完全正确的准确率
                - err_k_accuracy: 容忍k个错误的准确率（k为errK中的值）
        """
        pred_labels = [each['pred_labels'] for each in results]
        gt_labels = [each['gt_labels'] for each in results]
        
        # 计算样本总数
        total = len(pred_labels)
        if total == 0:
            return {'all_correct_accuracy': 0.0, **{f'err_{k}_accuracy': 0.0 for k in self.errK}}
        
        # 初始化结果字典
        result_metrics = {
            'all_correct_accuracy': 0,
            **{f'err_{k}_accuracy': 0 for k in self.errK}
        }
        
        # 计算每个样本的错误数量并更新指标
        for pred, gt in zip(pred_labels, gt_labels):
            # 计算错误位置的数量
            err_num = torch.sum(pred != gt).item()
            
            # 更新指标计数
            if err_num == 0:
                # 完全正确的样本
                result_metrics['all_correct_accuracy'] += 1
                # 完全正确的样本也满足所有errK条件
                for k in self.errK:
                    result_metrics[f'err_{k}_accuracy'] += 1
            else:
                # 对于有错误的样本，检查是否满足各个errK条件
                for k in self.errK:
                    if err_num <= k:
                        result_metrics[f'err_{k}_accuracy'] += 1
            
        # 转换为准确率（百分比）
        for key in result_metrics:
            result_metrics[key] = result_metrics[key] / total
        
        return result_metrics