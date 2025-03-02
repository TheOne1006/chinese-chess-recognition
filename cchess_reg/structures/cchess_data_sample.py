from mmpretrain.structures import DataSample
from mmpretrain.structures.utils import format_label, format_score, LABEL_TYPE, SCORE_TYPE
from .utils import format_label_2d, format_score_2d
import torch


class CChessDataSample(DataSample):

    def set_gt_label(self, value: LABEL_TYPE) -> 'CChessDataSample':
        """Set ``gt_label``."""
        self.set_field(format_label(value), 'gt_label', dtype=torch.Tensor)
        return self

    def set_gt_score(self, value: SCORE_TYPE) -> 'CChessDataSample':
        """Set ``gt_score``."""
        score = format_score_2d(value)
        self.set_field(score, 'gt_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> 'CChessDataSample':
        """Set ``pred_label``."""
        self.set_field(format_label(value), 'pred_label', dtype=torch.Tensor)
        return self

    def set_pred_score(self, value: SCORE_TYPE) -> 'CChessDataSample':
        """Set ``pred_label``."""
        score = format_score_2d(value)
        self.set_field(score, 'pred_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self
