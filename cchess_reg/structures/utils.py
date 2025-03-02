from mmpretrain.structures.utils import format_label, format_score, LABEL_TYPE, SCORE_TYPE
import torch
import numpy as np
from mmengine.utils import is_str
from typing import Sequence



def format_label_2d(value: LABEL_TYPE) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 2, \
        f'The dims of value should be 2, but got {value.ndim}.'

    return value


def format_score_2d(value: SCORE_TYPE) -> torch.Tensor:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence): Score values.

    Returns:
        :obj:`torch.Tensor`: The foramtted score tensor.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 2, \
        f'The dims of value should be 2, but got {value.ndim}.'

    return value

