"""
A module that contain functions that help to work with RCNN.
"""


from typing import Tuple, Union, Dict, Optional, List

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import patches


def draw_bounding_boxes(
    ax: plt.Axes,
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    index2name: Dict[int, str] = None
) -> plt.Axes:
    """
    Show bounding boxes and corresponding labels on given Axes.

    Parameters
    ----------
    ax : plt.Axes
        Axes with sample image.
    bboxes : torch.Tensor
        Bounding boxes.
    labels : torch.Tensor
        Labels that correspond bounding boxes.
    index2name : Dict[int, str], optional
        Converter from int labels to names, by default None

    Returns
    -------
    plt.Axes
        Given axis with added bounding boxes.
    """
    for bbox, label in zip(bboxes, labels):
        label = label.item()
        if index2name is not None:
            label = index2name[label]
        if label == 'pad':
            continue

        xmin, ymin, xmax, ymax = bbox.numpy()
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='y',
            facecolor='none')
        ax.add_patch(rect)

        ax.text(xmin + 5, ymin + 20, label,
                bbox=dict(facecolor='yellow', alpha=0.5))
    return ax


def generate_anchors(
        map_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor points on a feature map.

    Anchor points located between feature map pixels.

    Parameters
    ----------
    map_size : Tuple[int, int]
        Height and width of a feature map.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Tensors containing anchor points defined on feature map.
    """
    height, width = map_size

    anc_pts_x = torch.arange(0, width) + 0.5
    anc_pts_y = torch.arange(0, height) + 0.5

    return anc_pts_x, anc_pts_y


def show_anchors(
    x_points: Union[torch.Tensor, np.ndarray],
    y_points: Union[torch.Tensor, np.ndarray],
    ax: plt.Axes,
    special_point: Optional[Tuple[int, int]] = None
) -> plt.Axes:
    """Show anchor points on given Axes.

    Parameters
    ----------
    x_points : Union[torch.Tensor, np.ndarray]
        X coordinates for anchor points.
    y_points : Union[torch.Tensor, np.ndarray]
        Y coordinates for anchor points.
    ax : plt.Axes
        Axes for showing.
    special_point : Optional[Tuple[int, int]]
        A point to highlight.

    Returns
    -------
    plt.Axes
        Given Axes with showed anchors.
    """
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, marker='+', color='w')
    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
    return ax


def generate_anchor_boxes(
    x_anchors: torch.Tensor,
    y_anchors: torch.Tensor,
    anc_scales: Union[List[float], torch.Tensor],
    anc_ratios: Union[List[float], torch.Tensor],
    map_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Generate tensor with shape `[Hmap, Wmap, n_anchor_boxes, 4]`
    that consists of bounding boxes that are generated from given anchors
    and specified scales and ratios.

    The number of the anchor boxes is:\n
    `n_anchor_boxes = len(anc_scales) * len(anc_ratios)`

    Boxes are generating according to rules:\n
    `xmin = x_anc - scale * ratio / 2`\n
    `xmax = x_anc + scale * ratio / 2`\n
    `ymin = y_anc - scale / 2`\n
    `ymax = y_anc + scale / 2`

    Args:
        x_anchors (torch.Tensor): X coordinates of the anchors.
        y_anchors (torch.Tensor): Y coordinates of the anchors.
        anc_scales (Union[List[float], torch.Tensor]): The scales for boxes.
        anc_ratios (Union[List[float], torch.Tensor]): The ratios for boxes.
        map_size (Tuple[int, int]): A size of the map.

    Returns:
        torch.Tensor: The generated bounding boxes.
    """
    if isinstance(anc_scales, list):
        anc_scales = torch.Tensor(anc_scales)
    if isinstance(anc_ratios, list):
        anc_ratios = torch.Tensor(anc_ratios)

    scales = anc_scales.repeat(len(anc_ratios), 1).T.reshape(-1)
    ratios = anc_ratios.repeat(len(anc_scales))

    assert len(scales) == len(ratios)

    x_biases = scales * ratios / 2
    y_biases = scales / 2

    x_anchors_ext = x_anchors.repeat(len(x_biases), 1).T
    y_anchors_ext = y_anchors.repeat(len(y_biases), 1).T

    x_biases = x_biases.repeat(len(x_anchors), 1)
    y_biases = y_biases.repeat(len(y_anchors), 1)

    x_mins = x_anchors_ext - x_biases
    x_maxs = x_anchors_ext + x_biases
    y_mins = y_anchors_ext - y_biases
    y_maxs = y_anchors_ext + y_biases

    n_y = y_mins.shape[0]
    n_x = x_mins.shape[0]
    anc_base = torch.cat((
        torch.stack((x_mins[:, None, :].repeat(1, n_y, 1),
                    y_mins[None, :].repeat(n_x, 1, 1)),
                    dim=3),
        x_maxs[:, None, :].repeat(1, n_y, 1)[..., None],
        y_maxs[None, :].repeat(n_x, 1, 1)[..., None]),
        dim=3)

    orig_shape = anc_base.shape
    anc_base = torchvision.ops.clip_boxes_to_image(
        anc_base.reshape(-1, 4),
        map_size
    ).reshape(orig_shape)
    return anc_base
