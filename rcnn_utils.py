"""A module that contain functions that help to work with RCNN."""


from typing import Tuple, Union, Dict, Optional, List

import numpy as np
import torch
from torch import Tensor
import torchvision
import matplotlib.pyplot as plt
from matplotlib import patches


def draw_bounding_boxes(
    ax: plt.Axes,
    bboxes: Tensor,
    labels: Optional[Tensor] = None,
    index2name: Optional[Dict[int, str]] = None,
    line_width: Optional[int] = 2
) -> plt.Axes:
    """
    Show bounding boxes and corresponding labels on a given Axes.

    Parameters
    ----------
    ax : plt.Axes
        Axes with a sample image.
    bboxes : Tensor
        A tensor with shape `[N_bboxes, 4]` that contains the bounding boxes.
    labels : Optional[Tensor]
        Labels that correspond the bounding boxes.
    index2name : Optional[Dict[int, str]]
        A converter dict from int labels to names.
    line_width : Optional[int]
        A width of bounding boxes' lines

    Returns
    -------
    plt.Axes
        The given axis with added bounding boxes.
    """
    if labels is None:
        labels = torch.tensor([-1] * len(bboxes))
    for bbox, label in zip(bboxes, labels):
        label = label.item()
        if index2name is not None:
            label = index2name[label]
        xmin, ymin, xmax, ymax = bbox.numpy()
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=line_width,
            edgecolor='y', facecolor='none')
        ax.add_patch(rect)

        if label != 'pad' and label != -1:
            ax.text(xmin + 5, ymin + 20, label,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    return ax


def generate_anchors(
        map_size: Tuple[int, int]
) -> Tuple[Tensor, Tensor]:
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
    x_points: Union[Tensor, np.ndarray],
    y_points: Union[Tensor, np.ndarray],
    ax: plt.Axes,
    special_point: Optional[Tuple[int, int]] = None
) -> plt.Axes:
    """Show anchor points on given Axes.

    Parameters
    ----------
    x_points : Union[Tensor, np.ndarray]
        X coordinates for anchor points.
    y_points : Union[Tensor, np.ndarray]
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
    x_anchors: Tensor,
    y_anchors: Tensor,
    anc_scales: Union[List[float], Tensor],
    anc_ratios: Union[List[float], Tensor],
    map_size: Tuple[int, int]
) -> Tensor:
    """Create anchor boxes.

    Generate tensor with shape `[Hmap, Wmap, n_anchor_boxes, 4]`
    that consists of bounding boxes that are generated from given coordinates
    and specified scales and ratios.

    The number of the anchor boxes is:\n
    `n_anchor_boxes = len(anc_scales) * len(anc_ratios)`

    Boxes are generating according to rules:\n
    `xmin = x_anc - scale * ratio / 2`\n
    `xmax = x_anc + scale * ratio / 2`\n
    `ymin = y_anc - scale / 2`\n
    `ymax = y_anc + scale / 2`

    The generated anchors are clipped to fit the map (image).

    Parameters
    ----------
    x_anchors : Tensor
        X coordinates of the anchors.
    y_anchors : Tensor
        Y coordinates of the anchors.
    anc_scales : Union[List[float], Tensor]
        The scales for boxes.
    anc_ratios : Union[List[float], Tensor]
        The ratios for boxes.
    map_size : Tuple[int, int]
        A size of the map.

    Returns
    -------
    Tensor
        The generated bounding boxes.
    """
    if isinstance(anc_scales, list):
        anc_scales = torch.tensor(anc_scales)
    if isinstance(anc_ratios, list):
        anc_ratios = torch.tensor(anc_ratios)

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
        torch.stack((x_mins[None, ...].repeat(n_y, 1, 1),
                     y_mins[:, None, :].repeat(1, n_x, 1)),
                    dim=3),
        x_maxs[None, ...].repeat(n_y, 1, 1)[..., None],
        y_maxs[:, None, :].repeat(1, n_x, 1)[..., None]),
        dim=3)

    orig_shape = anc_base.shape
    anc_base = torchvision.ops.clip_boxes_to_image(
        anc_base.reshape(-1, 4),
        map_size
    ).reshape(orig_shape)
    return anc_base


def project_bboxes(
    bboxes: Tensor,
    width_scale_factor: float,
    height_scale_factor: float,
    mode='a2p'
) -> Tensor:
    """Project bounding boxes to a defined scaled space.

    Parameters
    ----------
    bboxes : Tensor
        A tensor with shape `[n_bboxes, 4]` that contains bounding boxes.
    width_scale_factor : float
        A scale factor across a width.
    height_scale_factor : float
        A scale factor across a height.
    mode : str, optional
        A mode of a projection. It can be `a2p` or `p2a`
        that correspond to an activation map to a pixel image projection and
        a pixel image to activation map. by default 'a2p'

    Returns
    -------
    Tensor
        The projected bounding boxes with shape `[n_bboxes, 4]`.

    Raises
    ------
    ValueError
        The mode must be either "a2p" or "p2a".
    """
    if mode not in ('a2p', 'p2a'):
        raise ValueError(
            'The mode must be either "a2p" or "p2a", '
            f'but given mode is {mode}.')
    batch_size = bboxes.shape[0]
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    pad_bbox_mask = (proj_bboxes == -1)  # indicating padded bboxes

    if mode == 'a2p':
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    elif mode == 'p2a':
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor

    proj_bboxes[pad_bbox_mask] = -1
    proj_bboxes = proj_bboxes.reshape(bboxes.shape)
    return proj_bboxes


def anc_gt_iou(
    anc_boxes_grid: Tensor, gt_boxes: Tensor
) -> Tensor:
    """
    Calculate intersection over union between anchor boxes and batch of
    ground truth boxes.

    Parameters
    ----------
    anc_boxes_all : Tensor
        A grid of the anchor boxes with a shape `[n_boxes, 4]`.
    gt_boxes : Tensor
        The ground truth boxes with a shape `[B, m_boxes, 4]`.

    Returns
    -------
    Tensor
        IoU tensor with shape `[B, n_boxes, m_boxes]`.

    Raises
    ------
    RuntimeError
        gt_boxes must have shape like `[B, m_boxes, 4]`.
    RuntimeError
        anc_boxes_grid must have shape like [B, m_boxes, 4].
    """
    if len(gt_boxes.shape) != 3:
        raise RuntimeError(
            'gt_boxes must have shape like [B, m_boxes, 4] but '
            f'it has {gt_boxes.shape}.')
    if len(anc_boxes_grid.shape) != 2:
        raise RuntimeError(
            'anc_boxes_grid must have shape like [B, m_boxes, 4] but '
            f'it has {gt_boxes.shape}.')
    
    b_size = gt_boxes.size(0)
    gt_boxes = gt_boxes.reshape(-1, 4)

    # shape (n_boxes, b * m_boxes)
    iou = torchvision.ops.box_iou(anc_boxes_grid, gt_boxes)

    # Cut into b pieces along dim1, and concatenate it along new dim that will
    # be equal b_size
    return torch.stack(torch.chunk(iou, b_size, dim=1))
