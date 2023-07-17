"""A module that contain functions that help to work with RCNN."""


from typing import Iterable, Tuple, Union, Dict, Optional, List

from numpy.typing import ArrayLike
import torch
from torch import FloatTensor, IntTensor
import torchvision
import matplotlib.pyplot as plt
from matplotlib import patches


def draw_bounding_boxes(
    ax: plt.Axes,
    bboxes: FloatTensor,
    labels: Union[IntTensor, List[int]] = None,
    index2name: Dict[int, str] = None,
    line_width: int = 2,
    color: str = 'y',
) -> plt.Axes:
    """
    Show bounding boxes and corresponding labels on a given Axes.

    Parameters
    ----------
    ax : plt.Axes
        Axes with a sample image.
    bboxes : FloatTensor
        A tensor with shape `[N_bboxes, 4]` that contains the bounding boxes.
    labels : Union[IntTensor, List[int]], optional
        Int tensor or list with length `N_bboxes` with labels corresponding
        to the bounding boxes.
    index2name : Dict[int, str], optional
        A converter dict from int labels to names.
    line_width : int, optional
        A width of the bounding boxes' lines, By default is 2.
    color : str, optional
        A color of the bounding boxes' lines. By default is "y".

    Returns
    -------
    plt.Axes
        The given axis with added bounding boxes.
    """
    if labels is None:
        labels = torch.tensor([-1] * len(bboxes))
    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.int16)
    for bbox, label in zip(bboxes, labels):
        label = label.item()
        if index2name is not None:
            label = index2name[label]
        xmin, ymin, xmax, ymax = bbox.numpy()
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=line_width,
            edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        if label != 'pad' and label != -1:
            ax.text(xmin + 5, ymin + 20, label,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    return ax


def generate_anchors(
        map_size: Tuple[int, int]
) -> Tuple[FloatTensor, FloatTensor]:
    """Generate anchor points on a feature map.

    Anchor points located between feature map pixels.

    Parameters
    ----------
    map_size : Tuple[int, int]
        Height and width of a feature map.

    Returns
    -------
    Tuple[FloatTensor, FloatTensor]
        Tensors containing anchor points defined on feature map.
    """
    height, width = map_size

    anc_pts_x = torch.arange(0, width) + 0.5
    anc_pts_y = torch.arange(0, height) + 0.5

    return anc_pts_x, anc_pts_y


def show_anchors(
    x_points: Union[FloatTensor, ArrayLike],
    y_points: Union[FloatTensor, ArrayLike],
    ax: plt.Axes,
    special_point: Optional[Tuple[int, int]] = None
) -> plt.Axes:
    """Show anchor points on given Axes.

    Parameters
    ----------
    x_points : Union[FloatTensor, ArrayLike]
        X coordinates for anchor points.
    y_points : Union[FloatTensor, ArrayLike]
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
    x_anchors: FloatTensor,
    y_anchors: FloatTensor,
    anc_scales: Union[Iterable[float], FloatTensor],
    anc_ratios: Union[Iterable[float], FloatTensor],
    map_size: Tuple[int, int]
) -> FloatTensor:
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
    x_anchors : FloatTensor
        X coordinates of the anchors.
    y_anchors : FloatTensor
        Y coordinates of the anchors.
    anc_scales : Union[Iterable[float], FloatTensor]
        The scales for boxes.
    anc_ratios : Union[Iterable[float], FloatTensor]
        The ratios for boxes.
    map_size : Tuple[int, int]
        A size of the map.

    Returns
    -------
    FloatTensor
        The generated bounding boxes with shape
        `[n_scales * n_ratios, h_n_anc, w_n_anc, 4]`.
    """
    if isinstance(anc_scales, (list, tuple)):
        anc_scales = torch.tensor(anc_scales)
    if isinstance(anc_ratios, (list, tuple)):
        anc_ratios = torch.tensor(anc_ratios)

    scales = anc_scales.repeat(len(anc_ratios), 1).T.reshape(-1)
    ratios = anc_ratios.repeat(len(anc_scales))

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
    bboxes: FloatTensor,
    width_scale_factor: float,
    height_scale_factor: float,
    mode='a2p'
) -> FloatTensor:
    """Project bounding boxes to a defined scaled space.

    Parameters
    ----------
    bboxes : FloatTensor
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
    FloatTensor
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
    anc_boxes_grid: FloatTensor, gt_boxes: FloatTensor
) -> FloatTensor:
    """
    Calculate intersection over union between anchor boxes and batch of
    ground truth boxes.

    Parameters
    ----------
    anc_boxes_grid : FloatTensor
        A grid of the anchor boxes with a shape `[n_boxes, 4]`.
    gt_boxes : FloatTensor
        The ground truth boxes with a shape `[B, m_boxes, 4]`.

    Returns
    -------
    FloatTensor
        IoU tensor with shape `[B, n_boxes, m_boxes]`.

    Raises
    ------
    RuntimeError
        gt_boxes must have shape like `[B, m_boxes, 4]`.
    RuntimeError
        anc_boxes_grid must have shape like [n_boxes, 4].
    """
    if len(gt_boxes.shape) != 3:
        raise RuntimeError(
            'gt_boxes must have shape like [B, m_boxes, 4] but '
            f'it has {gt_boxes.shape}.')
    if len(anc_boxes_grid.shape) != 2:
        raise RuntimeError(
            'anc_boxes_grid must have shape like [n_boxes, 4] but '
            f'it has {gt_boxes.shape}.')
    
    b_size = gt_boxes.size(0)
    gt_boxes = gt_boxes.reshape(-1, 4)

    # shape (n_boxes, b * m_boxes)
    iou = torchvision.ops.box_iou(anc_boxes_grid, gt_boxes)

    # Cut into b pieces along dim1, and concatenate it along new dim that will
    # be equal b_size
    return torch.stack(torch.chunk(iou, b_size, dim=1))


def calculate_gt_offsets(
    positive_anchors: FloatTensor,
    gt_bboxes: FloatTensor
) -> FloatTensor:
    """Calculate offsets between selected anchors and corresponding gt bboxes.

    Offsets are:
    1) dxc = (gt_cx - anc_cx) / anc_w
    2) dyc = (gt_cy - anc_cy) / anc_h
    3) dw = log(gt_w / anc_w)
    4) dh = log(gt_h / anc_h)

    Parameters
    ----------
    positive_anchors : FloatTensor
        The positive anchors in xyxy system with shape `[n_anc, 4]`.
    gt_bboxes : FloatTensor
        Ground truth bounding boxes in xyxy system with shape `[n_anc, 4]`.

    Returns
    -------
    FloatTensor
        The offsets with shape `[n_pos_anc, 4]`.
    """
    positive_anchors = torchvision.ops.box_convert(
        positive_anchors, 'xyxy', 'cxcywh')
    gt_bboxes = torchvision.ops.box_convert(gt_bboxes, 'xyxy', 'cxcywh')

    anc_cx, anc_cy, anc_w, anc_h = (
        positive_anchors[:, 0], positive_anchors[:, 1],
        positive_anchors[:, 2], positive_anchors[:, 3])
    gt_cx, gt_cy, gt_w, gt_h = (
        gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 2], gt_bboxes[:, 3])

    dxc = (gt_cx - anc_cx) / anc_w
    dyc = (gt_cy - anc_cy) / anc_h
    dw = torch.log(gt_w / anc_w)
    dh = torch.log(gt_h / anc_h)
    return torch.stack((dxc, dyc, dw, dh), -1)


def get_required_anchors(
    anc_boxes_all: FloatTensor,
    gt_boxes: FloatTensor,
    gt_classes: FloatTensor,
    pos_thresh: float = 0.7,
    neg_thresh: float = 0.2
) -> Tuple[IntTensor, IntTensor, IntTensor, FloatTensor, FloatTensor,
           FloatTensor, FloatTensor, FloatTensor]:
    """Get required anchors from all available ones.

    Get indices for positive anchor boxes, and the same number of indices for
    negative ones, `pos_anc_idxs` and `neg_anc_idxs` respectively.

    For positive boxes get indices of batch additionally, `pos_b_idxs`.

    With the indices get the corresponding boxes in xyxy system,
    `pos_ancs` and `neg_ancs`.

    For positive anchor boxes get IoU metric `pos_anc_conf_scores`,
    a corresponding class `gt_class_pos`
    and offsets from ground truth bounding boxes `gt_offsets`.

    Parameters
    ----------
    anc_boxes_all : FloatTensor
        All anchor boxes with shape `[b, n_anc_per_img, 4]`.
    gt_boxes : FloatTensor
        Ground truth bounding boxes with shape `[b, n_max_obj, 4]`.
    gt_classes : FloatTensor
        Classes corresponding the given ground truth boxes
        with shape `[b, n_max_obj]`.
    pos_thresh : float, optional
        Confidence threshold for positive anchor boxes. By default is 0.7.
    neg_thresh : float, optional
        Confidence threshold for negative anchor boxes. By default is 0.2.

    Returns
    -------
    Tuple[IntTensor, IntTensor, IntTensor, FloatTensor, FloatTensor,
          FloatTensor, FloatTensor, FloatTensor]
        Tuple consists of described above `pos_anc_idxs`, `neg_anc_idxs`,
        `pos_b_idxs`, `pos_ancs`, `neg_ancs`, `pos_anc_conf_scores`,
        `gt_class_pos` and `gt_offsets`.
    """
    anchors_per_img = anc_boxes_all.shape[1]
    b_size, n_max_objects = gt_boxes.shape[:2]

    # Send only anchor boxes grid (one slice of the anchors batch)
    # iou shape - (b, all_anc, max_obj)
    iou = anc_gt_iou(anc_boxes_all[0], gt_boxes)
    max_iou_per_gt, indexes = iou.max(dim=1, keepdim=True)  # (b, 1, max_obj)

    # Get max anchors' iou per each ground truth.
    positive_anc_mask = torch.logical_and(
        iou == max_iou_per_gt, max_iou_per_gt > 0.0)
    # and other that passed the threshold.
    positive_anc_mask = torch.logical_or(positive_anc_mask, iou > pos_thresh)

    # Get batch indexes of the positive anchors.
    # It is necessary to determine the ownership of the anchors.
    pos_b_idxs = torch.where(positive_anc_mask)[0]
    # Flat a batch dimension with num_all_anchors
    # (b * num_all_anchors, max_num_objects)
    positive_anc_mask = positive_anc_mask.flatten(end_dim=1)
    # We do not need to know number of object in image that has this positive
    # anchor. We need to know just index in num_all_anchors.
    pos_anc_idxs = torch.where(positive_anc_mask)[0]

    # Now need to determine the nearest gt bbox for every anchor
    max_iou_per_anc, max_iou_per_anc_idx = iou.max(dim=-1)  # b, all_anc
    # Flat tensor so the positive indexes fit
    # b * all_anc, max_obj
    max_iou_per_anc = max_iou_per_anc.flatten(end_dim=1)

    # Get score for each positive anchors.
    pos_anc_conf_scores = max_iou_per_anc[pos_anc_idxs]

    # Get classes of positive anchors
    # expand gt classes to map against every anchor box
    gt_classes_expand = gt_classes[:, None, :].expand(
        b_size, anchors_per_img, n_max_objects)
    # for every anchor box, consider only the class of the gt bbox
    # it overlaps with the most.
    anc_classes = torch.gather(
        gt_classes_expand, -1, max_iou_per_anc_idx[..., None]).squeeze(-1)
    # Flat tensor so the positive indexes fit
    anc_classes = anc_classes.flatten(start_dim=0, end_dim=1)
    gt_class_pos = anc_classes[pos_anc_idxs]

    # Expand and flat anchors to iterate with gotten positive indexes
    anc_boxes_all_flat = anc_boxes_all.flatten(end_dim=1)
    pos_ancs = anc_boxes_all_flat[pos_anc_idxs]
    
    # Expand gt boxes to map every anchor
    gt_boxes_expand = gt_boxes[:, None, ...].expand(
        b_size, anchors_per_img, n_max_objects, 4)  # b, all_anc, max_obj, 4
    # Get the nearest gt boxes from max_samples gt boxes
    nearest_gt_boxes = torch.gather(
        gt_boxes_expand,
        2,
        max_iou_per_anc_idx[..., None, None].repeat(1, 1, 1, 4)
    )  # b, all_anc, 1, 4
    nearest_gt_boxes = nearest_gt_boxes.flatten(end_dim=2)  # b*all_anc, 4
    nearest_gt_boxes = nearest_gt_boxes[pos_anc_idxs]  # pos_anc, 4

    # Get offsets for positive anchors.
    gt_offsets = calculate_gt_offsets(
        pos_ancs, nearest_gt_boxes)  # pos_anc, 4

    negative_mask = max_iou_per_anc < neg_thresh
    neg_anc_idxs = torch.where(negative_mask)[0]
    # Cut negative anchors. Get the same count as positive.
    neg_anc_idxs = neg_anc_idxs[
        torch.randint(0,
                      neg_anc_idxs.shape[0],
                      (pos_anc_idxs.shape[0],))]
    # Get negative anchors
    neg_ancs = anc_boxes_all_flat[neg_anc_idxs]

    return (pos_anc_idxs, neg_anc_idxs, pos_b_idxs, pos_ancs, neg_ancs,
            pos_anc_conf_scores, gt_class_pos, gt_offsets)
