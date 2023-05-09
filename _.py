from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

import rcnn_utils
import image_utils
from object_detection_dataset import ObjectDetectionDataset
import _utils


def calculate_gt_offsets(
    positive_anchors: torch.Tensor,
    gt_bboxes: torch.Tensor
) -> torch.Tensor:
    """
    Calculate offsets between selected anchors and corresponding gt bboxes.

    Offsets are:
    1) dxc = (gt_cx - anc_cx) / anc_w
    2) dyc = (gt_cy - anc_cy) / anc_h
    3) dw = log(gt_w / anc_w)
    4) dh = log(gt_h / anc_h)

    Args:
        positive_anchors (torch.Tensor): The positive anchors with shape
        `[n_anc, 4]` in xyxy system.
        gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape
        `[n_anc, 4]` in xyxy system.

    Returns:
        torch.Tensor: The offsets with shape `[]`
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
    anc_boxes_all: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    pos_thresh: float = 0.7,
    neg_thresh: float = 0.2
):
    """
    Подаётся сетка из якорных рамок (n_bboxes, 4), рамки истинных меток
    (b, n_max_objects, 4) и метки классов (b, n_max_objects)
    Также пороги прохождения.
    
    1.
    Нужно достать индексы положительных якорных рамок, которые
    1) Являются самыми близкими по iou к истинным рамкам
    2) Чьё iou больше порога
    
    Так как количество подходящих якорей к разным картинкам будет разное,
    придётся вытянуть их.
    b, n_pos_anc (разное), 4 -> b * n_pos_anc, 4 -> n_pos_anc_all, 4
    В итоге индексы этого вытянутого масива и будут индексами
    подходящих положительных.
    Однако так как батчи сольются вместе, придётся вместе с этим запомнить
    индексы исходных батчей для положительных.

    2.
    К этим положительным якорям нужно вытащить iou

    3.
    И классы gt, к которым они тянутся

    4.
    Для всех положительных якорей надо посчитать offsets к ближайшим gt

    5.
    Сами положительные якори тоже вернуть

    6.
    Посчитать индексы негативных якорей

    7.
    Сами негативные якори тоже вернуть

    

    Args:
        anc_boxes_all (torch.Tensor): _description_
        gt_boxes (torch.Tensor): _description_
        gt_classes (torch.Tensor): shape is `[b, max_num_objects]`
        pos_thresh (float, optional): _description_. Defaults to 0.7.
        neg_thresh (float, optional): _description_. Defaults to 0.2.
    """
    # тип должен получать батч картинок (b, num_all_anchors, max_num_objects)
    # саму сетку (num_all_anchors, 4)
    # Истинные боксы (b, max_num_objects, 4)
    # Классы к истинным боксам (b, max_num_objects)
    # Должен возвращать индексы истинных рамок (b, n_pos,) !не так!
    num_anchors = anc_boxes_all.shape[0]
    b_size, n_max_objects = gt_boxes.shape[0:2]

    iou = rcnn_utils.anc_gt_iou(anc_boxes_all, gt_boxes)  # b, all_anc, max_obj
    max_iou_per_gt, indexes = iou.max(dim=1, keepdim=True)  # b, 1, max_obj

    # Get max anchors' iou per each the object.
    positive_anc_mask = torch.logical_and(
        iou == max_iou_per_gt, max_iou_per_gt > 0.0)
    # and other that passed the threshold.
    positive_anc_mask = torch.logical_or(positive_anc_mask, iou > pos_thresh)

    # Get batch indexes of the positive anchors.
    # It is necessary to determine the ownership of the anchors.
    positive_batch_indexes = torch.where(positive_anc_mask)[0]
    # Flat a batch dimension with num_all_anchors
    # (b * num_all_anchors, max_num_objects)
    positive_anc_mask = positive_anc_mask.flatten(end_dim=1)
    # We do not need to know number of object in image that has this positive
    # anchor. We need to know just index in num_all_anchors.
    positive_anc_indexes = torch.where(positive_anc_mask)[0]

    # Now need to determine the nearest gt bbox for every anchor
    max_iou_per_anc, max_iou_per_anc_idx = iou.max(dim=-1)  # b, all_anc
    # Flat tensor so the positive indexes fit
    # b * all_anc, max_obj
    max_iou_per_anc = max_iou_per_anc.flatten(end_dim=1)

    # Get score for each positive anchors.
    anc_conf_scores = max_iou_per_anc[positive_anc_indexes]

    # Get classes of positive anchors
    # expand gt classes to map against every anchor box
    gt_classes_expand = gt_classes[:, None, :].expand(
        b_size, num_anchors, n_max_objects)
    # for every anchor box, consider only the class of the gt bbox
    # it overlaps with the most.
    anc_classes = torch.gather(
        gt_classes_expand, -1, max_iou_per_anc_idx[..., None]).squeeze(-1)
    # Flat tensor so the positive indexes fit
    anc_classes = anc_classes.flatten(start_dim=0, end_dim=1)
    gt_class_pos = anc_classes[positive_anc_indexes]

    # Expand and flat anchors to iterate with gotten positive indexes
    anc_boxes_all_expand = (anc_boxes_all[None, ...]
                            .expand(b_size, num_anchors, 4)
                            .flatten(end_dim=1))
    positive_ancs = anc_boxes_all_expand[positive_anc_indexes]
    
    # Expand gt boxes to map every anchor
    gt_boxes_expand = gt_boxes[:, None, ...].expand(
        b_size, num_anchors, n_max_objects, 4)  # b, all_anc, max_obj, 4
    # Get the nearest gt boxes from max_samples gt boxes
    nearest_gt_boxes = torch.gather(
        gt_boxes_expand,
        2,
        max_iou_per_anc_idx[..., None, None].repeat(1, 1, 1, 4)
    )  # b, all_anc, 1, 4
    nearest_gt_boxes = nearest_gt_boxes.flatten(end_dim=2)  # b*all_anc, 4
    nearest_gt_boxes = nearest_gt_boxes[positive_anc_indexes]  # pos_anc, 4

    # Get offsets for positive anchors.
    gt_offsets = calculate_gt_offsets(
        positive_ancs, nearest_gt_boxes)  # pos_anc, 4

    negative_mask = max_iou_per_anc < neg_thresh
    negative_anc_indexes = torch.where(negative_mask)[0]
    # Cut negative anchors. Get the same count as positive.
    negative_anc_indexes = negative_anc_indexes[
        torch.randint(0,
                      negative_anc_indexes.shape[0],
                      (positive_anc_indexes.shape[0],))]
    # Get negative nachors
    negative_ancs = anc_boxes_all_expand[negative_anc_indexes]

    return (positive_anc_indexes, negative_anc_indexes, anc_conf_scores,
            gt_offsets, gt_class_pos, positive_ancs, negative_ancs,
            positive_batch_indexes)


def main():
    path = Path(__file__).parent

    annotation_path = path.joinpath('data/annotations.xml')
    img_dir = path.joinpath('data/images')
    name2index = {'pad': -1, 'camel': 0, 'bird': 1}
    index2name = {-1: 'pad', 0: 'camel', 1: 'bird'}
    img_width = 640
    img_height = 480

    dset = ObjectDetectionDataset(
        annotation_path, img_dir, (img_width, img_height), name2index)

    resnet = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*list(resnet.children())[:8])

    dloader = DataLoader(dset, batch_size=2)
    sample = next(iter(dloader))
    image, gt_boxes, classes = sample

    backbone_out = backbone(sample[0])
    b, out_c, out_h, out_w = backbone_out.shape

    width_scale_factor = img_width // out_w
    height_scale_factor = img_height // out_h

    x_anchors, y_anchors = rcnn_utils.generate_anchors((out_h, out_w))
    projected_x_anchors = x_anchors * width_scale_factor
    projected_y_anchors = y_anchors * height_scale_factor

    anc_scales = [2, 4, 6]
    anc_ratios = [0.5, 1, 1.5]
    anc_bboxes = rcnn_utils.generate_anchor_boxes(
        x_anchors, y_anchors, anc_scales, anc_ratios, (out_h, out_w))
    dset_anc_bboxes = anc_bboxes.repeat(len(dset), 1, 1, 1, 1)
    
    projected_gt = rcnn_utils.project_bboxes(
        gt_boxes.reshape(-1, 4), width_scale_factor,
        height_scale_factor, 'p2a').reshape(gt_boxes.shape)

    positive_anc_ind, negative_anc_ind, GT_conf_scores, \
    GT_offsets, GT_class_pos, positive_anc_coords, \
    negative_anc_coords, positive_anc_ind_sep = get_required_anchors(
        anc_bboxes.reshape(-1, 4), projected_gt, classes)
    print()



if __name__ == '__main__':
    main()
    # a = torch.tensor([[1, -1, 3], [-1, 1, -1]])
    # res = torch.where(a > 0)
    # print(res)
