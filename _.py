from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

import rcnn_utils
import image_utils
from object_detection_dataset import ObjectDetectionDataset
import _utils


def get_required_anchors(
    anc_boxes_all: torch.Tensor,
    gt_boxes: torch.Tensor,
    pos_thresh: float = 0.7,
    neg_thresh: float = 0.2
):
    iou = rcnn_utils.anc_gt_iou(anc_boxes_all, gt_boxes)
    max_iou, indexes = iou.max(dim=1)

    indexes[max_iou == 0.0] = -1


    # нужно из anc_boxes_all с размерами (n_boxes, 4) взять только те, которые
    # в indexes с размером (b, max_gt)
    # То есть max_gt боксов для каждой из b картинок
    # Проблемы сейчас 2
    # 1) Индексы не подойдут, так как в некоторых из них 0 из-за нулевого iou
    # их надо как-то обойти. Предположительно с помощью маски с размером n_boxes
    # 2) Как-то обойти b_size. Желательно не с помощью цикла. Разные маски для разных картинок
    b_size = gt_boxes.size(0)
    mask = torch.zeros(*anc_boxes_all.shape, dtype=torch.bool)

    # 
    mask[indexes[1]] = True

    # те индексы, которые нужны
    indexes_mask = iou[1, indexes[1], torch.arange(6)] > pos_thresh

    torch.tensor([torch.arange(b_size), []])

    for i in range(b_size):
        mask[i]




path = Path('/home/pc0/projects/RCNN_proj/rcnn_proj')

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

projected_bboxes = rcnn_utils.project_bboxes(
    anc_bboxes.reshape(-1, 4), width_scale_factor, height_scale_factor)

get_required_anchors(projected_bboxes, gt_boxes)



