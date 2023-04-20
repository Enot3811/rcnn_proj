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
    gt_classes: torch.Tensor,
    pos_thresh: float = 0.7,
    neg_thresh: float = 0.2
):
    # тип должен получать батч картинок (b, num_all_anchors, max_num_objects)
    # саму сетку (num_all_anchors, 4)
    # Истинные боксы (b, max_num_objects, 4)
    # Классы к истинным боксам (b, max_num_objects)
    # Должен возвращать индексы истинных рамок (b, n_pos,) !не так!

    iou = rcnn_utils.anc_gt_iou(anc_boxes_all, gt_boxes)
    max_iou, indexes = iou.max(dim=1, keepdim=True)

    # Get max iou anchors
    positive_anc_mask = torch.logical_and(iou == max_iou, max_iou > 0.0)
    # and other that passed the threshold
    positive_anc_mask = torch.logical_or(positive_anc_mask, iou > pos_thresh)

    # Get batch indexes to indexing in flattened positive indexes tensor.
    positive_batch_indexes = torch.where(positive_anc_mask)[0]  # (n_pos,)
    # Flat batch dimension with num_all_anchors
    # (b * num_all_anchors, max_num_objects)
    positive_anc_mask = positive_anc_mask.flatten(end_dim=1)
    # We do not need to know number of object in image that has this positive
    # anchor. We need to know just index in num_all_anchors.
    positive_indexes = torch.where(positive_anc_mask)[0]

    max_iou_per_anc, max_iou_per_anc_ind = iou.max(axis=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(end_dim=1)

    gt_boxes.expand()
    print()


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

    projected_bboxes = rcnn_utils.project_bboxes(
        anc_bboxes.reshape(-1, 4), width_scale_factor, height_scale_factor)

    get_required_anchors(projected_bboxes, gt_boxes, classes)


if __name__ == '__main__':
    main()
    # a = torch.tensor([[1, -1, 3], [-1, 1, -1]])
    # res = torch.where(a > 0)
    # print(res)
