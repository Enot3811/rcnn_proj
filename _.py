from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

import rcnn_utils
import image_utils
from object_detection_dataset import ObjectDetectionDataset
import _utils


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

backbone_out = backbone(sample[0])
b, out_c, out_h, out_w = backbone_out.shape

width_scale_factor = img_width // out_w
height_scale_factor = img_height // out_h

x_anchors, y_anchors = rcnn_utils.generate_anchors((out_h, out_w))
projected_x_anchors = x_anchors * width_scale_factor
projected_y_anchors = y_anchors * height_scale_factor

anc_scales = [2, 4, 6]
anc_ratios = [0.5, 1, 1.5]
anc_base = rcnn_utils.generate_anchor_boxes(
    x_anchors, y_anchors, anc_scales, anc_ratios, (out_h, out_w))
dset_anc_bboxes = anc_base.repeat(len(dset), 1, 1, 1, 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
image_utils.display_image(dset[0][0], axes[0])
image_utils.display_image(dset[1][0], axes[1])
rcnn_utils.show_anchors(projected_x_anchors, y_anchors, axes[0])
rcnn_utils.show_anchors(projected_x_anchors, y_anchors, axes[1])

projected_bboxes = rcnn_utils.project_bboxes(
    dset_anc_bboxes.reshape(-1, 4), width_scale_factor, height_scale_factor)
rcnn_utils.draw_bounding_boxes(axes[0], projected_bboxes, line_width=1)
rcnn_utils.draw_bounding_boxes(axes[1], projected_bboxes, line_width=1)

plt.show()
