from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

import rcnn_model

from object_detection_dataset import ObjectDetectionDataset


def main():
    path = Path(__file__).parent

    # Dataset's settings
    annotation_path = path.joinpath('data/annotations.xml')
    img_dir = path.joinpath('data/images')
    name2index = {'pad': -1, 'camel': 0, 'bird': 1}
    index2name = {-1: 'pad', 0: 'camel', 1: 'bird'}
    img_width = 640
    img_height = 480
    n_cls = len(name2index) - 1

    # Thresholds for object detecting
    pos_thresh = 0.7
    neg_thresh = 0.3

    # Anchors parameters
    anc_scales = [2, 4, 6]
    anc_ratios = [0.5, 1, 1.5]

    # Roi size for classifier
    roi_size = (2, 2)

    # Get Dataset and Loader
    dset = ObjectDetectionDataset(
        annotation_path, img_dir, (img_width, img_height), name2index)
    dloader = DataLoader(dset, batch_size=2)

    # Get backbone for RPN
    resnet = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*list(resnet.children())[:8])

    # Use backbone to get it's output size
    batch = next(iter(dloader))
    images, gt_boxes, gt_cls = batch
    backbone_out = backbone(images)

    # Backbone output size
    b, out_c, out_h, out_w = backbone_out.shape

    # Get the model
    model = rcnn_model.RCNN_Detector(
        (img_height, img_width), (out_h, out_w), out_c, n_cls, roi_size,
        anc_scales, anc_ratios, pos_thresh, neg_thresh)
    
    model.load_state_dict(torch.load('model_with_permute.pt'))
    
    device = torch.device('cpu')
    model = model.to(device=device)

    # Iterate over dataset
    for batch in dloader:
        images, gt_boxes, gt_cls = batch
        images = images.to(device=device)
        gt_boxes = gt_boxes.to(device=device)
        gt_cls = gt_cls.to(device=device)
        proposals, cls_scores, total_loss = model(images, gt_boxes, gt_cls)
        break

    model.eval()
    for batch in dloader:
        images, gt_boxes, gt_cls = batch
        proposals, cls_scores = model(
            images, conf_thresh=0.80, nms_thresh=0.05)
        break


if __name__ == '__main__':
    main()
