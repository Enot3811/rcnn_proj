from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
import torchvision
from torch.utils.data import DataLoader

import rcnn_utils

from object_detection_dataset import ObjectDetectionDataset


def calculate_gt_offsets(
    positive_anchors: Tensor,
    gt_bboxes: Tensor
) -> Tensor:
    """
    Calculate offsets between selected anchors and corresponding gt bboxes.

    Offsets are:
    1) dxc = (gt_cx - anc_cx) / anc_w
    2) dyc = (gt_cy - anc_cy) / anc_h
    3) dw = log(gt_w / anc_w)
    4) dh = log(gt_h / anc_h)

    Args:
        positive_anchors (Tensor): The positive anchors with shape
        `[n_anc, 4]` in xyxy system.
        gt_bboxes (Tensor): Ground truth bounding boxes with shape
        `[n_anc, 4]` in xyxy system.

    Returns:
        Tensor: The offsets with shape `[]`
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
    anc_boxes_all: Tensor,
    gt_boxes: Tensor,
    gt_classes: Tensor,
    pos_thresh: float = 0.7,
    neg_thresh: float = 0.2
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    anc_boxes_all : Tensor
        All anchor boxes with shape `(b, n_anc_per_img, 4)`.
    gt_boxes : Tensor
        Ground truth bounding boxes with shape `(b, n_max_obj, 4)`.
    gt_classes : Tensor
        Classes corresponding the given ground truth boxes
        with shape `(b, n_max_obj)`.
    pos_thresh : float, optional
        Confidence threshold for positive anchor boxes. By default is 0.7.
    neg_thresh : float, optional
        Confidence threshold for negative anchor boxes. By default is 0.2.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        Tuple consists of described above `pos_anc_idxs`, `neg_anc_idxs`,
        `pos_b_idxs`, `pos_ancs`, `neg_ancs`, `pos_anc_conf_scores`,
        `gt_class_pos` and `gt_offsets`.
    """
    anchors_per_img = anc_boxes_all.shape[1]
    b_size, n_max_objects = gt_boxes.shape[:2]

    # Send only anchor boxes grid (one slice of the anchors batch)
    # iou shape - (b, all_anc, max_obj)
    iou = rcnn_utils.anc_gt_iou(anc_boxes_all[0], gt_boxes)
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
    anc_boxes_all_expand = (anc_boxes_all[None, ...]
                            .expand(b_size, anchors_per_img, 4)
                            .flatten(end_dim=1))
    pos_ancs = anc_boxes_all_expand[pos_anc_idxs]
    
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
    neg_ancs = anc_boxes_all_expand[neg_anc_idxs]

    return (pos_anc_idxs, neg_anc_idxs, pos_b_idxs, pos_ancs, neg_ancs,
            pos_anc_conf_scores, gt_class_pos, gt_offsets)


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
    anc_bboxes_grid = rcnn_utils.generate_anchor_boxes(
        x_anchors, y_anchors, anc_scales, anc_ratios, (out_h, out_w))
    all_anc_bboxes = anc_bboxes_grid.repeat(len(dset), 1, 1, 1, 1)
    
    projected_gt = rcnn_utils.project_bboxes(
        gt_boxes.reshape(-1, 4), width_scale_factor,
        height_scale_factor, 'p2a').reshape(gt_boxes.shape)

    positive_anc_ind, negative_anc_ind, GT_conf_scores, \
    GT_offsets, GT_class_pos, positive_anc_coords, \
    negative_anc_coords, positive_anc_ind_sep = get_required_anchors(
        all_anc_bboxes.reshape(b, -1, 4), projected_gt, classes)
    print()


if __name__ == '__main__':
    main()
    # source_pipeline()
    # a = torch.tensor([[1, -1, 3], [-1, 1, -1]])
    # res = torch.where(a > 0)
    # print(res)


def source_pipeline():
    from skimage import io
    from skimage.transform import resize
    from _model import TwoStageDetector
    import os

    import torch
    import torchvision
    from torchvision import ops
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.nn.utils.rnn import pad_sequence

    from _utils import parse_annotation, gen_anc_base, project_bboxes, get_req_anchors, gen_anc_centers

    class ObjectDetectionDataset(Dataset):
        '''
        A Pytorch Dataset class to load the images and their corresponding annotations.
        
        Returns
        ------------
        images: Tensor of size (B, C, H, W)
        gt bboxes: Tensor of size (B, max_objects, 4)
        gt classes: Tensor of size (B, max_objects)
        '''
        def __init__(self, annotation_path, img_dir, img_size, name2idx):
            self.annotation_path = annotation_path
            self.img_dir = img_dir
            self.img_size = img_size
            self.name2idx = name2idx
            
            self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()
            
        def __len__(self):
            return self.img_data_all.size(dim=0)
        
        def __getitem__(self, idx):
            return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]
            
        def get_data(self):
            img_data_all = []
            gt_idxs_all = []
            
            gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size)
            
            for i, img_path in enumerate(img_paths):
                
                # skip if the image path is not valid
                if (not img_path) or (not os.path.exists(img_path)):
                    continue
                    
                # read and resize image
                img = io.imread(img_path)
                img = resize(img, self.img_size)
                
                # convert image to torch tensor and reshape it so channels come first
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                
                # encode class names as integers
                gt_classes = gt_classes_all[i]
                gt_idx = torch.tensor([self.name2idx[name] for name in gt_classes])
                
                img_data_all.append(img_tensor)
                gt_idxs_all.append(gt_idx)
            
            # pad bounding boxes and classes so they are of the same size
            gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
            gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
            
            # stack all images
            img_data_stacked = torch.stack(img_data_all, dim=0)
            
            return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad
    
    img_width = 640
    img_height = 480
    annotation_path = str(Path(__file__).parent / 'data/annotations.xml')
    image_dir = str(Path(__file__).parent / 'data/images')
    name2idx = {'pad': -1, 'camel': 0, 'bird': 1}
    idx2name = {v:k for k, v in name2idx.items()}

    od_dataset = ObjectDetectionDataset(annotation_path, image_dir, (img_height, img_width), name2idx)
    od_dataloader = DataLoader(od_dataset, batch_size=2)

    for img_batch, gt_bboxes_batch, gt_classes_batch in od_dataloader:
        img_data_all = img_batch
        gt_bboxes_all = gt_bboxes_batch
        gt_classes_all = gt_classes_batch
        break

    model = torchvision.models.resnet50(pretrained=True)
    req_layers = list(model.children())[:8]
    backbone = nn.Sequential(*req_layers)

    # unfreeze all the parameters
    for param in backbone.named_parameters():
        param[1].requires_grad = True

    out = backbone(img_data_all)
    out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)

    width_scale_factor = img_width // out_w
    height_scale_factor = img_height // out_h

    anc_pts_x, anc_pts_y = gen_anc_centers((out_h, out_w))
    anc_pts_x_proj = anc_pts_x * width_scale_factor
    anc_pts_y_proj = anc_pts_y * height_scale_factor

    # project anchor centers onto the original image
    anc_pts_x_proj = anc_pts_x.clone() * width_scale_factor
    anc_pts_y_proj = anc_pts_y.clone() * height_scale_factor

    anc_scales = [2, 4, 6]
    anc_ratios = [0.5, 1, 1.5]
    # number of anchor boxes for each anchor point
    n_anc_boxes = len(anc_scales) * len(anc_ratios)

    anc_base = gen_anc_base(
        anc_pts_x, anc_pts_y, anc_scales, anc_ratios, (out_h, out_w))
    
    # since all the images are scaled to the same size
    # we can repeat the anchor base for all the images
    anc_boxes_all = anc_base.repeat(img_data_all.size(dim=0), 1, 1, 1, 1)

    # project gt bboxes onto the feature map
    gt_bboxes_proj = project_bboxes(gt_bboxes_all, width_scale_factor, height_scale_factor, mode='p2a')

    pos_thresh = 0.7
    neg_thresh = 0.3

    positive_anc_ind, negative_anc_ind, GT_conf_scores, \
    GT_offsets, GT_class_pos, positive_anc_coords, \
    negative_anc_coords, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, gt_classes_all, pos_thresh, neg_thresh)

    img_size = (img_height, img_width)
    out_size = (out_h, out_w)
    n_classes = len(name2idx) - 1 # exclude pad idx
    roi_size = (2, 2)

    detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size)

    detector.eval()
    total_loss = detector(img_batch, gt_bboxes_batch, gt_classes_batch)
    proposals_final, conf_scores_final, classes_final = detector.inference(img_batch)

    learning_rate = 1e-3
    n_epochs = 1000

    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
    
    detector.train()
    loss_list = []
    
    for i in range(n_epochs):
        total_loss = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in od_dataloader:
            
            # forward pass
            loss = detector(img_batch, gt_bboxes_batch, gt_classes_batch)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        loss_list.append(total_loss)
