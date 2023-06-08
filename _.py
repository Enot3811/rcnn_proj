from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

import rcnn_utils

from object_detection_dataset import ObjectDetectionDataset


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

    (pos_anc_idxs, neg_anc_idxs, pos_b_idxs,
     pos_ancs, neg_ancs, pos_anc_conf_scores,
     gt_class_pos, gt_offsets) = rcnn_utils.get_required_anchors(
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
