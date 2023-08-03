"""A module that contains RCNN model class."""

from typing import Tuple, Iterable, List
from functools import partial

import torch
from torch import FloatTensor, IntTensor
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.ops as ops

from rcnn_utils import (
    generate_anchors, get_required_anchors, generate_anchor_boxes,
    project_bboxes)


class FeatureExtractor(nn.Module):
    """Feature extractor backbone."""

    def __init__(self, model_name: str, input_size: Tuple[int, int]) -> None:
        super().__init__()

        models = {
            'resnet18': partial(
                torchvision.models.resnet18,
                weights=torchvision.models.ResNet18_Weights.DEFAULT),
            'resnet34': partial(
                torchvision.models.resnet34,
                weights=torchvision.models.ResNet34_Weights.DEFAULT),
            'resnet50': partial(
                torchvision.models.resnet50,
                weights=torchvision.models.ResNet50_Weights.DEFAULT),
            'resnet101': partial(
                torchvision.models.resnet101,
                weights=torchvision.models.ResNet101_Weights.DEFAULT),
            'resnet152': partial(
                torchvision.models.resnet152,
                weights=torchvision.models.ResNet152_Weights.DEFAULT),
        }
        if model_name not in models:
            raise KeyError('Got model that is not supported.')
        
        # Get pretrained backbone
        resnet = models[model_name]()
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:8])

        # Check out shape
        input_placeholder = torch.rand(1, 3, *input_size)
        _, self.out_c, self.out_h, self.out_w = (
            self.backbone(input_placeholder).shape)

        # Unfreeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, input_data: FloatTensor) -> FloatTensor:
        """Pass through backbone feature extractor.

        Parameters
        ----------
        input_data : FloatTensor
            An input images with shape `(b, 3, img_h, img_w)`.

        Returns
        -------
        FloatTensor
            An output feature map with shape `(b, map_c, map_h, map_w)`.
        """
        return self.backbone(input_data)
        

class ProposalModule(nn.Module):
    """A proposal module.
    
    It contains confidence and regression heads.
    First one predicts a probability that a given anchor box contains
    some object.
    Second one predicts offsets relative to a ground truth bounding box.
    Offsets are (dxc, dyc, dw, dh) tensor.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 512,
        n_anchors: int = 9,
        p_dropout: float = 0.3
    ) -> None:
        """Initialize `ProposalModule`.

        Parameters
        ----------
        in_features : int
            A number of channels of an input tensor.
        hidden_dim : int, optional
            A hidden number of channels that pass into heads. By default 512.
        n_anchors : int, optional
            A number of bounding boxes per an anchor point. By default is 9.
        p_dropout : float, optional
            A dropout probability. By default is 0.3.
        """
        super().__init__()
        self.hidden_conv = nn.Conv2d(
            in_features, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout, inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_scores_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)

    def forward(
        self,
        feature_maps: FloatTensor,
        pos_anc_idxs: IntTensor,
        neg_anc_idxs: IntTensor
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """Forward pass of `ProposalModule`.

        Parameters
        ----------
        feature_maps : FloatTensor
            A feature map tensor with shape `(b, n_channels, map_h, map_w)`.
        pos_anc_idxs : IntTensor
            Indexes of positive anchor boxes when batch tensor is flatten.
            Shape is `(n_pos_anc,)`.
        neg_anc_idxs : IntTensor
            Indexes of negative anchor boxes when batch tensor is flatten.
            Shape is `(n_pos_anc,)`.

        Returns
        -------
        Tuple[FloatTensor, FloatTensor, FloatTensor]
            Positive and negative object confidence scores,
            with shapes `(n_pos_anc,)` and `(n_neg_anc,)`
            and offsets of positive anchors with shape `(n_pos_anc, 4)`.
        """
        x = self.hidden_conv(feature_maps)
        x = self.relu(x)
        x = self.dropout(x)
        conf_scores_pred: FloatTensor = self.conf_scores_head(x)
        offsets_pred: FloatTensor = self.reg_head(x)

        conf_scores_pred = conf_scores_pred.permute(0, 2, 3, 1)
        offsets_pred = offsets_pred.permute(0, 2, 3, 1)

        pos_conf_scores = conf_scores_pred.flatten()[pos_anc_idxs]
        neg_conf_scores = conf_scores_pred.flatten()[neg_anc_idxs]
        pos_offsets = offsets_pred.contiguous().view(-1, 4)[pos_anc_idxs]
        return pos_conf_scores, neg_conf_scores, pos_offsets

    def inference(
        self, feature_maps: FloatTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        """Inference pass of `ProposalModule`.

        Parameters
        ----------
        feature_maps : FloatTensor
            Backbone's feature maps.

        Returns
        -------
        Tuple[FloatTensor, FloatTensor]
            Object confidence scores with shape `(b, n_anc_box, map_h, map_w)`
            and offsets for every anchor box
            with shape `(b, n_anc_box * 4, map_h, map_w)`.
        """
        x = self.hidden_conv(feature_maps)
        x = self.relu(x)
        x = self.dropout(x)
        conf_scores_pred: FloatTensor = self.conf_scores_head(x)
        offsets_pred: FloatTensor = self.reg_head(x)

        conf_scores_pred = conf_scores_pred.permute(0, 2, 3, 1)
        offsets_pred = offsets_pred.permute(0, 2, 3, 1)
        return conf_scores_pred, offsets_pred


class RegionProposalNetwork(nn.Module):
    """Region proposal network.

    It includes a `FeatureExtractor` backbone
    that is used for feature map extracting
    and a `ProposalModule` for bounding boxes generating.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        backbone_model: str = 'resnet50',
        anc_scales: Iterable[float] = (2.0, 4.0, 6.0),
        anc_ratios: Iterable[float] = (0.5, 1.0, 1.5),
        pos_anc_thresh: float = 0.7,
        neg_anc_thresh: float = 0.3,
        w_conf: float = 1.0,
        w_reg: float = 5.0,
        proposal_module_hid_dim: int = 512,
        proposals_module_p_dropout: float = 0.3
    ) -> None:
        """Initialize region proposal network.

        Parameters
        ----------
        input_size : Tuple[int, int]
            Input images' size.
        backbone_model : str, optional
            A name of a backbone model. A resnet family is supported.
            By default it is equal `"resnet50"`.
        anc_scales : Iterable[float], optional
            Scale factors of anchor bounding boxes.
            By default is (2.0, 4.0, 6.0).
        anc_ratios : Iterable[float], optional
            Ratio factors of anchor bounding boxes sides.
            By default is (0.5, 1.0, 1.5).
        pos_anc_thresh : float, optional
            A confidence threshold for positive anchors selecting.
            By default is 0.7.
        neg_anc_thresh : float, optional
            A confidence threshold for negative anchors selecting.
            By default is 0.3.
        w_conf : float, optional
            Weight coefficient for confidence loss. By default is 1.0.
        w_reg : float, optional
            Weight coefficient for predicted offsets regression loss.
            By default is 5.0.
        """
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone_model, input_size)
        self.backbone_c = self.feature_extractor.out_c
        self.backbone_h = self.feature_extractor.out_h
        self.backbone_w = self.feature_extractor.out_w
        self.proposal_module = ProposalModule(
            in_features=self.backbone_c,
            hidden_dim=proposal_module_hid_dim,
            n_anchors=len(anc_scales) * len(anc_ratios),
            p_dropout=proposals_module_p_dropout)
        self.scores_sigmoid = nn.Sigmoid()

        self.pos_anc_thresh = pos_anc_thresh
        self.neg_anc_thresh = neg_anc_thresh

        self.height_scale = input_size[0] // self.backbone_h
        self.width_scale = input_size[1] // self.backbone_w

        self.w_conf = w_conf
        self.w_reg = w_reg

        x_anc_pts, y_anc_pts = generate_anchors(
            (self.backbone_h, self.backbone_w))
        anchor_grid = torch.nn.Parameter(generate_anchor_boxes(
            x_anc_pts, y_anc_pts, anc_scales, anc_ratios,
            (self.backbone_h, self.backbone_w)))
        self.register_buffer('anchor_grid', anchor_grid)

    def forward(
        self,
        images: FloatTensor,
        gt_boxes: FloatTensor,
        gt_cls: FloatTensor
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, IntTensor, FloatTensor]:
        """Forward pass of the region proposal network.

        A given image pass through the backbone network,
        and feature map is gotten.
        Then this feature map is used for creating bounding boxes proposals.
        Loss is calculated based on the created bounding boxes proposals.

        Parameters
        ----------
        images : FloatTensor
            An input batch of images with shape `(b, c, h, w)`.
        gt_boxes : FloatTensor
            The ground truth bounding boxes with shape `(b, n_max_obj, 4)`.
        gt_cls : Tensor, optional
            The ground truth classes with shape `(b, n_max_obj)`.

        Returns
        -------
        Tuple[FloatTensor, FloatTensor, FloatTensor, IntTensor, FloatTensor]
            Return:
            `rpn_loss` containing loss,
            `feature_maps` with shape `(out_channels, out_size_h, out_size_w)`
            containing containing backbone's feature map,
            `proposals` with shape `(n_pred_pos_anc, 4)`
            containing generated proposals in format "xyxy",
            `pos_b_idxs` with shape `(n_pred_pos_anc,)`
            containing batch indexes
            and `gt_class_pos` with shape `(n_pred_pos_anc,)`,
            containing ground truth classes of predicted positive anchors.
        """
        b_size = images.shape[0]

        feature_maps = self.feature_extractor(images)

        batch_anc_grid = self.anchor_grid.repeat((b_size, 1, 1, 1, 1))
        batch_anc_grid = batch_anc_grid.view(b_size, -1, 4)

        gt_boxes_map = project_bboxes(
            gt_boxes, self.width_scale, self.height_scale, 'p2a')

        (pos_anc_idxs, neg_anc_idxs, pos_b_idxs,
            pos_ancs, neg_ancs, gt_pos_anc_conf_scores,
            gt_class_pos, gt_offsets) = get_required_anchors(
            batch_anc_grid, gt_boxes_map, gt_cls,
            self.pos_anc_thresh, self.neg_anc_thresh)

        (pos_conf_scores, neg_conf_scores,
            pos_offsets) = self.proposal_module(
            feature_maps, pos_anc_idxs, neg_anc_idxs)
        
        rpn_loss = self.rpn_loss(pos_conf_scores, neg_conf_scores, pos_offsets,
                                 gt_offsets, b_size)
        
        proposals = self._generate_proposals(pos_ancs, pos_offsets)

        return rpn_loss, feature_maps, proposals, pos_b_idxs, gt_class_pos
    
    def inference(
        self, images: FloatTensor, conf_thresh: float = 0.5
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, IntTensor]:
        """Inference pass of the region proposal network.

        Parameters
        ----------
        images : FloatTensor
            An input batch of images with shape `(b, c, h, w)`.
        conf_thresh : float, optional
            Object confidence threshold. By default is 0.5.

        Returns
        -------
        Tuple[FloatTensor, FloatTensor, FloatTensor, IntTensor]
            Return:
            `feature_maps` with shape `(out_channels, out_size_h, out_size_w)`
            containing backbone's feature map,
            `proposals` with shape `(n_pred_pos_anc, 4)`
            containing generated proposals in format "xyxy",
            `pos_confs` with shape `(n_pred_pos_anc,)`,
            containing object confidences for generated proposals
            and `pos_b_idxs` with shape `(n_pred_pos_anc,)` batch indexes.
        """
        b_size = images.shape[0]

        feature_maps = self.feature_extractor(images)
        scores, offsets = self.proposal_module.inference(feature_maps)
        confidences = self.scores_sigmoid(scores)

        pos_b_idxs = torch.where(confidences >= conf_thresh)[0]
        confidences = confidences.flatten()
        offsets = offsets.contiguous().view((-1, 4))
        pos_anc_idxs = torch.where(confidences >= conf_thresh)[0]

        pos_confs = confidences[pos_anc_idxs]
        pos_offsets = offsets[pos_anc_idxs]

        batch_anc_grid = self.anchor_grid.repeat((b_size, 1, 1, 1, 1))
        pos_ancs = batch_anc_grid.flatten(end_dim=-2)[pos_anc_idxs]

        proposals = self._generate_proposals(pos_ancs, pos_offsets)
        proposals = project_bboxes(
            proposals, self.width_scale, self.height_scale, mode='a2p')

        return feature_maps, proposals, pos_confs, pos_b_idxs
            
    def _generate_proposals(
        self,
        anchors: FloatTensor,
        offsets: FloatTensor
    ) -> FloatTensor:
        """Generate proposals with anchor boxes and its offsets.

        Get anchors and apply offsets to them.

        Parameters
        ----------
        anchors : FloatTensor
            The anchor boxes with shape `(n_anc, 4)`.
        offsets : FloatTensor
            The offsets with shape `(n_anc, 4)`.

        Returns
        -------
        FloatTensor
            The anchor boxes shifted by the offsets with shape `(n_anc, 4)`.
        """
        anchors = ops.box_convert(anchors, 'xyxy', 'cxcywh')
        proposals = torch.zeros_like(anchors)
        proposals[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
        proposals[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
        proposals[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
        proposals[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

        proposals = ops.box_convert(proposals, 'cxcywh', 'xyxy')
        return proposals

    def rpn_loss(
        self,
        pos_conf: FloatTensor,
        neg_conf: FloatTensor,
        pos_offsets: FloatTensor,
        gt_offsets: FloatTensor,
        b_size: int
    ) -> FloatTensor:
        """Calculate region proposal network's loss.

        RPN loss is a weighted sum of object confidence
        and bounding box offsets regression losses.

        Parameters
        ----------
        pos_conf : FloatTensor
            Confidence scores of predicted positive anchors.
        neg_conf : FloatTensor
            Confidence scores of gotten negative anchors.
        pos_offsets : FloatTensor
            Predicted offsets for positive anchors.
        gt_offsets : FloatTensor
            Ground truth offsets for positive anchors.
        b_size : int
            Batch size.

        Returns
        -------
        FloatTensor
            Calculated RPN loss.
        """
        reg_loss = bbox_reg_loss(pos_offsets, gt_offsets, b_size)
        conf_loss = confidence_loss(pos_conf, neg_conf, b_size)
        total_rpn_loss = self.w_reg * reg_loss + self.w_conf * conf_loss
        return total_rpn_loss
        
    
def bbox_reg_loss(
    predicted_offsets: FloatTensor, gt_offsets: FloatTensor, b_size: int
) -> FloatTensor:
    """Calculate bounding boxes regression loss.

    Loss is calculated as smooth l1 distance between the predicted offsets
    and the ground truth, divided by batch size.

    Parameters
    ----------
    predicted_offsets : FloatTensor
        Predicted offsets with shape `(n_pred_pos_anc, 4)`.
    gt_offsets : FloatTensor
        Ground truth offsets with shape `(n_pred_pos_anc, 4)`.
    b_size : int
        Batch size.

    Returns
    -------
    FloatTensor
        Calculated regression loss.
    """
    return F.smooth_l1_loss(
        predicted_offsets, gt_offsets, reduction='sum') / b_size
    

def confidence_loss(
    conf_scores_pos: FloatTensor,
    conf_scores_neg: FloatTensor,
    b_size: int
) -> FloatTensor:
    """Calculate object confidence loss.

    Loss is represented as binary cross entropy loss calculated by positive and
    negative anchors' confidences and corresponding ones
    and zeros ground truth.

    Parameters
    ----------
    conf_scores_pos : FloatTensor
        Predicted confidence scores of predicted positive anchors.
    conf_score_neg : FloatTensor
        Predicted confidence scores of selected negative anchors.
    b_size : int
        Batch size.

    Returns
    -------
    FloatTensor
        Calculated confidence loss.
    """
    gt_pos = torch.ones_like(conf_scores_pos)
    gt_neg = torch.zeros_like(conf_scores_neg)
    gt_scores = torch.cat((gt_pos, gt_neg))
    predicted_scores = torch.cat((conf_scores_pos, conf_scores_neg))
    return F.binary_cross_entropy_with_logits(
        predicted_scores, gt_scores, reduction='sum') / b_size


class ClassificationModule(nn.Module):
    def __init__(
        self,
        out_channels: int,
        n_cls: int,
        roi_size: Tuple[int, int],
        hidden_dim: int = 512,
        p_dropout: float = 0.3
    ) -> None:
        """Initialize `ClassificationModule`.

        Parameters
        ----------
        out_channels : int
            An expected number of channels of a backbone's feature map.
        n_cls : int
            A number of classes.
        roi_size : Tuple[int, int]
            Size of ROI pool.
        hidden_dim : int, optional
            Hidden dimension, by default is 512.
        p_dropout : float, optional
            Dropout's probability, by default is 0.3.
        """
        super().__init__()
        self.roi_size = roi_size
        self.avg_pool = nn.AvgPool2d(roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.cls_head = nn.Linear(hidden_dim, n_cls)

    def forward(
        self,
        feature_maps: FloatTensor,
        predicted_proposals: List[FloatTensor],
        gt_cls: FloatTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        """Forward pass of `ClassificationModule`.

        Calculate class scores for the given proposals
        and based on it cross entropy loss is calculated.

        Parameters
        ----------
        predicted_proposals : FloatTensor
            Feature map from RPN's backbone
            with shape `(b, out_channels, out_size_h, out_size_w)`.
        predicted_proposals : List[Tensor]
            Predicted proposals from RPN.
            List has length `n_pos_anc` and each element has shape `(4,)`.
        gt_cls : Tensor
            Ground truth classes with shape `(n_pos_anc,)`.

        Returns
        -------
        Tuple[FloatTensor, FloatTensor]
            Class scores with shape `(n_pos_anc,)'
            and additional class loss with shape `(n_cls,)`
        """
        # Check is there even one proposal
        props_mask = torch.tensor(
            [props.shape[0] == 0 for props in predicted_proposals])
        if props_mask.all():
            b = len(predicted_proposals)
            device = props_mask[0].device
            return torch.FloatTensor(size=(0, b), device=device), 0.0

        x = ops.roi_pool(feature_maps, predicted_proposals, self.roi_size)
        x = self.avg_pool(x).flatten(start_dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        cls_scores = self.cls_head(x)
        return cls_scores, F.cross_entropy(cls_scores, gt_cls.long())
        
    def inference(
        self,
        feature_maps: FloatTensor,
        predicted_proposals: List[FloatTensor]
    ) -> FloatTensor:
        """Inference pass of `ClassificationModule`.

        Calculate class scores for the given proposals.

        Parameters
        ----------
        feature_maps : FloatTensor
            Feature map from RPN's backbone
            with shape `(b, out_channels, out_size_h, out_size_w)`.
        predicted_proposals : List[FloatTensor]
            Predicted proposals from RPN.
            List has length `n_pos_anc` and each element has shape `(4,)`.

        Returns
        -------
        FloatTensor
            Class scores with shape `(n_pos_anc,)'.
        """
        x = ops.roi_pool(feature_maps, predicted_proposals, self.roi_size)
        x = self.avg_pool(x).flatten(start_dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        cls_scores = self.cls_head(x)
        return cls_scores


class RCNN_Detector(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int],
        n_cls: int,
        roi_size: Tuple[int, int],
        backbone_model: str = 'resnet50',
        anc_scales: Iterable[float] = (2.0, 4.0, 6.0),
        anc_ratios: Iterable[float] = (0.5, 1.0, 1.5),
        pos_anc_thresh: float = 0.7,
        neg_anc_thresh: float = 0.3,
        w_conf_loss: float = 1.0,
        w_reg_loss: float = 5.0,
        proposal_module_hid_dim: int = 512,
        classifier_hid_dim: int = 512,
        proposals_module_p_dropout: float = 0.3,
        classifier_p_dropout: float = 0.3
    ) -> None:
        """Initialize R-CNN network.

        Parameters
        ----------
        input_size : Tuple[int, int]
            A size of input images.
        n_cls : int
            A number of classification classes.
        roi_size : Tuple[int, int]
            A size for RoI pulling.
        backbone_model : str, optional
            A name of a backbone model. A resnet family is supported.
            By default it is equal `"resnet50"`.
        anc_scales : Iterable[float], optional
            Scale factors of anchor bounding boxes.
            By default is (2.0, 4.0, 6.0).
        anc_ratios : Iterable[float], optional
            Ratio factors of anchor bounding boxes sides.
            By default is (0.5, 1.0, 1.5).
        pos_anc_thresh : float, optional
            Confidence threshold for selecting positive anchors.
            By default is 0.7.
        neg_anc_thresh : float, optional
            Confidence threshold for selecting negative anchors.
            By default is 0.3.
        w_conf_loss : float, optional
            Weight coefficient for confidence loss. By default is 1.0.
        w_reg_loss : float, optional
             Weight coefficient for predicted offsets regression loss.
             By default is 5.0.
        classifier_hid_dim : int, optional
            A size of hidden dimension for classifier, by default is 512.
        classifier_p_dropout : float, optional
            A probability for classifier's dropout. By default is 0.3.
        """
        super().__init__()
        self.rpn = RegionProposalNetwork(
            input_size=input_size, backbone_model=backbone_model,
            anc_scales=anc_scales, anc_ratios=anc_ratios,
            pos_anc_thresh=pos_anc_thresh, neg_anc_thresh=neg_anc_thresh,
            w_conf=w_conf_loss, w_reg=w_reg_loss,
            proposal_module_hid_dim=proposal_module_hid_dim,
            proposals_module_p_dropout=proposals_module_p_dropout)
        self.classifier = ClassificationModule(
            out_channels=self.rpn.backbone_c, n_cls=n_cls, roi_size=roi_size,
            hidden_dim=classifier_hid_dim, p_dropout=classifier_p_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        images: FloatTensor,
        gt_boxes: FloatTensor,
        gt_cls: FloatTensor
    ) -> Tuple[List[FloatTensor], List[FloatTensor], FloatTensor]:
        """Forward pass of R-CNN.

        Get images, ground truth bounding boxes and corresponding classes
        and return proposal bounding boxes, corresponding classes probabilities
        and calculated loss.

        Parameters
        ----------
        images : Tensor
            A batch of the input images
            with shape `(b, 3, img_h, img_w)`.
        gt_boxes : Tensor, optional
            The ground truth bounding boxes with shape `(b, n_max_obj, 4)`.
        gt_cls : Tensor, optional
            The ground truth classes with shape `(b, n_max_obj)`.

        Returns
        -------
        Tuple[List[FloatTensor], List[FloatTensor], FloatTensor]]
            Generated proposals list with length `b_size`
            and each element has shape `(n_pos_anc_per_img, 4)`.
            Predicted classes confidences list with length 'b_size'
            that correspond to generated proposals.
            Each element has shape `(n_pos_anc_per_img, n_cls)`.
            Calculated R-CNN loss.
        """
        b_size = images.shape[0]

        rpn_loss, feature_maps, proposals, pos_b_idxs, gt_class_pos = (
            self.rpn(images, gt_boxes, gt_cls))
        
        proposals_list = []
        for i in range(b_size):
            proposals_idxs = torch.where(pos_b_idxs == i)[0]
            proposals_list.append(
                proposals[proposals_idxs].detach().clone())

        cls_scores, classifier_loss = (
            self.classifier(feature_maps, proposals_list, gt_class_pos))
        total_loss = rpn_loss + classifier_loss

        cls_scores_list = []
        idx = 0
        for image_proposals in proposals_list:
            cls_scores_list.append(cls_scores[idx:len(image_proposals)])
            idx += len(image_proposals)

        return proposals_list, cls_scores_list, total_loss
        
    def inference(
        self,
        images: FloatTensor,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.7
    ) -> Tuple[List[FloatTensor], List[FloatTensor]]:
        """Inference pass of R-CNN.

        Get images and generate proposal bounding boxes filtered with
        object confidence and IoU non maximum suppression thresholds.
        For these bounding boxes classes probabilities are calculated.

        Parameters
        ----------
        images : FloatTensor
            A batch of the input images
            with shape `(b, 3, img_h, img_w)`.
        conf_thresh : float, optional
            Object confidence threshold. By default is 0.5.
        nms_thresh : float, optional
            IoU NMS threshold. By default is 0.7.

        Returns
        -------
        Tuple[List[FloatTensor], List[FloatTensor]]
            Generated proposals list with length `b_size`
            and each element has shape `(n_pos_anc_per_img, 4)`
            and predicted classes list with length `b_size`
            and each element has shape `(n_pos_anc_per_img, n_cls)`.
        """
        with torch.no_grad():
            b_size = images.shape[0]

            feature_maps, proposals, confidences, pos_b_idxs = (
                self.rpn.inference(images, conf_thresh=conf_thresh))

            conf_pred = []
            proposals_list = []
            for i in range(b_size):
                cur_idxs = torch.where(pos_b_idxs == i)
                cur_confs = confidences[cur_idxs]
                cur_props = proposals[cur_idxs]
                nms_idxs = ops.nms(cur_props, cur_confs, nms_thresh)
                conf_pred.append(cur_confs[nms_idxs])
                proposals_list.append(cur_props[nms_idxs])

            cls_scores = self.classifier.inference(
                feature_maps=feature_maps, predicted_proposals=proposals_list)
            cls_conf = self.softmax(cls_scores)

            cls_conf_list = []
            idx = 0
            for image_proposals in proposals_list:
                cls_conf_list.append(
                    cls_conf[idx:idx + len(image_proposals)])
                idx += len(image_proposals)
            
            return proposals_list, cls_conf_list
