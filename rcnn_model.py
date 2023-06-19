"""A module that contains RCNN model class."""

from typing import Tuple, Iterable, Union

import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import torchvision.ops as ops

from rcnn_utils import (
    generate_anchors, get_required_anchors, generate_anchor_boxes,
    project_bboxes)


class FeatureExtractor(nn.Module):
    """Feature extractor backbone."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Get pretrained backbone
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:8])
        # Unfreeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, input_data: Tensor) -> Tensor:
        """Pass through backbone feature extractor.

        Parameters
        ----------
        input_data : Tensor
            An input image.

        Returns
        -------
        Tensor
            A feature map.
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
        p_dropout: float = 0.3,
        *args, **kwargs
    ) -> None:
        """Initialize ProposalModule

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
        super().__init__(*args, **kwargs)
        self.hidden_conv = nn.Conv2d(
            in_features, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout, inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)

    def forward(
        self,
        feature_maps: Tensor,
        pos_anc_idxs: Tensor = None,
        neg_anc_idxs: Tensor = None,
        pos_ancs: Tensor = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Get object confidence and offsets prediction.
        
        It has train and evaluation mode.
        If got only `feature_maps` then evaluation mode is activated
        to calculate confidence and offsets.
        If got all other parameters then train mode is activated to calculate
        positive anchors confidence, negative anchors confidence, positive
        anchors offsets and final proposals.

        Parameters
        ----------
        feature_maps : Tensor
            A feature map tensor with shape `[b, n_channels, map_h, map_w]`.
        pos_anc_idxs : Tensor, optional
            Indexes of positive anchor boxes when tensor is flatten.
            Shape is `[n_pos_anc,]`.
            By default is `None`, but during train must be given.
        neg_anc_idxs : Tensor, optional
            Indexes of negative anchor boxes when tensor is flatten.
            Shape is `[n_pos_anc,]`.
            By default is `None`, but during train must be given.
        pos_ancs : Tensor, optional
            Positive anchors with shape `[n_pos_anc, 4]`.
            By default is `None`, but during train must be given.

        Returns
        -------
        Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]
            When evaluate mode return the object confidence for every anchor
            box with shape `[b, n_anc_box, map_h, map_w]` and the offsets with
            shape `[b, n_anc_box * 4, map_h, map_w]`.
            When train mode return the positive and negative object confidence
            separately, with shape `[n_pos_anc,]` and `[n_neg_anc,]`, offsets
            of positive anchors with shape `[n_pos_anc, 4]`,
            proposals of network with shape `[n_pos_anc, 4]`.
        """
        x = self.hidden_conv(feature_maps)
        x = self.relu(x)
        x = self.dropout(x)
        conf_pred: Tensor = self.conf_head(x)
        offsets_pred: Tensor = self.reg_head(x)

        mode = ('eval' if (pos_anc_idxs is None or neg_anc_idxs is None or
                           pos_ancs is None)
                else 'train')
        
        if mode == 'train':
            pos_conf = conf_pred.permute(0, 2, 3, 1).flatten()[pos_anc_idxs]
            neg_conf = conf_pred.permute(0, 2, 3, 1).flatten()[neg_anc_idxs]

            pos_offsets = (offsets_pred.permute(0, 2, 3, 1)
                           .contiguous().view(-1, 4))[pos_anc_idxs]
            
            proposals = self._generate_proposals(pos_ancs, pos_offsets)

            return pos_conf, neg_conf, pos_offsets, proposals

        elif mode == 'eval':
            return conf_pred, offsets_pred
        else:
            raise

    def _generate_proposals(
        self,
        anchors: Tensor,
        offsets: Tensor
    ) -> Tensor:
        """Generate proposals with anchor boxes and its offsets.

        Get anchors and apply offsets to them.

        Parameters
        ----------
        anchors : Tensor
            The anchor boxes with shape `[n_anc, 4]`.
        offsets : Tensor
            The offsets with shape `[n_anc, 4]`.

        Returns
        -------
        Tensor
            The anchor boxes shifted by the offsets with shape `[n_anc, 4]`.
        """
        anchors = ops.box_convert(anchors, 'xyxy', 'cxcywh')
        proposals = torch.zeros_like(anchors)
        proposals[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
        proposals[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
        proposals[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
        proposals[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

        proposals = ops.box_convert(proposals, 'cxcywh', 'xyxy')
        return proposals


class RegionProposalNetwork(nn.Module):

    def __init__(
        self,
        input_size: Tuple[int, int],
        out_size: Tuple[int, int],
        out_channels: int,
        anc_scales: Iterable[float] = (2.0, 4.0, 6.0),
        anc_ratios: Iterable[float] = (0.5, 1.0, 1.5),
        pos_anc_thresh: float = 0.7,
        neg_anc_thresh: float = 0.3,
        w_conf: float = 1.0,
        w_reg: float = 5.0,
        *args, **kwargs
    ) -> None:
        # TODO write docstring
        super().__init__(*args, **kwargs)
        self.feature_extractor = FeatureExtractor()
        self.proposal_module = ProposalModule(
            out_channels, len(anc_scales) * len(anc_ratios))

        self.pos_anc_thresh = pos_anc_thresh
        self.neg_anc_thresh = neg_anc_thresh

        self.height_scale = input_size[0] // out_size[0]
        self.width_scale = input_size[1] // out_size[1]

        x_anc_pts, y_anc_pts = generate_anchors(out_size)
        self.anchor_grid = generate_anchor_boxes(
            x_anc_pts, y_anc_pts, anc_scales, anc_ratios, out_size)


    def forward(self, input_batch: Tensor):
        # TODO Complete development
        # TODO Write docs
        images, gt_boxes, gt_cls = input_batch
        b_size = gt_cls.shape[0]
        batch_anc_grid = self.anchor_grid.repeat((b_size, 1, 1, 1, 1))

        gt_boxes_map = project_bboxes(
            gt_boxes, self.width_scale, self.height_scale, 'p2a')

        feature_maps = self.feature_extractor(images)

        (pos_anc_idxs, neg_anc_idxs, pos_b_idxs,
         pos_ancs, neg_ancs, pos_anc_conf_scores,
         gt_class_pos, gt_offsets) = get_required_anchors(
            batch_anc_grid, gt_boxes_map, gt_cls,
            self.pos_anc_thresh, self.neg_anc_thresh)

        pos_conf, neg_conf, pos_offsets, proposals = self.proposal_module(
            feature_maps, pos_anc_idxs, neg_anc_idxs, pos_ancs)
        
    
def bbox_reg_loss(
    predicted_offsets: Tensor, gt_offsets: Tensor, b_size: int
) -> Tensor:
    """Calculate bounding boxes regression loss.

    Loss is calculated as smooth l1 distance between the predicted offsets
    and the ground truth, divided by batch size.

    Parameters
    ----------
    predicted_offsets : Tensor
        Predicted offsets with shape `[n_pred_pos_anc, 4]`.
    gt_offsets : Tensor
        Ground truth offsets with shape `[n_pred_pos_anc, 4]`.
    b_size : int
        Batch size.

    Returns
    -------
    Tensor
        Calculated regression loss.
    """
    return F.smooth_l1_loss(
        predicted_offsets, gt_offsets, reduction='sum') / b_size
    


def cls_loss():
    pass
    # TODO develope this loss
