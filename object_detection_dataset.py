"""
A module that contains ObjectDetectionDataset class.
"""


from pathlib import Path
from typing import Tuple, Dict, List
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import image_utils


class ObjectDetectionDataset(Dataset):
    """
    A Pytorch Dataset class to load the images and
    their corresponding annotations.

    Each sample is a tuple consists of:
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    """
    def __init__(
        self,
        annotation_path: Path,
        img_dir: Path,
        img_size: Tuple[int, int],
        name2index: Dict[str, int]
    ):
        self.img_size = img_size
        self.bboxes, self.labels, self.img_paths = self.parse_annotation(
            annotation_path, img_size, img_dir, name2index)

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Image tensor, bounding boxes' tensor, labels' tensor
        """
        image = image_utils.read_image(self.img_paths[idx])
        image = torch.from_numpy(image_utils.resize_image(
            image, self.img_size).astype(np.float32)).permute(2, 0, 1) / 255
        return image, self.bboxes[idx], self.labels[idx]

    def parse_annotation(
        self,
        annotation_path: Path,
        img_size: Tuple[int, int],
        image_dir: Path,
        name2index: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        Parse an annotation xml. Converts input data to an appropriate format.

        Parameters
        ----------
        annotation_path : Path
            Path to annotation xml file.
        img_size : Tuple[int, int]
            A size to rescale images in dataset.
        image_dir : Path
            A path to directory that contains images.
        name2index: Dict[str, int]
            A dictionary that contains pairs consisting of a label name
            and a corresponding index.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, List]
            Ground truth bounding boxes with shape `[n_samples, n_max_obj, 4]`,
            ground truth classes with shape `[n_samples, n_max_obj]` and
            a list of image paths.
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        new_width, new_height = img_size

        img_paths: List[Path] = []
        gt_boxes: List[torch.Tensor] = []
        gt_classes: List[torch.Tensor] = []

        for sample in root.findall('image'):

            img_paths.append(image_dir.joinpath(sample.get('name')))

            orig_width = int(sample.get('width'))
            orig_height = int(sample.get('height'))

            scale_x = new_width / orig_width
            scale_y = new_height / orig_height

            sample_boxes: List[List[int]] = []
            sample_classes: List[int] = []
            for box in sample.findall('box'):
                xmin = float(box.get('xtl'))
                ymin = float(box.get('ytl'))
                xmax = float(box.get('xbr'))
                ymax = float(box.get('ybr'))

                bbox = torch.Tensor([xmin, ymin, xmax, ymax])
                bbox[[0, 2]] *= scale_x
                bbox[[1, 3]] *= scale_y

                sample_boxes.append(bbox)

                sample_classes.append(name2index[box.get('label')])

            gt_boxes.append(torch.stack(sample_boxes))
            gt_classes.append(
                torch.tensor(sample_classes, dtype=torch.float32))

        gt_boxes = pad_sequence(gt_boxes, batch_first=True, padding_value=-1)
        gt_classes = pad_sequence(gt_classes, batch_first=True,
                                  padding_value=-1)

        return gt_boxes, gt_classes, img_paths
