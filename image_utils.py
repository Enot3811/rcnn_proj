"""
A module that contain functions for working with images.
"""


from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch


def read_image(path: Union[Path, str], grayscale: bool = False) -> np.ndarray:
    """
    Read image to numpy array.

    Parameters
    ----------
    path : Union[Path, str]
        Path to image file
    grayscale : bool, optional
        Whether read image in grayscale, by default False

    Returns
    -------
    np.ndarray
        Array containing read image.

    Raises
    ------
    FileNotFoundError
        Did not find image.
    ValueError
        Image reading is not correct.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Did not find image {path}.')
    flag = cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_COLOR
    img = cv.imread(str(path), flag)
    if img is None:
        raise ValueError('Image reading is not correct.')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def resize_image(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to given size.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    new_size : Tuple[int, int]
        Tuple containing new image size.

    Returns
    -------
    np.ndarray
        Resized image
    """
    return cv.resize(
        image, new_size, None, None, None, interpolation=cv.INTER_LINEAR)


def display_image(
    img: Union[torch.Tensor, np.ndarray],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Display an image on a matplotlib figure.

    Parameters
    ----------
    img : Union[torch.Tensor, np.ndarray]
        An image to display. If got torch.Tensor then convert it
        to np.ndarray with axes permutation.
    ax : Optional[plt.Axes], optional
        Axes for image showing. If not given then a new Figure and Axes
        will be created.

    Returns
    -------
    plt.Axes
        Axes with showed image.
    """
    if isinstance(img, torch.Tensor):
        img = img.clone().detach().cpu().permute(1, 2, 0).numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(img)
    return ax
