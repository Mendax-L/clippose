import collections.abc as collections
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
from PIL import Image

class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.conf.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.conf.resize,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale


def map_tensor(input_, func: Callable):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = "cpu", non_blocking: bool = True):
    """Move batch (dict) to device"""

    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking).detach()

    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


# def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
#     """Read an image from path as RGB or grayscale"""
#     if not Path(path).exists():
#         raise FileNotFoundError(f"No image at path {path}.")
#     mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
#     image = cv2.imread(str(path), mode)
#     if image is None:
#         raise IOError(f"Could not read image at {path}.")
#     if not grayscale:
#         image = image[..., ::-1]
#     return image
def read_image(image: np.ndarray, grayscale: bool = False) -> np.ndarray:
    """Process an image (either numpy array or PIL image) as RGB or grayscale"""
    if isinstance(image, np.ndarray):
        # If the input is already a numpy array, check if it's grayscale or color
        if grayscale:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        else:
            return image[..., ::-1]  # Convert BGR to RGB
    elif isinstance(image, Image.Image):
        # Convert PIL Image to numpy array
        image = np.array(image)
        return read_image(image, grayscale)
    else:
        raise TypeError("Input image must be a numpy array or PIL Image.")


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


# def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
#     image = read_image(path)
#     if resize is not None:
#         image, _ = resize_image(image, resize, **kwargs)
#     return numpy_image_to_torch(image)



def load_image(image: np.ndarray, resize: int = None, **kwargs) -> torch.Tensor:
    """Load image from numpy array or PIL image, and resize if needed."""
    image = read_image(image)
    if resize is not None:
        image = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)

class Extractor(torch.nn.Module):
    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})

    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats

def extract_features(extractor, image: torch.Tensor, **preprocess):
    """
    Extract features from a single image using the given extractor.

    Args:
    - extractor: The feature extractor (e.g., SuperPoint).
    - image (torch.Tensor): The input image tensor.
    - **preprocess: Additional preprocessing configurations.

    Returns:
    - feats (dict): The extracted features.
    """
    feats = extractor.extract(image, **preprocess)
    return feats

def match_pair(
    matcher,
    feats0,
    feats1,
    device: str = "cpu",
):
    """Match a pair of images (image0, image1) with an extractor and matcher"""
    matches01 = matcher({"image0": feats0, "image1": feats1})
    data = [feats0, feats1, matches01]
    # remove batch dim and move to target device
    feats0, feats1, matches01 = [batch_to_device(rbd(x), device) for x in data]
    return feats0, feats1, matches01
