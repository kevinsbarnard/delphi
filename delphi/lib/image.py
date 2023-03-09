"""
Image operations.
"""

import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
IMAGE_PATTERN = r".*\.(" + "|".join(IMAGE_EXTENSIONS) + ")$"
IMAGE_REGEX = re.compile(IMAGE_PATTERN, re.IGNORECASE)


def find_images(image_dir: Path, recursive: bool = False) -> Iterable[Path]:
    """
    Find image files in a directory.

    Args:
        image_dir: Directory to search.
        recursive: Whether to search recursively.

    Yields:
        Image paths.

    Examples:
        >>> image_dir = Path("tests/data")
        >>> list(find_images(image_dir))
        [PosixPath('tests/data/1.jpg'), PosixPath('tests/data/2.jpg')]
    """
    for file in image_dir.iterdir():
        if file.is_dir() and recursive:
            yield from find_images(file, recursive)
        elif file.is_file() and IMAGE_REGEX.match(file.name):
            yield file


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image.

    Args:
        image_path: Path to image.

    Returns:
        Image as a numpy array.

    Examples:
        >>> image = load_image(Path("tests/data/1.jpg"))
        >>> image.shape
        (480, 640, 3)
    """
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def get_pixels_in_circle(image: np.ndarray, center: tuple, radius: int) -> np.ndarray:
    """
    Get the pixels in a circle.

    Args:
        image: Image to get pixels from.
        center: Center of the circle.
        radius: Radius of the circle.

    Returns:
        Pixels in the circle.

    Examples:
        >>> image = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
        >>> get_pixels_in_circle(image, (2, 2), 1)
        array([1, 1, 1, 1, 1, 1, 1, 1])
    """
    y, x = np.ogrid[
        -center[0] : image.shape[0] - center[0], -center[1] : image.shape[1] - center[1]
    ]
    mask = x * x + y * y <= radius * radius
    return image[mask]


def get_pixels_at_radius(image: np.ndarray, center: tuple, radius: int) -> np.ndarray:
    """
    Get the pixels at a radius.

    Args:
        image: Image to get pixels from.
        center: Center of the circle.
        radius: Radius of the circle.

    Returns:
        Pixels at the radius.

    Examples:
        >>> image = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
        >>> get_pixels_at_radius(image, (2, 2), 1)
        array([1, 1, 1, 1])
    """
    y, x = np.ogrid[
        -center[0] : image.shape[0] - center[0], -center[1] : image.shape[1] - center[1]
    ]
    mask = x * x + y * y == radius * radius
    return image[mask]
