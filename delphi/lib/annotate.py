"""
Annotation operations.
"""

import warnings
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from delphi.lib.image import find_images, load_image
from delphi.lib.log import get_logger

warnings.simplefilter("ignore")

logger = get_logger(__name__)


def find_annotations(annotation_dir: Path, recursive: bool = False) -> Iterable[Path]:
    """
    Find annotation files in a directory.

    Args:
        annotation_dir: Directory to search.
        recursive: Whether to search recursively.

    Yields:
        Annotation file paths.

    Examples:
        >>> annotation_dir = Path("tests/data")
        >>> list(find_annotation_files(annotation_dir))
        [PosixPath('tests/data/1.txt'), PosixPath('tests/data/2.txt')]
    """
    for file in annotation_dir.iterdir():
        if file.is_dir() and recursive:
            yield from find_annotations(file, recursive)
        elif file.is_file() and file.suffix == ".txt":
            yield file


def load_annotation_file(annotation_path: Path) -> np.ndarray:
    """
    Load an annotation file.

    Args:
        annotation_path: Path to annotation file.

    Returns:
        Annotated points. Each row is an x, y pair.

    Examples:
        >>> annotations = load_annotation_file(Path("tests/data/1.txt"))
        >>> annotations.shape
        (2, 2)
        >>> annotations
        array([[  0,  10],
               [100, 200]])
    """
    return np.loadtxt(annotation_path, delimiter=",", dtype=np.int32)


def annotate_images(image_dir: Path, annotation_dir: Path, n: int = -1):
    """
    Annotate images using an OpenCV GUI.

    Args:
        image_dir: Directory containing images. Subdirectories are searched recursively.
        annotation_dir: Directory to save annotations. Subdirectories are created as needed to match the image directory structure.
        n: Number of images to annotate. If -1, all images are annotated.
    """
    # Find images
    image_paths = list(find_images(image_dir, recursive=True))
    logger.debug(f"Found {len(image_paths)} images")

    # Limit number of images to annotate
    n = len(image_paths) if n == -1 else n
    logger.info(f"Annotating {n} images")

    for image_path in image_paths[:n]:
        # Load the image
        image = load_image(image_path)
        logger.debug(f"Loaded image {image_path.name} with shape {image.shape}")

        # Construct path to annotation file
        annotation_path = annotation_dir / image_path.relative_to(
            image_dir
        ).with_suffix(".txt")
        logger.debug(f"Annotation path: {annotation_path}")
        annotation_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing annotations, if the file exists
        annotations = []
        if annotation_path.exists():
            annotations = load_annotation_file(annotation_path).tolist()
            logger.debug(
                f"Loaded {len(annotations)} existing annotations from {annotation_path.name}"
            )

        # Define function to show image with annotations
        def show():
            for x, y in annotations:
                cv2.circle(image, (x, y), 3, (0, 0, 255), 1)
            cv2.imshow(str(image_path.name), image)

        # Define callback for mouse click
        def on_mouse_click(event, x, y, flags, param):
            nonlocal annotations
            if event == cv2.EVENT_LBUTTONDOWN:
                annotations.append((x, y))
            show()

        # Show image and wait for keyboard input to move to next image
        show()
        cv2.setMouseCallback(str(image_path.name), on_mouse_click)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save annotations
        with annotation_path.open("w") as f:
            for x, y in annotations:
                f.write(f"{x},{y}\n")
        logger.info(f"Saved {len(annotations)} annotations to {annotation_path.name}")
