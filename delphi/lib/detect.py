"""
Point detection routines and utilities.
"""

import itertools
import json
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from delphi.lib.annotate import find_annotations, load_annotation_file
from delphi.lib.image import find_images, load_image
from delphi.lib.log import get_logger

logger = get_logger(__name__)


def detect_points(
    image_dir: Path, annotation_dir: Path, model_path: Path
) -> Iterable[Tuple[Path, np.ndarray]]:
    """
    Detect points in images using a trained model.

    Args:
        image_dir: Directory containing images. Subdirectories are searched recursively.
        annotation_dir: Directory containing annotations. Subdirectories are searched recursively.
        model_path: Path to model file.

    Yields:
        Pairs of image paths and point coordinates.

    Examples:
        >>> image_dir = Path("tests/data")
        >>> annotation_dir = Path("tests/data")
        >>> model_path = Path("tests/data/model.json")
        >>> detect_points(image_dir, annotation_dir, model_path)
        detect - INFO - Points for image 1.jpg: [[100 200], [300 200]]
        detect - INFO - Points for image 2.jpg: [[100 205], [300 204]]
        <generator object detect_points at 0x7f8b1c0b9f68>
    """
    # Find images
    image_paths = list(find_images(image_dir, recursive=True))
    logger.debug(f"Found {len(image_paths)} images")

    # Find annotations
    annotation_paths = list(find_annotations(annotation_dir, recursive=True))
    logger.debug(f"Found {len(annotation_paths)} annotations")

    # Load annotations
    annotations = [
        load_annotation_file(annotation_path) for annotation_path in annotation_paths
    ]

    # Load model
    with open(model_path, "r") as f:
        model = json.load(f)
    logger.debug(f"Loaded model: {model}")

    # Unpack model parameters
    color = np.array(model["color"])
    threshold = model["threshold"]

    # Run detection on all images
    for image_path in image_paths:
        # Load image
        image = load_image(image_path)
        logger.debug(f"Loaded image {image_path.name} with shape {image.shape}")

        # Compute pixel distances from color and threshold
        pixel_distances = np.linalg.norm(image - color, axis=2)
        mask = pixel_distances < threshold

        # Perform open morphological operation to remove noise
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )

        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        logger.debug(f"Found {len(contours)} contours")

        # Weight contours by average pixel distance
        weights = []
        for contour in contours:
            weights.append(np.mean(pixel_distances[contour[:, 0, 1], contour[:, 0, 0]]))
        weights = np.array(weights)

        # Select top 5 contours by weight
        top_contours = sorted(zip(contours, weights), key=lambda x: x[1])[:5]

        # Compute centers of contours
        moments = [cv2.moments(contour) for contour, _ in top_contours]
        centers = [
            (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in moments
        ]
        logger.debug(f"Centers: {centers}")

        # Find best permutation of centers across all annotations
        best_permutation = None
        best_loss = np.inf
        valid_annotations = (
            annotation for annotation in annotations if annotation.shape[0] > 0
        )
        for annotation in valid_annotations:
            if (
                annotation.shape[0] == 0 or len(centers) < annotation.shape[0]
            ):  # Skip empty annotations and too few centers
                continue

            # Compute permutations
            permutations = list(itertools.permutations(centers, annotation.shape[0]))
            logger.debug(f"Found {len(permutations)} permutations")

            for permutation in permutations:
                current_loss = np.mean(
                    np.linalg.norm(np.array(permutation) - annotation, axis=1)
                )
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_permutation = permutation

        if best_permutation is None:
            logger.warning(f"No dots found for {image_path.name}")
            continue

        logger.info(f"Points for {image_path.name}: {best_permutation}")

        yield image_path, best_permutation
