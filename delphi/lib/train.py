"""
Training routines and utilities.
"""

import json
from pathlib import Path

import numpy as np

from delphi.lib.annotate import find_annotations, load_annotation_file
from delphi.lib.cluster import cluster_pixels
from delphi.lib.common import associate_by_filename
from delphi.lib.image import (
    find_images,
    get_pixels_at_radius,
    get_pixels_in_circle,
    load_image,
)
from delphi.lib.log import get_logger

logger = get_logger(__name__)


def train_model(
    image_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    delta_1: int = 25,
    delta_2: int = 3,
) -> Path:
    """
    Train a model.

    Args:
        image_dir: Directory containing images. Subdirectories are searched recursively.
        annotation_dir: Directory containing annotations. Subdirectories are searched recursively. Paired with images by filename stem.
        output_dir: Directory to save model file to.
        delta_1: Radius of negative pixels.
        delta_2: Radius of positive pixels.

    Returns:
        Path to model file.

    Examples:
        >>> train_model(Path("images"), Path("annotations"), Path("models"), 25, 3)
        train - INFO - Training on 10000 pixels: 1000 positives, 9000 negatives.
        train - INFO - Saved model to models/model.json
        Path("models/model.json")
    """
    # Find images
    image_paths = list(find_images(image_dir, recursive=True))
    logger.debug(f"Found {len(image_paths)} images.")

    # Find annotations
    annotation_paths = list(find_annotations(annotation_dir, recursive=True))
    logger.debug(f"Found {len(annotation_paths)} annotations.")

    # Pair images and annotations
    image_annotation_pairs = associate_by_filename(image_paths, annotation_paths)

    # Find the positives and negatives in all training images
    positives = set()
    negatives = set()
    for image_path, annotation_path in image_annotation_pairs:
        logger.debug(f"Processing {image_path.name} and {annotation_path.name}")
        image = load_image(image_path)
        logger.debug(f"Loaded image {image_path.name} with shape {image.shape}")
        annotation = load_annotation_file(annotation_path)
        logger.debug(
            f"Loaded annotation {annotation_path.name} with shape {annotation.shape}"
        )

        for point in annotation:
            annotation_positives = set(
                tuple(pixel)
                for pixel in get_pixels_in_circle(image, point[::-1], delta_2)
            )
            positives.update(annotation_positives)

            annotation_negatives = set(
                tuple(pixel)
                for pixel in get_pixels_at_radius(image, point[::-1], delta_1)
            )
            negatives.update(annotation_negatives)

    pixels = positives | negatives
    logger.info(
        f"Training on {len(pixels)} pixels: {len(positives)} positives, {len(negatives)} negatives."
    )

    # Turn pixel information into Numpy arrays
    pixels_array = np.array(list(pixels))
    positives_array = np.array(list(positives))

    # Cluster pixels
    kmeans_model = cluster_pixels(pixels_array)

    # Predict on positives
    positive_clusters = kmeans_model.predict(positives_array)

    # Find most common cluster
    most_common_cluster = np.bincount(positive_clusters).argmax()

    # Compute mean distance of positives in most common cluster to cluster center
    positive_distances = kmeans_model.transform(positives_array)
    most_common_distances = positive_distances[positive_clusters == most_common_cluster]
    mean_most_common_distance = np.mean(most_common_distances, axis=(0, 1))

    # Get centroid of most common cluster
    centroid = kmeans_model.cluster_centers_[most_common_cluster]

    output = {"color": centroid.tolist(), "threshold": mean_most_common_distance}
    logger.debug(f"Output: {output}")

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    with open(model_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved model to {model_path}")

    return model_path
