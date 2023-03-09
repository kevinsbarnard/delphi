"""
K-Means clustering.
"""

import numpy as np
from sklearn.cluster import KMeans


def cluster_pixels(pixels: np.ndarray, k: int = 7) -> KMeans:
    """
    Cluster pixels using K-Means.

    Args:
        pixels: Pixels to cluster.
        k: Number of clusters.

    Returns:
        K-Means model.

    Examples:
        >>> pixels = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]])
        >>> model = cluster_pixels(pixels, k=3)
        >>> model.cluster_centers_
        array([[4.5, 4.5, 4.5],
               [1.5, 1.5, 1.5],
               [7.5, 7.5, 7.5]])
    """
    model = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(pixels)
    return model
