"""
Common utilities.
"""

from pathlib import Path
from typing import Iterable, Tuple


def associate_by_filename(
    paths_a: Iterable[Path], paths_b: Iterable[Path]
) -> Iterable[Tuple[Path, Path]]:
    """
    Associate two collections of paths by filename stem.

    Args:
        paths_a: First collection of paths.
        paths_b: Second collection of paths.

    Yields:
        Pairs of paths with matching filename stems.

    Examples:
        >>> list(associate_by_filename([Path("a.jpg"), Path("b.jpg")], [Path("b.txt"), Path("c.txt")]))
        [(Path('b.jpg'), Path('b.txt'))]
    """
    remaining_paths_b = set(paths_b)

    for path_a in paths_a:
        for path_b in remaining_paths_b:
            if path_a.stem == path_b.stem:
                yield path_a, path_b
                remaining_paths_b.remove(path_b)
                break
