"""
DELPHI command-line interface.
"""

from collections import deque
from pathlib import Path

from typer import Argument, Typer

from delphi.lib.annotate import annotate_images
from delphi.lib.detect import detect_points
from delphi.lib.log import LogLevel, set_stream_handler_level
from delphi.lib.train import train_model

delphi = Typer(
    name="delphi",
    help="DELPHI: DEtection of Laser Points in Huge image collections using Iterative learning",
)


@delphi.callback()
def global_options(
    verbose: bool = False,
):
    """
    Global options.

    Args:
        verbose: Whether to print verbose output.
    """
    set_stream_handler_level(LogLevel.DEBUG if verbose else LogLevel.INFO)


@delphi.command()
def train(
    image_dir: Path = Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing images.",
    ),
    annotation_dir: Path = Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing annotations.",
    ),
    output_dir: Path = Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        help="Directory to save model to.",
    ),
    delta_1: int = Argument(25, help="Delta 1."),
    delta_2: int = Argument(3, help="Delta 2."),
):
    """
    Train a model.
    """
    train_model(image_dir, annotation_dir, output_dir, delta_1, delta_2)


@delphi.command()
def annotate(
    image_dir: Path = Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing images.",
    ),
    annotation_dir: Path = Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        help="Directory to save annotations to.",
    ),
    n: int = Argument(-1, help="Number of images to annotate. -1 for all."),
):
    """
    Annotate images.
    """
    annotate_images(image_dir, annotation_dir, n)


@delphi.command()
def detect(
    image_dir: Path = Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing images.",
    ),
    annotation_dir: Path = Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing annotations.",
    ),
    model_path: Path = Argument(
        ..., exists=True, file_okay=True, dir_okay=False, help="Path to model."
    ),
):
    """
    Detect points in images.
    """
    deque(detect_points(image_dir, annotation_dir, model_path), maxlen=0)


if __name__ == "__main__":
    delphi()
