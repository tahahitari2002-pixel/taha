"""Pixel -> platform coordinate transform.

This module is deliberately the **only** place where axis conventions are
handled. Nothing else in the vision code flips, swaps or rotates axes.

Pixel frame (OpenCV / numpy):
    origin at the top-left corner, +x to the right, +y downwards.

Platform frame (what the controller consumes):
    origin at the platform's mechanical centre, axes aligned with the
    platform's physical X/Y axes (defined by the leg layout), units = pixels.

The transform is applied in three explicit steps:

1. **Re-centre**: subtract half the frame size so (0, 0) is the image centre.
2. **Axis transform**: multiply by ``CoordinateConfig.axis_matrix`` to align
   the camera's axes with the platform's (rotation and/or flips).
3. **Calibration offset**: subtract ``CoordinateConfig.center_offset_px`` so
   (0, 0) is the mechanical centre rather than the optical centre.

CALIBRATION PENDING: once the camera is mounted, roll the ball along the
platform's +X axis and watch the debug server; adjust ``axis_matrix`` until the
reported X grows and Y stays ~0, then repeat for +Y. Normally only the config
values need to change; if a non-linear correction (e.g. lens distortion or
perspective) is ever needed, replace the body of
:func:`pixel_to_platform_coords` and nothing else.
"""

from __future__ import annotations

from typing import Tuple

from .config import CoordinateConfig, Size


def pixel_to_platform_coords(
    px: float,
    py: float,
    frame_size: Size,
    config: CoordinateConfig,
) -> Tuple[float, float]:
    """Convert a pixel position to platform-centred coordinates.

    Args:
        px: Pixel x (column), origin top-left.
        py: Pixel y (row), origin top-left, increasing downwards.
        frame_size: ``(width, height)`` of the frame the pixel belongs to.
        config: Axis matrix and calibration offset.

    Returns:
        ``(x, y)`` in the platform frame, in pixels.
    """
    width, height = frame_size
    # 1. Re-centre on the optical centre of the image.
    cx = px - width / 2.0
    cy = py - height / 2.0
    # 2. Rotate/flip into the platform's axes.
    (a, b), (c, d) = config.axis_matrix
    x = a * cx + b * cy
    y = c * cx + d * cy
    # 3. Shift so that (0, 0) is the platform's mechanical centre.
    off_x, off_y = config.center_offset_px
    return (x - off_x, y - off_y)


def platform_to_pixel_coords(
    x: float,
    y: float,
    frame_size: Size,
    config: CoordinateConfig,
) -> Tuple[float, float]:
    """Exact inverse of :func:`pixel_to_platform_coords`.

    Used by the debug overlay to draw the platform's mechanical centre and
    axes on the image, which is how the calibration is checked visually.
    """
    width, height = frame_size
    off_x, off_y = config.center_offset_px
    x, y = x + off_x, y + off_y
    (a, b), (c, d) = config.axis_matrix
    det = a * d - b * c  # validated non-zero in CoordinateConfig
    cx = (d * x - b * y) / det
    cy = (-c * x + a * y) / det
    return (cx + width / 2.0, cy + height / 2.0)
