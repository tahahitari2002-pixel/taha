"""Shared helpers for synthetic test images."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.config import CoordinateConfig, VisionConfig  # noqa: E402

WIDTH, HEIGHT = 640, 480


def hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    """Convert one OpenCV-scale HSV colour to a BGR tuple."""
    px = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in px)


#: A saturated pink well inside the default HSV range (140-180, 150+, 50+).
PINK_BGR = hsv_to_bgr(165, 220, 230)
#: Grey-green background, far outside the pink range.
BACKGROUND_BGR = (90, 110, 90)


def blank_frame(colour: tuple[int, int, int] = BACKGROUND_BGR) -> np.ndarray:
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = colour
    return frame


def frame_with_ball(cx: int, cy: int, r: int, colour=PINK_BGR) -> np.ndarray:
    frame = blank_frame()
    cv2.circle(frame, (cx, cy), r, colour, -1, lineType=cv2.LINE_AA)
    return frame


@pytest.fixture
def identity_config() -> VisionConfig:
    """Config with an identity axis matrix so platform = re-centred pixels."""
    return VisionConfig(coordinates=CoordinateConfig(axis_matrix=((1.0, 0.0), (0.0, 1.0))))
