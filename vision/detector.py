"""Ball detection: a pure, camera-independent algorithm.

Nothing in this module touches the camera, threads, or global state, so it can
be unit-tested with synthetic images (see ``tests/test_detector.py``).

Per-frame pipeline (implemented in :func:`find_ball_candidate`):

1. Light Gaussian blur (configurable, may be disabled).
2. Convert to HSV.
3. ``cv2.inRange`` with the configured bounds -> binary mask.
4. Small erode + dilate (morphological opening) on the mask.
5. Find external contours.
6. For every contour compute minimum enclosing circle, area and circularity
   ``4*pi*area / perimeter**2``.
7. Keep contours whose radius is within ``[min_radius_px, max_radius_px]``
   **and** whose circularity is ``>= min_circularity``.
8. Pick the surviving contour with the largest area.

Tracking across frames (implemented in :func:`update_track`) is also a pure
function: the previous :class:`DetectionResult` is passed in explicitly and a
new one is returned. If no candidate is found the last good position is
re-reported with status ``STALE`` for up to ``max_stale_frames`` frames; after
that the status is ``LOST`` and *no* coordinate is reported, so a downstream
controller can never mistake an old position for a live one.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from .config import DetectionConfig, TrackingConfig, VisionConfig
from .coordinates import pixel_to_platform_coords


class TrackingStatus(str, Enum):
    """Quality of the position contained in a :class:`DetectionResult`."""

    #: Ball detected in this very frame; coordinates are fresh.
    DETECTED = "detected"
    #: Ball not seen this frame; coordinates are the last good ones and
    #: ``stale_frames`` tells how many frames old they are.
    STALE = "stale"
    #: Ball missing for too long (or never seen); coordinates are ``None``.
    LOST = "lost"


@dataclass(frozen=True)
class BallCandidate:
    """A single contour that passed all filters (pixel coordinates)."""

    pixel_x: float
    pixel_y: float
    radius: float
    area: float
    circularity: float


@dataclass(frozen=True)
class DetectionResult:
    """Output of the vision subsystem for one frame.

    Coordinates ``x``/``y`` are in the platform frame (pixels, origin at the
    platform's mechanical centre; see :mod:`vision.coordinates`). They are
    floats because the enclosing-circle centre is sub-pixel accurate, which
    gives the controller a smoother derivative term than integer rounding.

    Attributes:
        status: DETECTED / STALE / LOST, see :class:`TrackingStatus`.
        x, y: Platform coordinates, ``None`` when ``status`` is LOST.
        pixel_x, pixel_y: Same position in raw pixel coordinates (top-left
            origin), used for drawing overlays. ``None`` when LOST.
        area: Contour area in px^2 (0.0 when LOST).
        radius: Minimum enclosing circle radius in px (0.0 when LOST).
        circularity: Circularity of the chosen contour (0.0 when LOST).
        stale_frames: Consecutive frames without a real detection
            (0 when DETECTED).
        frame_id: Monotonic frame counter assigned by the caller.
        timestamp: Capture time in seconds (``time.monotonic`` clock).
    """

    status: TrackingStatus
    x: Optional[float]
    y: Optional[float]
    pixel_x: Optional[float]
    pixel_y: Optional[float]
    area: float
    radius: float
    circularity: float
    stale_frames: int
    frame_id: int
    timestamp: float

    @property
    def found(self) -> bool:
        """True only if the ball was actually detected in this frame."""
        return self.status is TrackingStatus.DETECTED

    @property
    def has_position(self) -> bool:
        """True if ``x``/``y`` hold a (fresh or still-trusted stale) position."""
        return self.status is not TrackingStatus.LOST

    def as_lost(self) -> "DetectionResult":
        """Return a copy of this result downgraded to LOST (no coordinates)."""
        return lost_result(self.stale_frames, self.frame_id, self.timestamp)

    def to_dict(self) -> dict:
        """JSON-friendly representation (used by the debug server)."""
        return {
            "status": self.status.value,
            "found": self.found,
            "x": self.x,
            "y": self.y,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "area": self.area,
            "radius": self.radius,
            "circularity": self.circularity,
            "stale_frames": self.stale_frames,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }


def lost_result(stale_frames: int, frame_id: int, timestamp: float) -> DetectionResult:
    """Build a LOST result (no coordinates)."""
    return DetectionResult(
        status=TrackingStatus.LOST,
        x=None, y=None, pixel_x=None, pixel_y=None,
        area=0.0, radius=0.0, circularity=0.0,
        stale_frames=stale_frames, frame_id=frame_id, timestamp=timestamp,
    )


# --------------------------------------------------------------------------- #
# Stage 1: colour mask                                                        #
# --------------------------------------------------------------------------- #

def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame must have shape (H, W, 3); got {frame.shape}. "
            "4-channel formats such as XRGB8888 are intentionally not supported."
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame is empty")


def build_mask(
    frame: np.ndarray,
    config: DetectionConfig,
    channel_order: str = "BGR",
) -> np.ndarray:
    """Return the cleaned binary mask (uint8, 0/255) of ball-coloured pixels.

    Args:
        frame: ``(H, W, 3)`` uint8 image.
        config: Detection parameters.
        channel_order: ``"BGR"`` (Picamera2 ``RGB888``, OpenCV default) or
            ``"RGB"``. Choosing the conversion explicitly avoids the
            channel-mismatch bug found in one reference project.
    """
    _validate_frame(frame)
    if config.blur_kernel_size > 0:
        k = config.blur_kernel_size
        frame = cv2.GaussianBlur(frame, (k, k), 0)

    if channel_order == "BGR":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif channel_order == "RGB":
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    else:
        raise ValueError(f"channel_order must be 'BGR' or 'RGB', got {channel_order!r}")

    lower = np.array(config.hsv_lower, dtype=np.uint8)
    upper = np.array(config.hsv_upper, dtype=np.uint8)
    if config.hue_wraps:
        # Hue interval crosses 180 -> 0: union of [lower_h, 180] and [0, upper_h].
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower, np.array((180, upper[1], upper[2]), dtype=np.uint8)),
            cv2.inRange(hsv, np.array((0, lower[1], lower[2]), dtype=np.uint8), upper),
        )
    else:
        mask = cv2.inRange(hsv, lower, upper)

    if config.morph_kernel_size > 0 and config.morph_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
        )
        mask = cv2.erode(mask, kernel, iterations=config.morph_iterations)
        mask = cv2.dilate(mask, kernel, iterations=config.morph_iterations)
    return mask


# --------------------------------------------------------------------------- #
# Stage 2: contour analysis and filtering                                     #
# --------------------------------------------------------------------------- #

def circularity(area: float, perimeter: float) -> float:
    """``4*pi*area / perimeter**2``; 0.0 for degenerate contours."""
    if perimeter <= 0.0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def find_ball_candidate_in_mask(
    mask: np.ndarray, config: DetectionConfig
) -> Optional[BallCandidate]:
    """Apply the contour filters to an existing mask and pick the ball.

    Returns the largest-area contour that passes both the radius and the
    circularity filters, or ``None`` if no contour qualifies.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[BallCandidate] = None
    for contour in contours:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not (config.min_radius_px <= radius <= config.max_radius_px):
            continue
        area = float(cv2.contourArea(contour))
        circ = circularity(area, float(cv2.arcLength(contour, True)))
        if circ < config.min_circularity:
            continue
        if best is None or area > best.area:
            best = BallCandidate(float(cx), float(cy), float(radius), area, circ)
    return best


def find_ball_candidate(
    frame: np.ndarray,
    config: DetectionConfig,
    channel_order: str = "BGR",
) -> Optional[BallCandidate]:
    """Run the full single-frame detection (mask + filters). Stateless."""
    return find_ball_candidate_in_mask(build_mask(frame, config, channel_order), config)


# --------------------------------------------------------------------------- #
# Stage 3: temporal tracking (stale / lost logic) and coordinate transform    #
# --------------------------------------------------------------------------- #

def update_track(
    candidate: Optional[BallCandidate],
    previous: Optional[DetectionResult],
    tracking: TrackingConfig,
    to_platform,  # Callable[[float, float], tuple[float, float]]
    frame_id: int,
    timestamp: float,
) -> DetectionResult:
    """Combine this frame's candidate with the previous result.

    Args:
        candidate: Ball found this frame, or ``None``.
        previous: Result of the previous frame, or ``None`` at start-up.
        tracking: Stale/lost policy.
        to_platform: Function mapping pixel (x, y) to platform (x, y).
        frame_id: Identifier stored in the result.
        timestamp: Capture time stored in the result.
    """
    if candidate is not None:
        x, y = to_platform(candidate.pixel_x, candidate.pixel_y)
        return DetectionResult(
            status=TrackingStatus.DETECTED,
            x=x, y=y,
            pixel_x=candidate.pixel_x, pixel_y=candidate.pixel_y,
            area=candidate.area, radius=candidate.radius,
            circularity=candidate.circularity,
            stale_frames=0, frame_id=frame_id, timestamp=timestamp,
        )

    stale = (previous.stale_frames if previous is not None else 0) + 1
    if previous is None or not previous.has_position or stale > tracking.max_stale_frames:
        return lost_result(stale, frame_id, timestamp)

    # Re-report the last good measurement, clearly flagged as stale.
    return DetectionResult(
        status=TrackingStatus.STALE,
        x=previous.x, y=previous.y,
        pixel_x=previous.pixel_x, pixel_y=previous.pixel_y,
        area=previous.area, radius=previous.radius,
        circularity=previous.circularity,
        stale_frames=stale, frame_id=frame_id, timestamp=timestamp,
    )


def detect_ball(
    frame: np.ndarray,
    config: VisionConfig,
    previous: Optional[DetectionResult] = None,
    *,
    frame_id: Optional[int] = None,
    timestamp: Optional[float] = None,
) -> DetectionResult:
    """Detect the ball in ``frame`` and return platform coordinates.

    This is the main entry point. It is a pure function of its arguments:
    feed back the returned result as ``previous`` on the next call to get
    stale-frame counting.

    Args:
        frame: ``(H, W, 3)`` uint8 image in ``config.camera.frame_channel_order``.
        config: Full vision configuration.
        previous: Result returned for the previous frame (``None`` at start).
        frame_id: Optional frame counter; defaults to ``previous.frame_id + 1``.
        timestamp: Optional capture time; defaults to ``time.monotonic()``.
    """
    if frame_id is None:
        frame_id = previous.frame_id + 1 if previous is not None else 0
    if timestamp is None:
        timestamp = time.monotonic()

    candidate = find_ball_candidate(frame, config.detection, config.camera.frame_channel_order)
    frame_size = (frame.shape[1], frame.shape[0])  # derived from the frame itself

    def to_platform(px: float, py: float) -> tuple[float, float]:
        return pixel_to_platform_coords(px, py, frame_size, config.coordinates)

    return update_track(candidate, previous, config.tracking, to_platform, frame_id, timestamp)


class BallTracker:
    """Small stateful convenience wrapper around :func:`detect_ball`.

    Not thread-safe by design: it is meant to be owned by a single thread
    (the pipeline's processing thread, or a test / offline script).
    """

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._last: Optional[DetectionResult] = None

    @property
    def last_result(self) -> Optional[DetectionResult]:
        return self._last

    def reset(self) -> None:
        """Forget the track (next miss will be reported as LOST)."""
        self._last = None

    def update(
        self,
        frame: np.ndarray,
        *,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> DetectionResult:
        self._last = detect_ball(
            frame, self._config, self._last, frame_id=frame_id, timestamp=timestamp
        )
        return self._last
