"""Central configuration for the Vision subsystem.

Every tunable number used by the vision code lives in this file, grouped into
small frozen dataclasses:

* :class:`CameraConfig`      - Picamera2 / IMX708 settings
* :class:`DetectionConfig`   - colour threshold, cleanup and candidate filters
* :class:`TrackingConfig`    - how long a stale position is trusted
* :class:`CoordinateConfig`  - pixel -> platform axis transform and offset
* :class:`PipelineConfig`    - threading cadence and timeouts
* :class:`VisionConfig`      - the aggregate passed around the code

There are two ways to retune values without touching any logic code:

1. Edit the defaults below (simplest during development).
2. Write a JSON file with only the keys you want to override and load it with
   :meth:`VisionConfig.from_json`; the debug server accepts ``--config``.
   Example::

       {
         "detection": {"hsv_lower": [145, 120, 60], "min_radius_px": 8},
         "camera":    {"lens_position": 4.2}
       }

All dataclasses validate themselves in ``__post_init__`` so a typo such as a
negative radius or an inverted range fails immediately and loudly at startup
instead of silently producing bad detections at 100 fps.

Items marked ``CALIBRATION PENDING`` are placeholders that cannot be
determined until the hardware is finalised. See README.md, "Open items".
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Tuple, Type, TypeVar

HSVTriple = Tuple[int, int, int]
Size = Tuple[int, int]  # (width, height), the Picamera2 convention
Matrix2x2 = Tuple[Tuple[float, float], Tuple[float, float]]

# --------------------------------------------------------------------------- #
# Named constants - the values most likely to be edited by hand.              #
# --------------------------------------------------------------------------- #

#: Lower HSV bound for the ball (OpenCV scale: H 0-180, S 0-255, V 0-255).
#: CALIBRATION PENDING: final ball colour is not purchased yet; this default
#: comes from the reference project that used a bright pink/fluorescent ball.
BALL_HSV_LOWER: HSVTriple = (140, 150, 50)
#: Upper HSV bound for the ball. See :data:`BALL_HSV_LOWER`.
BALL_HSV_UPPER: HSVTriple = (180, 255, 255)

#: Manual focus position in dioptres (1 / distance_in_metres).
#: 5.0 dioptres ~= 20 cm. CALIBRATION PENDING: depends on the final camera
#: mount height chosen by the mechanical sub-team.
LENS_POSITION_DIOPTRES: float = 5.0

#: Fixed white-balance (red_gain, blue_gain). AWB is disabled on purpose
#: because drifting colour breaks HSV thresholding. CALIBRATION PENDING: needs
#: a one-time calibration against the real room lighting (see README).
COLOUR_GAINS: Tuple[float, float] = (1.4, 1.5)


def _check(condition: bool, message: str) -> None:
    """Raise :class:`ValueError` with ``message`` if ``condition`` is false."""
    if not condition:
        raise ValueError(message)


def _is_odd_or_zero(value: int) -> bool:
    return value == 0 or (value > 0 and value % 2 == 1)


@dataclass(frozen=True)
class CameraConfig:
    """Picamera2 settings for the Raspberry Pi Camera Module 3 (IMX708).

    The baseline was verified on the target hardware:

    * sensor mode 1536x864 @ 120 fps (the only IMX708 mode >= 60 fps),
    * ISP output 640x480 ``RGB888`` for processing,
    * 10 ms frame duration (~100 fps, safely below the 120 fps ceiling).
    """

    #: ISP output ("main" stream) size as (width, height). This is the single
    #: source of truth for the frame shape used everywhere else.
    main_size: Size = (640, 480)
    #: Pixel format of the main stream. ``RGB888`` in libcamera/Picamera2
    #: naming yields a 3-channel array whose bytes are ordered B, G, R, i.e.
    #: exactly what OpenCV expects (see ``frame_channel_order`` below).
    main_format: str = "RGB888"
    #: Sensor ("raw") mode. Selecting it forces the fast 120 fps binned mode.
    raw_size: Size = (1536, 864)
    #: (min, max) frame duration in microseconds. 10000 us == 100 fps.
    frame_duration_limits_us: Tuple[int, int] = (10000, 10000)
    #: Channel order of the numpy array produced for ``main_format``.
    #: Picamera2 documents ``RGB888`` as "[B, G, R]" in memory, so this must be
    #: "BGR" for RGB888. Verify on hardware with
    #: ``python -m vision.tools.check_channel_order`` (see README).
    frame_channel_order: str = "BGR"

    #: libcamera AfMode. 0 == Manual: focus must NOT hunt while tracking.
    af_mode: int = 0
    #: Manual lens position in dioptres. CALIBRATION PENDING.
    lens_position: float = LENS_POSITION_DIOPTRES
    #: Auto white balance must stay off (HSV thresholds rely on stable colour).
    awb_enable: bool = False
    #: Fixed (red, blue) colour gains. CALIBRATION PENDING.
    colour_gains: Tuple[float, float] = COLOUR_GAINS

    #: Optional fixed exposure in microseconds. ``None`` keeps auto exposure.
    #: Locking exposure (and gain) makes V-channel thresholds even more stable
    #: once lighting is known; it must be <= the frame duration.
    exposure_time_us: int | None = None
    #: Optional fixed analogue gain. ``None`` keeps automatic gain.
    analogue_gain: float | None = None

    #: Number of frames to discard after start-up while AE settles.
    warmup_frames: int = 5

    def __post_init__(self) -> None:
        for name in ("main_size", "raw_size"):
            w, h = getattr(self, name)
            _check(w > 0 and h > 0, f"camera.{name} must be positive, got {(w, h)}")
        lo, hi = self.frame_duration_limits_us
        _check(0 < lo <= hi, f"camera.frame_duration_limits_us invalid: {(lo, hi)}")
        _check(self.main_format in ("RGB888", "BGR888"),
               f"camera.main_format must be RGB888 or BGR888 (3-channel), got {self.main_format!r}")
        _check(self.frame_channel_order in ("BGR", "RGB"),
               f"camera.frame_channel_order must be 'BGR' or 'RGB', got {self.frame_channel_order!r}")
        _check(self.lens_position >= 0.0, "camera.lens_position must be >= 0 dioptres")
        _check(all(g > 0 for g in self.colour_gains), "camera.colour_gains must be positive")
        if self.exposure_time_us is not None:
            _check(0 < self.exposure_time_us <= hi,
                   "camera.exposure_time_us must be > 0 and <= the max frame duration")
        if self.analogue_gain is not None:
            _check(self.analogue_gain >= 1.0, "camera.analogue_gain must be >= 1.0")
        _check(self.warmup_frames >= 0, "camera.warmup_frames must be >= 0")

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """numpy shape ``(height, width, channels)`` of a captured frame."""
        width, height = self.main_size
        return (height, width, 3)

    @property
    def target_fps(self) -> float:
        """Frame rate implied by the maximum frame duration."""
        return 1_000_000.0 / self.frame_duration_limits_us[1]


@dataclass(frozen=True)
class DetectionConfig:
    """Colour threshold, noise cleanup and candidate filters for the ball."""

    #: Inclusive HSV lower bound (OpenCV scale). If ``hsv_lower[0] >
    #: hsv_upper[0]`` the hue range is treated as wrapping through 180/0,
    #: which is handy for red-ish balls (e.g. lower H=170, upper H=10).
    hsv_lower: HSVTriple = BALL_HSV_LOWER
    #: Inclusive HSV upper bound (OpenCV scale).
    hsv_upper: HSVTriple = BALL_HSV_UPPER

    #: Gaussian blur kernel size applied before thresholding. Must be odd;
    #: 0 disables blurring. Kept small so a small ball isn't smeared away.
    blur_kernel_size: int = 5
    #: Square kernel size for the erode+dilate (morphological opening) pass on
    #: the mask. 0 disables it. Opening removes speckle and smooths edges,
    #: which matters because circularity is sensitive to jagged contours.
    morph_kernel_size: int = 3
    #: Number of erode iterations followed by the same number of dilations.
    morph_iterations: int = 1

    #: Accepted minimum-enclosing-circle radius range in pixels (inclusive).
    #: CALIBRATION PENDING: depends on final ball diameter and camera height.
    min_radius_px: float = 5.0
    max_radius_px: float = 100.0
    #: Minimum circularity 4*pi*area/perimeter^2 (1.0 = perfect circle).
    min_circularity: float = 0.6

    def __post_init__(self) -> None:
        for name in ("hsv_lower", "hsv_upper"):
            h, s, v = getattr(self, name)
            _check(0 <= h <= 180 and 0 <= s <= 255 and 0 <= v <= 255,
                   f"detection.{name} out of OpenCV HSV range: {(h, s, v)}")
        _check(self.hsv_lower[1] <= self.hsv_upper[1] and self.hsv_lower[2] <= self.hsv_upper[2],
               "detection: S and V lower bounds must be <= upper bounds")
        _check(_is_odd_or_zero(self.blur_kernel_size),
               "detection.blur_kernel_size must be 0 (off) or a positive odd number")
        _check(self.morph_kernel_size >= 0, "detection.morph_kernel_size must be >= 0")
        _check(self.morph_iterations >= 0, "detection.morph_iterations must be >= 0")
        _check(0 < self.min_radius_px <= self.max_radius_px,
               "detection: need 0 < min_radius_px <= max_radius_px")
        _check(0.0 <= self.min_circularity <= 1.0, "detection.min_circularity must be in [0, 1]")

    @property
    def hue_wraps(self) -> bool:
        """True when the hue interval wraps around the 180 -> 0 boundary."""
        return self.hsv_lower[0] > self.hsv_upper[0]


@dataclass(frozen=True)
class TrackingConfig:
    """Behaviour when the ball is not detected in a frame."""

    #: Number of consecutive missed frames for which the last good position is
    #: still reported (status STALE). On the next miss the status becomes LOST
    #: and no coordinate is reported. At 100 fps, 10 frames == 100 ms.
    max_stale_frames: int = 10

    def __post_init__(self) -> None:
        _check(self.max_stale_frames >= 0, "tracking.max_stale_frames must be >= 0")


@dataclass(frozen=True)
class CoordinateConfig:
    """Pixel -> platform coordinate transform (see ``vision.coordinates``).

    CALIBRATION PENDING: both values depend on how the camera is physically
    mounted relative to the platform legs and cannot be known yet.
    """

    #: 2x2 matrix applied to image-centred pixel coordinates (x right, y down)
    #: to obtain platform axes: ``[X, Y]^T = M @ [x, y]^T``. The default flips
    #: Y so that +Y points "up" in the image (a right-handed frame). Common
    #: alternatives once the mount is known:
    #:   identity               ((1, 0), (0, 1))
    #:   rotate 90 deg (CCW)    ((0, 1), (-1, 0)) applied after the flip, etc.
    #:   swap + negate (ref.)   ((0, -1), (-1, 0))
    axis_matrix: Matrix2x2 = ((1.0, 0.0), (0.0, -1.0))
    #: Position of the platform's *mechanical* centre expressed in platform
    #: coordinates (pixels). It is subtracted from every ball position, so the
    #: reported (0, 0) is the mechanical centre rather than the optical one.
    #: This is equivalent to adding the offset to the controller's set-point,
    #: which is what one reference project did. Default (0, 0) until measured.
    center_offset_px: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        (a, b), (c, d) = self.axis_matrix
        det = a * d - b * c
        _check(math.isfinite(det) and abs(det) > 1e-9,
               f"coordinates.axis_matrix must be invertible, got determinant {det}")


@dataclass(frozen=True)
class PipelineConfig:
    """Threading cadence and health timeouts for :mod:`vision.pipeline`."""

    #: Minimum time between two captures. The camera call already blocks until
    #: the next frame (~10 ms) but this guarantees a bounded cadence even if a
    #: frame source returns immediately (e.g. errors, test sources).
    capture_min_period_s: float = 0.005
    #: How long the processing thread waits for a new frame before re-checking
    #: its stop flag. Waiting is blocking (no CPU), never a busy loop.
    process_wait_timeout_s: float = 0.1
    #: Sleep after a capture error before retrying.
    capture_error_backoff_s: float = 0.05
    #: After this many consecutive capture errors the pipeline stops capturing
    #: and reports itself unhealthy.
    max_consecutive_capture_errors: int = 50
    #: If the newest detection is older than this, ``get_latest_detection``
    #: reports LOST: a frozen camera must never look like a still ball.
    result_timeout_s: float = 0.25
    #: Window (in frames) used for the rolling FPS estimate.
    fps_window: int = 60

    def __post_init__(self) -> None:
        _check(self.capture_min_period_s >= 0.0, "pipeline.capture_min_period_s must be >= 0")
        _check(self.process_wait_timeout_s > 0.0, "pipeline.process_wait_timeout_s must be > 0")
        _check(self.capture_error_backoff_s > 0.0, "pipeline.capture_error_backoff_s must be > 0")
        _check(self.max_consecutive_capture_errors >= 1,
               "pipeline.max_consecutive_capture_errors must be >= 1")
        _check(self.result_timeout_s > 0.0, "pipeline.result_timeout_s must be > 0")
        _check(self.fps_window >= 2, "pipeline.fps_window must be >= 2")


T = TypeVar("T")


def _build(cls: Type[T], overrides: Mapping[str, Any]) -> T:
    """Instantiate dataclass ``cls`` from a (possibly partial) mapping.

    Lists coming from JSON are converted to tuples (recursively) so the frozen
    dataclasses stay hashable and compare equal to their defaults. Unknown
    keys raise ``ValueError`` so that typos in a config file are not ignored.
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(overrides) - known
    _check(not unknown, f"unknown {cls.__name__} keys: {sorted(unknown)}")

    def to_tuple(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(to_tuple(v) for v in value)
        return value

    return cls(**{k: to_tuple(v) for k, v in overrides.items()})


@dataclass(frozen=True)
class VisionConfig:
    """Aggregate configuration for the whole vision subsystem."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    coordinates: CoordinateConfig = field(default_factory=CoordinateConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisionConfig":
        """Build a config from a nested mapping of *overrides*.

        Any section or key that is omitted keeps its default value.
        """
        sections = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(sections)
        _check(not unknown, f"unknown config sections: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name, section in data.items():
            _check(isinstance(section, Mapping), f"config section {name!r} must be an object")
            section_cls = type(getattr(cls(), name))
            kwargs[name] = _build(section_cls, section)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "VisionConfig":
        """Load overrides from a JSON file (see module docstring)."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _check(isinstance(data, dict), f"{path}: top-level JSON value must be an object")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Return the full configuration as plain Python types (JSON-safe)."""
        return asdict(self)

    def with_overrides(self, **sections: Any) -> "VisionConfig":
        """Return a copy with whole sections replaced, e.g.
        ``cfg.with_overrides(tracking=TrackingConfig(max_stale_frames=5))``."""
        for value in sections.values():
            _check(is_dataclass(value), "with_overrides expects config dataclass instances")
        return replace(self, **sections)


DEFAULT_CONFIG = VisionConfig()
