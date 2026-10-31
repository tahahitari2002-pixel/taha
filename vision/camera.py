"""Picamera2 wrapper for the Raspberry Pi Camera Module 3 (IMX708).

The class :class:`Camera` owns the full camera lifecycle::

    with Camera(config.camera) as cam:
        frame = cam.capture_frame()      # Frame(image, timestamp, sequence)

``picamera2`` is imported lazily inside :meth:`Camera.start`, so the rest of
the package (detector, tests, debug drawing) can be imported on a development
machine that has no camera stack installed.

Channel order (bug #4 from the reference audit): we request ``RGB888``.
In libcamera/Picamera2 naming this is a *3-channel* format whose bytes are
stored as **B, G, R** (Picamera2 manual, "Image formats": "RGB888 - 24 bits
per pixel, ordered [B, G, R]"). The resulting numpy array is therefore
directly usable with OpenCV's ``COLOR_BGR2HSV``. This is recorded in
``CameraConfig.frame_channel_order = "BGR"`` and must be confirmed on the
real hardware with ``python -m vision.tools.check_channel_order`` (point the
camera at a red object; the script reports which channel is brightest).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened, configured or read."""


@dataclass(frozen=True)
class Frame:
    """A captured image plus timing metadata.

    Attributes:
        image: ``(H, W, 3)`` uint8 array, marked read-only so that threads
            sharing it cannot corrupt each other's view of the frame.
        timestamp: Capture time on the ``time.monotonic()`` clock (seconds).
        sequence: Monotonically increasing frame number from the source.
    """

    image: np.ndarray
    timestamp: float
    sequence: int


class FrameSource(Protocol):
    """Anything the pipeline can pull frames from (camera, video, test fake)."""

    @property
    def frame_shape(self) -> tuple[int, int, int]: ...

    def start(self) -> None: ...

    def capture_frame(self) -> Frame: ...

    def stop(self) -> None: ...


class Camera:
    """Picamera2-backed :class:`FrameSource`.

    Args:
        config: Camera settings; ``config.main_size`` is the single source of
            truth for the frame shape.
        camera_index: Index passed to ``Picamera2`` (0 for the only camera).
    """

    def __init__(self, config: CameraConfig, camera_index: int = 0) -> None:
        self._config = config
        self._camera_index = camera_index
        self._picam2: Any = None
        self._sequence = 0
        self._shape_verified = False

    # ----------------------------------------------------------------- props
    @property
    def config(self) -> CameraConfig:
        return self._config

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Expected ``(H, W, 3)`` shape, derived from the configuration."""
        return self._config.frame_shape

    @property
    def is_running(self) -> bool:
        return self._picam2 is not None

    # ------------------------------------------------------------- lifecycle
    def _build_controls(self) -> dict[str, Any]:
        c = self._config
        controls: dict[str, Any] = {
            "AfMode": c.af_mode,            # manual focus: no hunting while tracking
            "LensPosition": c.lens_position,
            "AwbEnable": c.awb_enable,      # AWB off: HSV thresholds need stable colour
            "ColourGains": tuple(c.colour_gains),
        }
        if c.exposure_time_us is not None:
            controls["AeEnable"] = False
            controls["ExposureTime"] = c.exposure_time_us
        if c.analogue_gain is not None:
            controls["AeEnable"] = False
            controls["AnalogueGain"] = c.analogue_gain
        return controls

    def start(self) -> None:
        """Open, configure and start the camera.

        Raises:
            CameraError: if ``picamera2`` is missing or the camera fails.
        """
        if self._picam2 is not None:
            return
        try:
            from picamera2 import Picamera2  # lazy: only needed on the Pi
        except ImportError as exc:
            raise CameraError(
                "picamera2 is not installed. On Raspberry Pi OS run "
                "'sudo apt install python3-picamera2' (and create any venv "
                "with --system-site-packages)."
            ) from exc

        c = self._config
        picam2 = None
        try:
            picam2 = Picamera2(self._camera_index)
            video_config = picam2.create_video_configuration(
                main={"size": tuple(c.main_size), "format": c.main_format},
                raw={"size": tuple(c.raw_size)},
                controls={"FrameDurationLimits": tuple(c.frame_duration_limits_us)},
            )
            picam2.configure(video_config)
            picam2.set_controls(self._build_controls())
            picam2.start()
        except Exception as exc:  # picamera2/libcamera raise many types
            if picam2 is not None:
                self._safe_close(picam2)
            raise CameraError(f"failed to start camera {self._camera_index}: {exc}") from exc

        self._picam2 = picam2
        self._sequence = 0
        self._shape_verified = False
        log.info(
            "camera started: main=%sx%s %s, raw=%sx%s, target %.1f fps, lens=%.2f dpt",
            *c.main_size, c.main_format, *c.raw_size, c.target_fps, c.lens_position,
        )
        for _ in range(c.warmup_frames):
            self.capture_frame()

    def capture_frame(self) -> Frame:
        """Block until the next frame is available and return it.

        Raises:
            CameraError: if the camera is not started, the capture fails, or
                the frame does not have the configured shape.
        """
        if self._picam2 is None:
            raise CameraError("camera is not started")
        try:
            request = self._picam2.capture_request()
        except Exception as exc:
            raise CameraError(f"capture failed: {exc}") from exc
        try:
            image = request.make_array("main")  # a copy; safe after release()
            metadata = request.get_metadata()
        except Exception as exc:
            raise CameraError(f"could not read captured frame: {exc}") from exc
        finally:
            request.release()  # always return the buffer to libcamera

        self._verify_shape(image)
        image.flags.writeable = False
        self._sequence += 1
        return Frame(image=image, timestamp=self._timestamp_from(metadata), sequence=self._sequence)

    def stop(self) -> None:
        """Stop and close the camera. Safe to call multiple times."""
        picam2, self._picam2 = self._picam2, None
        if picam2 is not None:
            self._safe_close(picam2)
            log.info("camera stopped")

    # --------------------------------------------------------------- helpers
    def _verify_shape(self, image: np.ndarray) -> None:
        """Fail loudly once if the real frame disagrees with the config."""
        if self._shape_verified:
            return
        if image.shape != self.frame_shape or image.dtype != np.uint8:
            raise CameraError(
                f"camera produced {image.shape} {image.dtype}, expected "
                f"{self.frame_shape} uint8 - check CameraConfig.main_size/main_format"
            )
        self._shape_verified = True

    @staticmethod
    def _timestamp_from(metadata: Optional[dict]) -> float:
        """Convert libcamera's SensorTimestamp (ns, CLOCK_BOOTTIME) to monotonic.

        On Linux ``time.monotonic`` does not advance during suspend whereas
        CLOCK_BOOTTIME does; on a Pi that never suspends they are equivalent
        up to a constant, so we simply re-base onto the monotonic clock using
        the current offset. Falls back to "now" if metadata is missing.
        """
        now_mono = time.monotonic()
        if not metadata or "SensorTimestamp" not in metadata:
            return now_mono
        try:
            sensor_s = metadata["SensorTimestamp"] / 1e9
            boot_now = time.clock_gettime(time.CLOCK_BOOTTIME)
        except (AttributeError, TypeError, OSError):
            return now_mono
        return now_mono - (boot_now - sensor_s)

    @staticmethod
    def _safe_close(picam2: Any) -> None:
        for action in ("stop", "close"):
            try:
                getattr(picam2, action)()
            except Exception as exc:  # best-effort cleanup must never raise
                log.warning("camera %s() failed during cleanup: %s", action, exc)

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
