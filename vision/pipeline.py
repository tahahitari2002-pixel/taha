"""Threaded capture + detection pipeline.

Architecture::

    ┌──────────────┐  latest frame   ┌──────────────────┐  latest result
    │ capture      │ ──(Condition)──▶│ detection        │ ──(Lock)──▶ get_latest_detection()
    │ thread       │  1-slot buffer  │ thread           │             listeners (callbacks)
    └──────────────┘                 └──────────────────┘

* The **capture thread** blocks on the camera and writes each new frame into a
  single-slot buffer protected by a ``threading.Condition`` (which owns a
  lock). Old unprocessed frames are overwritten: the controller always wants
  the freshest position, never a backlog. Overwritten frames are counted as
  ``frames_dropped``.
* The **detection thread** waits on the condition (a blocking wait with a
  timeout - zero CPU while idle, never a busy loop), runs
  :func:`vision.detector.detect_ball`, and publishes the result under a lock.
* Any thread may call :meth:`VisionPipeline.get_latest_detection` at any
  time. If the newest result is older than ``result_timeout_s`` (camera
  frozen, thread died) it is reported as LOST instead of a stale coordinate.

This addresses bugs #1 (unsynchronised shared state), #2 (unthrottled loop on
failure) and #6 (misleading thread names) from the reference audit.

Typical use by a future PID loop::

    with VisionPipeline(config) as vision:
        while running:
            det = vision.get_latest_detection()
            if det.has_position:
                tilt = pid.update(det.x, det.y, det.timestamp)
            else:
                tilt = level_platform()
            ...
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

from .camera import Camera, Frame, FrameSource
from .config import VisionConfig
from .detector import DetectionResult, TrackingStatus, detect_ball, lost_result

log = logging.getLogger(__name__)

DetectionListener = Callable[[DetectionResult], None]


class FpsMeter:
    """Rolling-window frame-rate estimate. Not thread-safe (one owner each)."""

    def __init__(self, window: int) -> None:
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self, t: Optional[float] = None) -> None:
        self._times.append(time.monotonic() if t is None else t)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


@dataclass(frozen=True)
class PipelineStats:
    """Snapshot of pipeline health counters."""

    running: bool
    healthy: bool
    capture_fps: float
    detection_fps: float
    frames_captured: int
    frames_processed: int
    frames_dropped: int
    capture_errors: int
    detection_errors: int
    last_error: Optional[str]

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class VisionPipeline:
    """Owns a frame source and runs capture and detection in background threads.

    Args:
        config: Full vision configuration.
        source: Frame source; defaults to a :class:`~vision.camera.Camera`
            built from ``config.camera``. Inject a fake source for tests or
            offline replay.
    """

    def __init__(self, config: VisionConfig, source: Optional[FrameSource] = None) -> None:
        self._config = config
        self._source: FrameSource = source if source is not None else Camera(config.camera)

        # Frame hand-off (capture -> detection). Condition wraps its own lock.
        self._frame_cond = threading.Condition()
        self._pending_frame: Optional[Frame] = None

        # Result hand-off (detection -> any reader).
        self._result_cond = threading.Condition()
        self._latest_result: DetectionResult = lost_result(0, -1, time.monotonic())
        self._latest_result_frame: Optional[np.ndarray] = None

        self._listeners: List[DetectionListener] = []
        self._listeners_lock = threading.Lock()

        # Counters: each written by exactly one thread, read under _stats_lock.
        self._stats_lock = threading.Lock()
        self._frames_captured = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._capture_errors = 0
        self._detection_errors = 0
        self._last_error: Optional[str] = None
        self._capture_failed = False
        self._capture_fps = FpsMeter(config.pipeline.fps_window)
        self._detection_fps = FpsMeter(config.pipeline.fps_window)

        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lifecycle_lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    @property
    def config(self) -> VisionConfig:
        return self._config

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        return self._source.frame_shape

    @property
    def is_running(self) -> bool:
        return bool(self._threads) and not self._stop_event.is_set()

    def start(self) -> None:
        """Start the frame source and both worker threads.

        Raises:
            CameraError: if the frame source fails to start.
        """
        with self._lifecycle_lock:
            if self._threads:
                return
            self._source.start()  # raises CameraError on failure
            self._stop_event.clear()
            self._capture_failed = False
            self._threads = [
                threading.Thread(target=self._capture_loop, name="vision-capture", daemon=True),
                threading.Thread(target=self._detection_loop, name="vision-detect", daemon=True),
            ]
            for t in self._threads:
                t.start()
            log.info("vision pipeline started")

    def stop(self, join_timeout_s: float = 2.0) -> None:
        """Stop worker threads and release the frame source. Idempotent."""
        with self._lifecycle_lock:
            if not self._threads:
                return
            self._stop_event.set()
            with self._frame_cond:
                self._frame_cond.notify_all()
            with self._result_cond:
                self._result_cond.notify_all()
            for t in self._threads:
                t.join(timeout=join_timeout_s)
                if t.is_alive():
                    log.warning("thread %s did not stop within %.1fs", t.name, join_timeout_s)
            self._threads = []
            try:
                self._source.stop()
            except Exception as exc:  # cleanup must not mask the caller's state
                log.warning("frame source stop failed: %s", exc)
            log.info("vision pipeline stopped")

    def __enter__(self) -> "VisionPipeline":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ------------------------------------------------------------ public API
    def get_latest_detection(self) -> DetectionResult:
        """Return the newest :class:`DetectionResult`. Thread-safe, non-blocking.

        If the newest result is older than ``pipeline.result_timeout_s`` (or
        the pipeline is not running / capture has failed) a LOST result is
        returned so that a frozen camera can never masquerade as a ball that
        is standing still.
        """
        with self._result_cond:
            result = self._latest_result
        age = time.monotonic() - result.timestamp
        if (not self.is_running or self._capture_failed
                or age > self._config.pipeline.result_timeout_s):
            return result if result.status is TrackingStatus.LOST else result.as_lost()
        return result

    def wait_for_detection(
        self, after_frame_id: int, timeout_s: float
    ) -> Optional[DetectionResult]:
        """Block until a result with ``frame_id > after_frame_id`` exists.

        Returns that result, or ``None`` on timeout / shutdown. Useful for a
        consumer that wants to run exactly once per processed frame.
        """
        deadline = time.monotonic() + timeout_s
        with self._result_cond:
            while self._latest_result.frame_id <= after_frame_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop_event.is_set():
                    return None
                self._result_cond.wait(remaining)
            return self._latest_result

    def get_latest_frame_and_detection(self) -> Tuple[Optional[np.ndarray], DetectionResult]:
        """Return the frame the newest result was computed on, plus that result.

        The frame is read-only and shared; copy it before drawing on it.
        Intended for debugging/visualisation only.
        """
        with self._result_cond:
            return self._latest_result_frame, self._latest_result

    def add_listener(self, callback: DetectionListener) -> None:
        """Register ``callback(result)`` to be called for every new result.

        Callbacks run on the detection thread: keep them short and
        non-blocking. Exceptions are logged and do not stop the pipeline.
        """
        with self._listeners_lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: DetectionListener) -> None:
        with self._listeners_lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def stats(self) -> PipelineStats:
        """Snapshot of throughput and error counters."""
        with self._stats_lock:
            return PipelineStats(
                running=self.is_running,
                healthy=self.is_running and not self._capture_failed,
                capture_fps=self._capture_fps.fps,
                detection_fps=self._detection_fps.fps,
                frames_captured=self._frames_captured,
                frames_processed=self._frames_processed,
                frames_dropped=self._frames_dropped,
                capture_errors=self._capture_errors,
                detection_errors=self._detection_errors,
                last_error=self._last_error,
            )

    # -------------------------------------------------------------- threads
    def _record_error(self, counter: str, message: str) -> None:
        with self._stats_lock:
            setattr(self, counter, getattr(self, counter) + 1)
            self._last_error = message

    def _capture_loop(self) -> None:
        """Capture thread: camera -> single-slot frame buffer."""
        cfg = self._config.pipeline
        consecutive_errors = 0
        next_allowed = time.monotonic()
        while not self._stop_event.is_set():
            # Bounded cadence regardless of outcome (bug #2): never spin.
            delay = next_allowed - time.monotonic()
            if delay > 0 and self._stop_event.wait(delay):
                break
            next_allowed = time.monotonic() + cfg.capture_min_period_s

            try:
                frame = self._source.capture_frame()
            except Exception as exc:  # CameraError, or anything a custom source raises
                consecutive_errors += 1
                self._record_error("_capture_errors", f"capture: {exc}")
                log.warning("capture error %d/%d: %s", consecutive_errors,
                            cfg.max_consecutive_capture_errors, exc)
                if consecutive_errors >= cfg.max_consecutive_capture_errors:
                    log.error("too many consecutive capture errors; capture stopped")
                    self._capture_failed = True
                    return
                self._stop_event.wait(cfg.capture_error_backoff_s)
                continue
            consecutive_errors = 0

            with self._frame_cond:
                dropped = self._pending_frame is not None
                self._pending_frame = frame
                self._frame_cond.notify()
            with self._stats_lock:
                self._frames_captured += 1
                self._frames_dropped += int(dropped)
                self._capture_fps.tick(frame.timestamp)

    def _detection_loop(self) -> None:
        """Detection thread: frame buffer -> detect_ball -> published result."""
        cfg = self._config
        previous: Optional[DetectionResult] = None  # owned by this thread only
        while not self._stop_event.is_set():
            with self._frame_cond:
                # Blocking wait with timeout so the stop flag is re-checked.
                if self._pending_frame is None:
                    self._frame_cond.wait(cfg.pipeline.process_wait_timeout_s)
                frame, self._pending_frame = self._pending_frame, None
            if frame is None:
                continue

            try:
                result = detect_ball(
                    frame.image, cfg, previous,
                    frame_id=frame.sequence, timestamp=frame.timestamp,
                )
            except (ValueError, TypeError) as exc:  # e.g. malformed frame
                self._record_error("_detection_errors", f"detection: {exc}")
                log.warning("detection failed on frame %d: %s", frame.sequence, exc)
                continue
            previous = result

            with self._result_cond:
                self._latest_result = result
                self._latest_result_frame = frame.image
                self._result_cond.notify_all()
            with self._stats_lock:
                self._frames_processed += 1
                self._detection_fps.tick()

            with self._listeners_lock:
                listeners = list(self._listeners)
            for callback in listeners:
                try:
                    callback(result)
                except Exception:  # a buggy consumer must not kill vision
                    log.exception("detection listener %r raised", callback)
