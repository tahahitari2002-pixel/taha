"""Tests for the threaded pipeline using a fake frame source (no camera)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from conftest import blank_frame, frame_with_ball
from vision.camera import CameraError, Frame
from vision.config import PipelineConfig, TrackingConfig, VisionConfig
from vision.detector import TrackingStatus
from vision.mjpeg_debug_server import draw_overlay
from vision.pipeline import VisionPipeline


class FakeSource:
    """Frame source that serves a configurable image at ~200 fps."""

    def __init__(self, image: np.ndarray, period_s: float = 0.005) -> None:
        self.image = image
        self.period_s = period_s
        self.started = False
        self.stopped = False
        self.fail = False
        self.freeze = threading.Event()
        self._seq = 0

    @property
    def frame_shape(self):
        return self.image.shape

    def start(self) -> None:
        self.started = True

    def capture_frame(self) -> Frame:
        time.sleep(self.period_s)
        while self.freeze.is_set():
            time.sleep(0.01)
        if self.fail:
            raise CameraError("simulated failure")
        self._seq += 1
        img = self.image.copy()
        img.flags.writeable = False
        return Frame(img, time.monotonic(), self._seq)

    def stop(self) -> None:
        self.stopped = True


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def make_config(**pipeline_kwargs) -> VisionConfig:
    return VisionConfig(
        tracking=TrackingConfig(max_stale_frames=3),
        pipeline=PipelineConfig(**pipeline_kwargs),
    )


def test_pipeline_reports_detection_and_then_lost():
    source = FakeSource(frame_with_ball(320 + 40, 240, 20))
    with VisionPipeline(make_config(), source) as vision:
        assert wait_until(lambda: vision.get_latest_detection().found)
        det = vision.get_latest_detection()
        assert det.x == pytest.approx(40, abs=1.0)

        source.image = blank_frame()
        assert wait_until(lambda: vision.get_latest_detection().status is TrackingStatus.LOST)
        stats = vision.stats()
        assert stats.healthy and stats.frames_processed > 0 and stats.capture_fps > 0
    assert source.started and source.stopped
    assert not vision.is_running
    assert vision.get_latest_detection().status is TrackingStatus.LOST


def test_frozen_camera_turns_result_lost():
    source = FakeSource(frame_with_ball(320, 240, 20))
    with VisionPipeline(make_config(result_timeout_s=0.1), source) as vision:
        assert wait_until(lambda: vision.get_latest_detection().found)
        source.freeze.set()
        try:
            assert wait_until(lambda: vision.get_latest_detection().status is TrackingStatus.LOST)
        finally:
            source.freeze.clear()


def test_repeated_capture_errors_mark_pipeline_unhealthy():
    source = FakeSource(blank_frame(), period_s=0.0)
    source.fail = True
    cfg = make_config(capture_error_backoff_s=0.001, max_consecutive_capture_errors=5)
    with VisionPipeline(cfg, source) as vision:
        assert wait_until(lambda: not vision.stats().healthy)
        stats = vision.stats()
        assert stats.capture_errors == 5
        assert "simulated failure" in stats.last_error


def test_concurrent_readers_and_listener():
    source = FakeSource(frame_with_ball(300, 200, 25))
    received = []
    errors = []
    with VisionPipeline(make_config(), source) as vision:
        vision.add_listener(received.append)
        vision.add_listener(lambda r: 1 / 0)  # faulty listener must not break anything

        def reader():
            try:
                for _ in range(200):
                    det = vision.get_latest_detection()
                    assert det.status in TrackingStatus
            except AssertionError as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert wait_until(lambda: len(received) >= 5)
        first = vision.get_latest_detection().frame_id
        newer = vision.wait_for_detection(first, timeout_s=1.0)
        assert newer is not None and newer.frame_id > first
    assert not errors
    ids = [r.frame_id for r in received]
    assert ids == sorted(ids)


def test_draw_overlay_does_not_modify_input():
    cfg = VisionConfig()
    frame = frame_with_ball(320, 240, 20)
    frame.flags.writeable = False
    from vision.detector import detect_ball

    for img in (frame, blank_frame()):
        result = detect_ball(img, cfg)
        out = draw_overlay(img, result, cfg, fps=99.0)
        assert out.shape == img.shape
        assert out is not img
