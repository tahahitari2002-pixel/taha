"""Unit tests for vision.detector using synthetic images (no camera)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from conftest import HEIGHT, PINK_BGR, WIDTH, blank_frame, frame_with_ball, hsv_to_bgr
from vision.config import DetectionConfig, TrackingConfig, VisionConfig
from vision.detector import (
    BallTracker,
    TrackingStatus,
    build_mask,
    circularity,
    detect_ball,
    find_ball_candidate,
)


# ------------------------------------------------------------------ detection
@pytest.mark.parametrize("cx,cy,r", [(320, 240, 30), (100, 80, 15), (550, 400, 60)])
def test_detects_pink_circle_at_known_location(identity_config, cx, cy, r):
    result = detect_ball(frame_with_ball(cx, cy, r), identity_config)

    assert result.found
    assert result.status is TrackingStatus.DETECTED
    assert result.stale_frames == 0
    assert result.pixel_x == pytest.approx(cx, abs=1.0)
    assert result.pixel_y == pytest.approx(cy, abs=1.0)
    # Identity transform: platform coords are pixels re-centred on the image.
    assert result.x == pytest.approx(cx - WIDTH / 2, abs=1.0)
    assert result.y == pytest.approx(cy - HEIGHT / 2, abs=1.0)
    assert result.radius == pytest.approx(r, abs=2.0)
    assert result.area == pytest.approx(math.pi * r * r, rel=0.1)
    assert result.circularity > 0.8


def test_default_config_flips_y_axis():
    """Default axis matrix: +Y points up in the image."""
    result = detect_ball(frame_with_ball(320 + 50, 240 - 40, 20), VisionConfig())
    assert result.x == pytest.approx(50, abs=1.0)
    assert result.y == pytest.approx(40, abs=1.0)


def test_no_matching_colour_is_lost_from_start(identity_config):
    green_ball = frame_with_ball(320, 240, 30, colour=hsv_to_bgr(60, 220, 230))
    result = detect_ball(green_ball, identity_config)

    assert not result.found
    assert result.status is TrackingStatus.LOST
    assert result.x is None and result.y is None
    assert result.stale_frames == 1


def test_rejects_non_circular_pink_blob(identity_config):
    frame = blank_frame()
    # Long thin bar: radius within range, circularity ~0.2.
    cv2.rectangle(frame, (100, 230), (300, 250), PINK_BGR, -1)
    assert find_ball_candidate(frame, identity_config.detection) is None


def test_rejects_noisy_star_blob_but_keeps_real_ball(identity_config):
    frame = frame_with_ball(500, 350, 25)
    # A spiky star-shaped pink blob, larger in area than the ball.
    pts = []
    for i in range(16):
        ang = i * math.pi / 8
        rad = 90 if i % 2 == 0 else 25
        pts.append((int(200 + rad * math.cos(ang)), int(200 + rad * math.sin(ang))))
    cv2.fillPoly(frame, [np.array(pts, dtype=np.int32)], PINK_BGR)

    result = detect_ball(frame, identity_config)
    assert result.found
    assert result.pixel_x == pytest.approx(500, abs=1.0)
    assert result.pixel_y == pytest.approx(350, abs=1.0)


@pytest.mark.parametrize("r", [3, 130])
def test_radius_filter_rejects_too_small_and_too_large(identity_config, r):
    frame = frame_with_ball(320, 240, r)
    assert find_ball_candidate(frame, identity_config.detection) is None


def test_picks_largest_valid_candidate(identity_config):
    frame = frame_with_ball(150, 150, 15)
    cv2.circle(frame, (450, 300), 40, PINK_BGR, -1, lineType=cv2.LINE_AA)
    result = detect_ball(frame, identity_config)
    assert result.pixel_x == pytest.approx(450, abs=1.0)
    assert result.pixel_y == pytest.approx(300, abs=1.0)


def test_speckle_noise_is_removed_by_morphology(identity_config):
    rng = np.random.default_rng(0)
    frame = frame_with_ball(320, 240, 30)
    ys = rng.integers(0, HEIGHT, 400)
    xs = rng.integers(0, WIDTH, 400)
    frame[ys, xs] = PINK_BGR  # isolated single-pixel pink speckles
    mask = build_mask(frame, identity_config.detection)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    assert detect_ball(frame, identity_config).found


def test_hue_wraparound_range():
    red = hsv_to_bgr(2, 220, 230)
    cfg = DetectionConfig(hsv_lower=(170, 150, 50), hsv_upper=(10, 255, 255))
    assert cfg.hue_wraps
    assert find_ball_candidate(frame_with_ball(320, 240, 30, colour=red), cfg) is not None


def test_rgb_channel_order_is_honoured():
    bgr = frame_with_ball(320, 240, 30)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cfg = DetectionConfig()
    assert find_ball_candidate(rgb, cfg, channel_order="RGB") is not None
    # Interpreting RGB data as BGR swaps red/blue: pink becomes a blue-ish hue.
    assert find_ball_candidate(rgb, cfg, channel_order="BGR") is None


@pytest.mark.parametrize("bad", [
    np.zeros((480, 640), dtype=np.uint8),          # grayscale
    np.zeros((480, 640, 4), dtype=np.uint8),       # XRGB8888-style 4 channel
    np.zeros((480, 640, 3), dtype=np.float32),     # wrong dtype
])
def test_invalid_frames_raise(identity_config, bad):
    with pytest.raises(ValueError):
        detect_ball(bad, identity_config)


def test_circularity_formula():
    r = 10.0
    assert circularity(math.pi * r * r, 2 * math.pi * r) == pytest.approx(1.0)
    assert circularity(10.0, 0.0) == 0.0


# ------------------------------------------------------------------- tracking
def test_stale_counting_then_lost(identity_config):
    cfg = VisionConfig(
        coordinates=identity_config.coordinates,
        tracking=TrackingConfig(max_stale_frames=3),
    )
    tracker = BallTracker(cfg)
    first = tracker.update(frame_with_ball(400, 300, 20))
    assert first.found

    empty = blank_frame()
    for expected in (1, 2, 3):
        r = tracker.update(empty)
        assert r.status is TrackingStatus.STALE
        assert not r.found
        assert r.stale_frames == expected
        assert (r.x, r.y) == (first.x, first.y)  # last known good position

    lost = tracker.update(empty)
    assert lost.status is TrackingStatus.LOST
    assert lost.stale_frames == 4
    assert lost.x is None and lost.pixel_x is None
    # Stays LOST and keeps counting.
    assert tracker.update(empty).stale_frames == 5

    # Recovery resets the counter.
    back = tracker.update(frame_with_ball(200, 100, 20))
    assert back.found and back.stale_frames == 0
    assert back.pixel_x == pytest.approx(200, abs=1.0)


def test_default_stale_limit_is_ten():
    tracker = BallTracker(VisionConfig())
    tracker.update(frame_with_ball(320, 240, 20))
    statuses = [tracker.update(blank_frame()).status for _ in range(11)]
    assert statuses[:10] == [TrackingStatus.STALE] * 10
    assert statuses[10] is TrackingStatus.LOST


def test_frame_ids_and_timestamps(identity_config):
    a = detect_ball(blank_frame(), identity_config, frame_id=7, timestamp=1.5)
    assert (a.frame_id, a.timestamp) == (7, 1.5)
    b = detect_ball(blank_frame(), identity_config, a)
    assert b.frame_id == 8


def test_detect_ball_is_pure(identity_config):
    frame = frame_with_ball(320, 240, 30)
    before = frame.copy()
    r1 = detect_ball(frame, identity_config, frame_id=1, timestamp=0.0)
    r2 = detect_ball(frame, identity_config, frame_id=1, timestamp=0.0)
    assert r1 == r2
    assert np.array_equal(frame, before)  # input not modified
