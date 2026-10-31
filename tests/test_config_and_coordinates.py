"""Tests for configuration validation/loading and the coordinate transform."""

from __future__ import annotations

import json

import pytest

from vision.config import (
    CameraConfig,
    CoordinateConfig,
    DetectionConfig,
    TrackingConfig,
    VisionConfig,
)
from vision.coordinates import pixel_to_platform_coords, platform_to_pixel_coords

SIZE = (640, 480)


def test_defaults_match_verified_hardware_baseline():
    cam = CameraConfig()
    assert cam.main_size == (640, 480)
    assert cam.main_format == "RGB888"
    assert cam.raw_size == (1536, 864)
    assert cam.frame_duration_limits_us == (10000, 10000)
    assert cam.af_mode == 0 and cam.awb_enable is False
    assert cam.frame_shape == (480, 640, 3)
    assert cam.target_fps == pytest.approx(100.0)


@pytest.mark.parametrize("kwargs", [
    {"blur_kernel_size": 4},
    {"min_radius_px": 50, "max_radius_px": 10},
    {"min_circularity": 1.5},
    {"hsv_lower": (140, 200, 50), "hsv_upper": (180, 100, 255)},
    {"hsv_lower": (190, 150, 50)},
])
def test_detection_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DetectionConfig(**kwargs)


def test_camera_config_rejects_4_channel_format():
    with pytest.raises(ValueError):
        CameraConfig(main_format="XRGB8888")


def test_singular_axis_matrix_rejected():
    with pytest.raises(ValueError):
        CoordinateConfig(axis_matrix=((1.0, 0.0), (2.0, 0.0)))


def test_from_json_partial_overrides(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "detection": {"hsv_lower": [145, 120, 60], "min_radius_px": 8},
        "coordinates": {"axis_matrix": [[0, -1], [-1, 0]], "center_offset_px": [3, -2]},
        "tracking": {"max_stale_frames": 5},
    }))
    cfg = VisionConfig.from_json(path)
    assert cfg.detection.hsv_lower == (145, 120, 60)
    assert cfg.detection.hsv_upper == DetectionConfig().hsv_upper  # untouched
    assert cfg.detection.min_radius_px == 8
    assert cfg.coordinates.axis_matrix == ((0, -1), (-1, 0))
    assert cfg.tracking == TrackingConfig(max_stale_frames=5)
    assert cfg.camera == CameraConfig()


def test_from_dict_rejects_typos():
    with pytest.raises(ValueError):
        VisionConfig.from_dict({"detection": {"min_raduis_px": 3}})
    with pytest.raises(ValueError):
        VisionConfig.from_dict({"detecton": {}})


def test_to_dict_round_trip():
    cfg = VisionConfig()
    assert VisionConfig.from_dict(cfg.to_dict()) == cfg


def test_recentring_identity():
    cfg = CoordinateConfig(axis_matrix=((1, 0), (0, 1)))
    assert pixel_to_platform_coords(320, 240, SIZE, cfg) == (0.0, 0.0)
    assert pixel_to_platform_coords(0, 0, SIZE, cfg) == (-320.0, -240.0)


def test_swap_and_negate_transform_and_offset():
    cfg = CoordinateConfig(axis_matrix=((0, -1), (-1, 0)), center_offset_px=(5.0, -3.0))
    # Pixel (330, 260) -> centred (10, 20) -> swapped/negated (-20, -10) -> minus offset.
    assert pixel_to_platform_coords(330, 260, SIZE, cfg) == (-25.0, -7.0)


@pytest.mark.parametrize("matrix", [
    ((1, 0), (0, 1)), ((1, 0), (0, -1)), ((0, 1), (-1, 0)), ((0, -1), (-1, 0)),
])
def test_inverse_transform_round_trip(matrix):
    cfg = CoordinateConfig(axis_matrix=matrix, center_offset_px=(4.5, -7.25))
    for px, py in [(0, 0), (123.4, 456.7), (639, 1)]:
        x, y = pixel_to_platform_coords(px, py, SIZE, cfg)
        bx, by = platform_to_pixel_coords(x, y, SIZE, cfg)
        assert (bx, by) == pytest.approx((px, py))
