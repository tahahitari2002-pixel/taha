"""Vision subsystem of the Steady (Tydi) ball-balancing robot.

Public API::

    from vision import VisionConfig, VisionPipeline

    with VisionPipeline(VisionConfig()) as vision:
        det = vision.get_latest_detection()   # thread-safe, non-blocking
        if det.has_position:
            use(det.x, det.y)                 # platform-centred pixels

For camera-free use (tests, offline images) call :func:`detect_ball`
directly. Importing this package never imports ``picamera2``.
"""

from .camera import Camera, CameraError, Frame, FrameSource
from .config import (
    CameraConfig,
    CoordinateConfig,
    DetectionConfig,
    PipelineConfig,
    TrackingConfig,
    VisionConfig,
)
from .coordinates import pixel_to_platform_coords, platform_to_pixel_coords
from .detector import (
    BallCandidate,
    BallTracker,
    DetectionResult,
    TrackingStatus,
    build_mask,
    detect_ball,
    find_ball_candidate,
)
from .pipeline import PipelineStats, VisionPipeline

__all__ = [
    "BallCandidate", "BallTracker", "Camera", "CameraConfig", "CameraError",
    "CoordinateConfig", "DetectionConfig", "DetectionResult", "Frame",
    "FrameSource", "PipelineConfig", "PipelineStats", "TrackingConfig",
    "TrackingStatus", "VisionConfig", "VisionPipeline", "build_mask",
    "detect_ball", "find_ball_candidate", "pixel_to_platform_coords",
    "platform_to_pixel_coords",
]
