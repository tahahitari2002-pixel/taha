"""Empirically verify the channel order of Picamera2 frames (run on the Pi).

Hold a clearly **red** object (or a red sheet of paper) so it fills the centre
of the camera's view, then run::

    python -m vision.tools.check_channel_order

The script captures a few frames with the project's real camera configuration
and reports the mean value of each array channel in the centre region. For a
red target, channel 2 should be clearly brightest if the array is BGR
(``frame_channel_order = "BGR"``, the expected result for ``RGB888``); if
channel 0 is brightest the array is RGB and the config must be changed.

Record the result (date, lighting, output) in README.md, section
"Verified on hardware".
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from vision.camera import Camera, CameraError
from vision.config import VisionConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Picamera2 channel order with a red target.")
    parser.add_argument("--frames", type=int, default=10, help="frames to average")
    parser.add_argument("--config", help="optional JSON config overrides")
    args = parser.parse_args(argv)

    config = VisionConfig.from_json(args.config) if args.config else VisionConfig()
    h, w, _ = config.camera.frame_shape
    y0, y1, x0, x1 = h // 3, 2 * h // 3, w // 3, 2 * w // 3

    try:
        with Camera(config.camera) as cam:
            means = np.mean(
                [cam.capture_frame().image[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
                 for _ in range(max(1, args.frames))],
                axis=0,
            )
    except CameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"format={config.camera.main_format}  centre-region channel means:")
    for i, m in enumerate(means):
        print(f"  channel {i}: {m:6.1f}")
    brightest = int(np.argmax(means))
    detected = {0: "RGB", 2: "BGR"}.get(brightest)
    if detected is None:
        print("Inconclusive: green channel brightest - is the target really red?")
        return 2
    configured = config.camera.frame_channel_order
    verdict = "OK, matches config" if detected == configured else "MISMATCH - update config!"
    print(f"Red target brightest in channel {brightest} -> array is {detected}. "
          f"Configured: {configured}. {verdict}")
    return 0 if detected == configured else 3


if __name__ == "__main__":
    raise SystemExit(main())
