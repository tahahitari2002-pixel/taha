# Steady (Tydi) – Vision module

Vision subsystem of the *Top Dengeleme Sistemi* / **Steady** ball-balancing robot
(3RRS parallel platform). A Raspberry Pi Camera Module 3 looks down at the top
plate; this package finds the ball in every frame and reports its position in
platform-centred coordinates for the (separately developed) PID → inverse
kinematics → PCA9685 servo chain.

```
vision/
├── config.py              all tunable constants (dataclasses, JSON overrides)
├── coordinates.py         pixel → platform transform (the ONLY axis logic)
├── detector.py            pure detection + stale/lost tracking (no camera)
├── camera.py              Picamera2 wrapper (start / capture_frame / stop)
├── pipeline.py            capture thread + detection thread, thread-safe API
├── mjpeg_debug_server.py  browser viewer with overlay, mask view, live stats
└── tools/check_channel_order.py   hardware check for BGR/RGB order
tests/                     pytest suite (synthetic images, fake camera)
```

## Hardware baseline (verified, do not change)

| Item | Value |
|---|---|
| Compute | Raspberry Pi 5, Raspberry Pi OS Bookworm (`libcamera` / `picamera2`) |
| Camera | Camera Module 3 Standard (IMX708) |
| Sensor mode | 1536×864 @ 120 fps (`raw` stream) |
| Processing stream | 640×480 `RGB888` (`main` stream) |
| Frame duration | 10 000 µs → ~100 fps |
| Focus | manual (`AfMode=0`), `LensPosition=5.0` dpt (~20 cm) |
| White balance | AWB **off**, fixed `ColourGains` |

## Installation (on the Pi)

`picamera2` comes from apt and cannot be pip-installed reliably, so a venv must
see system packages:

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-numpy
```

```bash
python3 -m venv --system-site-packages ~/steady-venv
```

```bash
source ~/steady-venv/bin/activate && pip install pytest
```

On a development PC (no camera) `pip install -e ".[dev]"` is enough to run the
tests; `picamera2` is only imported when a real `Camera` is started.

## Running the debug viewer

```bash
python -m vision.mjpeg_debug_server
```

Open `http://<pi-ip>:8000` in any browser (no VNC/VLC needed). The page shows:

* **overlay** view – ball enclosing circle + crosshair (green = detected,
  amber = stale, red = lost), the platform centre with **+X/+Y arrows** drawn
  from the current coordinate calibration, and a status line;
* **raw** view – camera image only;
* **HSV mask** view – the cleaned binary mask; use this while tuning colours;
* a live table with x, y, radius, area, circularity, stale frames, capture
  and detection FPS, dropped frames and errors.

Options: `--port`, `--config tuning.json`, `--stream-fps` (browser stream only,
default 30; detection is never throttled by viewers), `--jpeg-quality`, `-v`.
Ctrl+C shuts down cleanly and releases the camera.

## Using it from the controller (future PID stage)

```python
from vision import VisionConfig, VisionPipeline

with VisionPipeline(VisionConfig()) as vision:
    while running:
        det = vision.get_latest_detection()     # thread-safe, non-blocking
        if det.has_position:                    # DETECTED or still-trusted STALE
            command = pid.update(det.x, det.y, det.timestamp)
        else:                                   # LOST: ball gone / camera frozen
            command = level_platform()
        ...
```

`DetectionResult` fields:

| field | meaning |
|---|---|
| `status` | `DETECTED` (seen this frame), `STALE` (last good position, ball missed for `stale_frames` frames), `LOST` (no trusted position) |
| `found` | `True` only for `DETECTED` |
| `has_position` | `True` for `DETECTED` and `STALE` |
| `x`, `y` | platform coordinates in pixels (float, sub-pixel), `None` when `LOST` |
| `pixel_x`, `pixel_y` | raw image coordinates (for overlays) |
| `radius`, `area`, `circularity` | of the chosen contour |
| `stale_frames` | consecutive frames without a real detection |
| `frame_id`, `timestamp` | frame counter and capture time (`time.monotonic` clock) – use the timestamp for the PID `dt` |

Other API: `wait_for_detection(after_frame_id, timeout_s)` blocks until a newer
result exists (run the PID exactly once per frame); `add_listener(cb)` calls
`cb(result)` on the detection thread for every frame; `stats()` returns FPS and
error counters.

Safety behaviour: a missed frame keeps the last position as `STALE` for up to
`tracking.max_stale_frames` (default 10 = 100 ms at 100 fps), then reports
`LOST` with `x = y = None`. If no new result arrives within
`pipeline.result_timeout_s` (default 250 ms – frozen camera, crashed thread)
`get_latest_detection()` also reports `LOST`. A stale coordinate can therefore
never silently masquerade as a live one.

## Detection algorithm

Per frame (`vision/detector.py`):

1. Gaussian blur (`blur_kernel_size`, default 5, `0` = off)
2. BGR → HSV
3. `cv2.inRange(hsv_lower, hsv_upper)` → mask (hue ranges that wrap past 180 are supported: set `lower H > upper H`)
4. erode + dilate (`morph_kernel_size` 3, `morph_iterations` 1)
5. external contours
6. per contour: minimum enclosing circle, area, circularity `4πA/P²`
7. keep contours with `min_radius_px ≤ r ≤ max_radius_px` **and** circularity ≥ `min_circularity`
8. pick the largest area; transform its centre to platform coordinates

### Coordinate transform

`vision/coordinates.py::pixel_to_platform_coords` is the only place axes are
handled: re-centre on the image centre → multiply by
`coordinates.axis_matrix` → subtract `coordinates.center_offset_px`. The
default matrix `((1, 0), (0, -1))` gives +X right, +Y up in the image.

## Retuning

All constants are in `vision/config.py`. Either edit the defaults there or put
overrides in a JSON file (only the keys you want to change; typos are rejected):

```json
{
  "detection":   {"hsv_lower": [145, 120, 60], "hsv_upper": [175, 255, 255],
                  "min_radius_px": 8, "max_radius_px": 60, "min_circularity": 0.7},
  "tracking":    {"max_stale_frames": 8},
  "camera":      {"lens_position": 4.2, "colour_gains": [1.6, 1.4]},
  "coordinates": {"axis_matrix": [[0, -1], [-1, 0]], "center_offset_px": [4, -3]}
}
```

```bash
python -m vision.mjpeg_debug_server --config tuning.json
```

**HSV range** – open the *HSV mask* view with the ball on the plate under the
real lighting. Raise `S`/`V` minimums until the background is black, then widen
`H` until the ball is a solid white disc. Move the ball to the plate edges and
corners to check it stays solid. Remember OpenCV hue is 0–180, not 0–360.

**Radius range** – read the ball's radius from the stats table at the closest
and farthest positions, then set `min/max_radius_px` ~30 % beyond them. A tight
range is the best defence against false positives.

**Circularity** – real ball readings should be ~0.8–0.9. Lower the threshold
only if the ball is rejected near the edges (motion blur, partial occlusion).

**Noise cleanup** – increase `morph_kernel_size` (odd) if speckle survives; reduce
it if a small/distant ball disappears from the mask.

## Open items (CALIBRATION PENDING)

These are deliberately left as marked placeholders in `vision/config.py`:

| Item | Config key | Why pending | How to calibrate |
|---|---|---|---|
| Ball colour / diameter | `detection.hsv_lower/upper`, `min/max_radius_px` | ball not purchased yet | see "Retuning" |
| White balance | `camera.colour_gains` (placeholder 1.4, 1.5) | needs the real room lighting | put a white/grey card on the plate, briefly run with `awb_enable: true`, read the `ColourGains` metadata Picamera2 reports, then set them here with AWB off again |
| Focus | `camera.lens_position` (5.0 dpt ≈ 20 cm) | mount height not final | `LensPosition ≈ 1 / distance_m`; fine-tune for sharpest ball edge |
| Axis transform | `coordinates.axis_matrix` | camera not mounted | roll the ball along the platform's physical +X; adjust the matrix until reported X grows and Y ≈ 0 (the overlay arrows must match the physical axes); repeat for +Y |
| Centre offset | `coordinates.center_offset_px` | camera not mounted | level the platform, find the point where the ball rests / the mechanical centre, set its reported (x, y) as the offset |

Optional later improvement: once the lighting is fixed, also lock
`camera.exposure_time_us` (≤ 10 000) and `camera.analogue_gain` so brightness
(the V channel) cannot drift either.

## Channel order (reference bug #4)

The camera is configured as `RGB888`. In libcamera/Picamera2 naming this is a
3-channel buffer stored **B, G, R**, i.e. OpenCV-native, so the detector uses
`COLOR_BGR2HSV` (`camera.frame_channel_order = "BGR"`). The unit tests prove the
detector honours the setting, but the camera's real byte order must be checked
once on the Pi with a red target:

```bash
python -m vision.tools.check_channel_order
```

### Verified on hardware

- [ ] Channel order check (date / result): _not yet run_
- [ ] Detection FPS on Pi 5 with debug server open: _not yet measured_

## Reference-project pitfalls and how they are avoided

| # | Pitfall | Here |
|---|---|---|
| 1 | `Lock` imported but unused for frame hand-off | `threading.Condition` around the frame slot, lock around the result; frames are read-only arrays |
| 2 | loop spins at 100 % CPU when the ball is lost | detection thread blocks on a condition with timeout; capture loop has an explicit minimum period and error back-off |
| 3 | dead per-frame computations (`height = 10000 / diameter`) | none; every computed value is returned or used |
| 4 | 4-channel XRGB8888 fed to `BGR2HSV` | 3-channel `RGB888`, explicit channel-order setting, frames of the wrong shape are rejected |
| 5 | no / heavy noise cleanup | light, configurable blur + opening, chosen because circularity needs clean edges |
| 6 | misleading names (`cont_rob()` only detected) | threads `vision-capture` / `vision-detect`; functions named for what they do |
| 7 | placeholder buffers with hand-typed shapes | shape derived only from `CameraConfig.main_size`; the camera verifies the first real frame against it |

## Tests

```bash
python -m pytest
```

The suite uses synthetic images and a fake frame source, so it runs on any PC:
known-position circles (with and without axis flips), no-ball → `LOST`,
stale counting → `LOST` → recovery, star-shaped / bar-shaped pink blobs
rejected by circularity, radius limits, largest-candidate selection, speckle
removal, hue wrap-around, channel order, invalid frames, config validation and
JSON loading, coordinate round-trips, and the threaded pipeline (concurrent
readers, faulty listeners, frozen camera → `LOST`, repeated capture errors →
unhealthy).
