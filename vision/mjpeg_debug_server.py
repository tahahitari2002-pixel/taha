"""Browser-based debug viewer: live MJPEG stream with detection overlay.

Run on the Pi (over SSH)::

    python -m vision.mjpeg_debug_server            # http://<pi-ip>:8000
    python -m vision.mjpeg_debug_server --port 8080 --config my_tuning.json

Endpoints:

* ``/``                      HTML page: stream + live (x, y), status and FPS
* ``/stream.mjpg``           camera image with overlay (``?overlay=0`` for raw)
* ``/mask.mjpg``             the cleaned binary HSV mask (for HSV tuning)
* ``/stats.json``            latest detection + pipeline stats as JSON

Unlike the earlier version (which streamed Picamera2's hardware JPEG
encoder output directly), this server consumes :class:`VisionPipeline`, so
the stream shows *exactly* the frames and detections the controller sees.
JPEG encoding happens in the HTTP client threads, never in the capture or
detection threads, so viewing the stream cannot slow detection down beyond
sharing the CPU. Each client stream is rate-limited (``--stream-fps``).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .config import VisionConfig
from .coordinates import platform_to_pixel_coords
from .detector import DetectionResult, TrackingStatus, build_mask
from .pipeline import PipelineStats, VisionPipeline

log = logging.getLogger(__name__)

BOUNDARY = "FRAME"

# BGR colours for the overlay.
_COLOUR = {
    TrackingStatus.DETECTED: (0, 255, 0),    # green
    TrackingStatus.STALE: (0, 200, 255),     # amber
    TrackingStatus.LOST: (0, 0, 255),        # red
}
_CENTRE_COLOUR = (255, 255, 0)               # cyan: platform centre + axes


def draw_overlay(
    frame: np.ndarray,
    result: DetectionResult,
    config: VisionConfig,
    fps: Optional[float] = None,
) -> np.ndarray:
    """Return a BGR copy of ``frame`` annotated with the detection.

    Draws the platform centre and +X/+Y axes (from the coordinate config, so
    the axis calibration can be checked visually), the ball's enclosing circle
    and crosshair, and a status line. The input frame is never modified.
    """
    if config.camera.frame_channel_order == "RGB":
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        img = frame.copy()
    h, w = img.shape[:2]
    size = (w, h)
    coords = config.coordinates

    # Platform origin and axes (axis length 40 px in platform units).
    ox, oy = platform_to_pixel_coords(0.0, 0.0, size, coords)
    origin = (int(round(ox)), int(round(oy)))
    for label, (ax, ay) in (("+X", (40.0, 0.0)), ("+Y", (0.0, 40.0))):
        px, py = platform_to_pixel_coords(ax, ay, size, coords)
        tip = (int(round(px)), int(round(py)))
        cv2.arrowedLine(img, origin, tip, _CENTRE_COLOUR, 1, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(img, label, (tip[0] + 3, tip[1] + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, _CENTRE_COLOUR, 1, cv2.LINE_AA)

    colour = _COLOUR[result.status]
    if result.pixel_x is not None and result.pixel_y is not None:
        centre = (int(round(result.pixel_x)), int(round(result.pixel_y)))
        radius = max(1, int(round(result.radius)))
        thickness = 2 if result.found else 1
        cv2.circle(img, centre, radius, colour, thickness, cv2.LINE_AA)
        arm = radius + 6
        cv2.line(img, (centre[0] - arm, centre[1]), (centre[0] + arm, centre[1]), colour, 1)
        cv2.line(img, (centre[0], centre[1] - arm), (centre[0], centre[1] + arm), colour, 1)

    if result.has_position:
        text = f"{result.status.value.upper()} x={result.x:+.1f} y={result.y:+.1f} r={result.radius:.1f}"
    else:
        text = f"LOST ({result.stale_frames} missed)"
    if result.status is TrackingStatus.STALE:
        text += f" stale={result.stale_frames}"
    if fps is not None:
        text += f"  {fps:.0f} fps"
    cv2.rectangle(img, (0, 0), (w, 20), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return img


_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Steady Vision Debug</title>
<style>
 body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}
 img{max-width:100%;border:1px solid #444}
 .row{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
 table{border-collapse:collapse;font-variant-numeric:tabular-nums}
 td{padding:2px 10px;border-bottom:1px solid #333}
 .detected{color:#4f4}.stale{color:#fc4}.lost{color:#f55}
 a{color:#8cf}
</style></head><body>
<h2>Steady &ndash; vision debug</h2>
<p>View: <a href="/">overlay</a> | <a href="/?view=raw">raw</a> | <a href="/?view=mask">HSV mask</a></p>
<div class="row">
 <img id="stream" alt="stream">
 <table id="stats"></table>
</div>
<script>
const view = new URLSearchParams(location.search).get("view") || "overlay";
document.getElementById("stream").src =
  view === "mask" ? "/mask.mjpg" : (view === "raw" ? "/stream.mjpg?overlay=0" : "/stream.mjpg");
const fmt = v => (typeof v === "number") ? (Number.isInteger(v) ? v : v.toFixed(2)) : String(v);
async function poll(){
  try{
    const s = await (await fetch("/stats.json", {cache:"no-store"})).json();
    const d = s.detection, p = s.pipeline;
    const rows = [
      ["status", `<span class="${d.status}">${d.status.toUpperCase()}</span>`],
      ["x (platform px)", fmt(d.x)], ["y (platform px)", fmt(d.y)],
      ["radius px", fmt(d.radius)], ["area px²", fmt(d.area)],
      ["circularity", fmt(d.circularity)], ["stale frames", d.stale_frames],
      ["capture fps", fmt(p.capture_fps)], ["detection fps", fmt(p.detection_fps)],
      ["frames dropped", p.frames_dropped], ["capture errors", p.capture_errors],
      ["healthy", p.healthy], ["last error", p.last_error ?? "-"],
    ];
    document.getElementById("stats").innerHTML =
      rows.map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  }catch(e){ /* server restarting; keep polling */ }
  setTimeout(poll, 200);
}
poll();
</script></body></html>
"""


class DebugServer(ThreadingHTTPServer):
    """HTTP server holding a reference to the running pipeline."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], pipeline: VisionPipeline,
                 stream_fps: float, jpeg_quality: int) -> None:
        super().__init__(address, DebugRequestHandler)
        self.pipeline = pipeline
        self.stream_period_s = 1.0 / stream_fps
        self.jpeg_quality = jpeg_quality
        self.shutting_down = threading.Event()


class DebugRequestHandler(BaseHTTPRequestHandler):
    server: DebugServer  # narrowed type for readability

    def log_message(self, fmt: str, *args: object) -> None:  # route to logging
        log.debug("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self._send_bytes(_INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif url.path == "/stats.json":
            self._send_stats()
        elif url.path == "/stream.mjpg":
            overlay = query.get("overlay", ["1"])[0] != "0"
            self._stream(mode="overlay" if overlay else "raw")
        elif url.path == "/mask.mjpg":
            self._stream(mode="mask")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    # ----------------------------------------------------------- responses
    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_stats(self) -> None:
        pipeline = self.server.pipeline
        stats: PipelineStats = pipeline.stats()
        payload = {
            "detection": pipeline.get_latest_detection().to_dict(),
            "pipeline": stats.to_dict(),
        }
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json")

    def _render(self, mode: str, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        cfg = self.server.pipeline.config
        if mode == "mask":
            return build_mask(frame, cfg.detection, cfg.camera.frame_channel_order)
        if mode == "raw":
            if cfg.camera.frame_channel_order == "RGB":
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        fps = self.server.pipeline.stats().detection_fps
        return draw_overlay(frame, result, cfg, fps)

    def _stream(self, mode: str) -> None:
        pipeline = self.server.pipeline
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.server.jpeg_quality]
        last_frame_id = -1
        while not self.server.shutting_down.is_set():
            started = time.monotonic()
            # Block (bounded) until a newer processed frame exists.
            if pipeline.wait_for_detection(last_frame_id, timeout_s=0.5) is None:
                if not pipeline.is_running:
                    break
                continue
            frame, result = pipeline.get_latest_frame_and_detection()
            if frame is None:
                continue
            last_frame_id = result.frame_id

            ok, jpeg = cv2.imencode(".jpg", self._render(mode, frame, result), encode_params)
            if not ok:
                log.warning("JPEG encoding failed for frame %d", result.frame_id)
                continue
            data = jpeg.tobytes()
            try:
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                log.info("stream client %s disconnected", self.address_string())
                return
            # Rate-limit each client so the browser can't starve detection.
            remaining = self.server.stream_period_s - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default all interfaces)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", help="JSON file with config overrides")
    parser.add_argument("--stream-fps", type=float, default=30.0,
                        help="max frames/s sent to each browser (detection is not limited)")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.stream_fps <= 120:
        parser.error("--stream-fps must be in (0, 120]")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    config = VisionConfig.from_json(args.config) if args.config else VisionConfig()
    pipeline = VisionPipeline(config)
    try:
        pipeline.start()
    except Exception as exc:  # report camera init failure clearly and exit
        log.error("could not start vision pipeline: %s", exc)
        return 1

    server = DebugServer((args.host, args.port), pipeline, args.stream_fps, args.jpeg_quality)

    def _shutdown(signum: int, _frame: object) -> None:
        log.info("signal %d received, shutting down", signum)
        server.shutting_down.set()
        # shutdown() blocks until serve_forever returns, so run it elsewhere.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    log.info("debug server on http://%s:%d  (Ctrl+C to stop)", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.shutting_down.set()
        server.server_close()
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
