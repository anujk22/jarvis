"""
Fullscreen Three.js audio visualizer support: serve Parcel-built static files,
expose /api/viz with values derived from Jarvis PCM playback, open kiosk browser.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


class JarvisThreeJsVizState:
    """Thread-safe frequency / RMS snapshot for the browser poll endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._avg_frequency: float = 0.0
        self._rms: float = 0.0
        self._last_feed_t: float = 0.0

    def feed_mono_pcm(self, mono_int16: np.ndarray, samplerate: int) -> None:
        """Update from one block of mono int16 PCM (any length; padded internally)."""
        x = np.asarray(mono_int16, dtype=np.float64).reshape(-1)
        n = int(x.shape[0])
        if n < 64:
            return
        bs = min(2048, max(1024, n))
        if n > bs:
            x = x[:bs]
        elif n < bs:
            x = np.pad(x, (0, bs - n))
        x = x / 32768.0
        w = np.hanning(bs)
        spec = np.abs(np.fft.rfft(x * w))
        if spec.shape[0] < 2:
            return
        tail = spec[1:]
        rms = float(np.sqrt(np.mean((x * w) ** 2)))
        mean_mag = float(np.mean(tail))
        peak_mag = float(np.max(tail))
        # Web Audio's getAverageFrequency() is much "hotter" than a raw FFT mean; speech needs gain.
        gain = float(os.environ.get("JARVIS_THREEJS_VIZ_GAIN", "1"))
        combined = peak_mag * 520.0 + mean_mag * 320.0 + rms * 560.0
        avg_u8 = float(np.clip(combined * gain, 0.0, 255.0))
        rms_u8 = float(np.clip((rms * 720.0 + peak_mag * 180.0) * gain, 0.0, 255.0))
        now = time.monotonic()
        with self._lock:
            self._avg_frequency = 0.62 * self._avg_frequency + 0.38 * avg_u8
            self._rms = 0.55 * self._rms + 0.45 * rms_u8
            self._last_feed_t = now

    def get_json_bytes(self) -> bytes:
        now = time.monotonic()
        with self._lock:
            avg = self._avg_frequency
            rms = self._rms
            if now - self._last_feed_t > 0.12:
                # Decay when Jarvis is silent (SAPI and idle).
                decay = 0.88 ** min(40.0, (now - self._last_feed_t) / 0.05)
                avg *= decay
                rms *= decay
                self._avg_frequency = avg
                self._rms = rms
            return json.dumps({"avg": avg, "rms": rms}).encode("utf-8")


def _make_http_handler_class(
    viz: JarvisThreeJsVizState, directory: str
) -> type[http.server.SimpleHTTPRequestHandler]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):
            if os.environ.get("JARVIS_VIZ_HTTP_LOG", "").strip() in ("1", "true", "yes"):
                super().log_message(format, *args)

        def do_GET(self) -> None:  # noqa: N802
            req_path = urlparse(self.path).path
            if req_path.rstrip("/") == "/api/viz":
                body = viz.get_json_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    return Handler


def start_viz_http_server(viz: JarvisThreeJsVizState, dist_dir: Path) -> tuple[socketserver.TCPServer, int]:
    Handler = _make_http_handler_class(viz, str(dist_dir.resolve()))
    host = os.environ.get("JARVIS_VIZ_BIND", "127.0.0.1").strip() or "127.0.0.1"
    port_env = os.environ.get("JARVIS_VIZ_PORT", "").strip()
    bind_port = int(port_env) if port_env.isdigit() else 0
    httpd = socketserver.ThreadingTCPServer((host, bind_port), Handler)
    _, port = httpd.server_address[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, int(port)


def ensure_audiovisualizer_dist(repo_root: Path) -> Path:
    av = repo_root / "audiovisualizer"
    dist = av / "dist"
    if not (av / "package.json").is_file():
        raise FileNotFoundError(f"Missing audiovisualizer bundle: {av}")
    if (dist / "index.html").is_file():
        return dist
    npm = shutil_which("npm")
    if not npm:
        raise RuntimeError(
            "audiovisualizer is not built (no dist/) and npm was not found in PATH. "
            f"Run: cd \"{av}\" && npm install && npm run build"
        )
    subprocess.run([npm, "install"], cwd=str(av), check=True, shell=False)
    subprocess.run(
        [npm, "run", "build"],
        cwd=str(av),
        check=True,
        shell=False,
    )
    if not (dist / "index.html").is_file():
        raise RuntimeError(f"Build finished but {dist / 'index.html'} is missing")
    return dist


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def open_kiosk_visualizer(url: str) -> subprocess.Popen | None:
    if sys.platform == "win32":
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        ]
        for exe in candidates:
            if exe and os.path.isfile(exe):
                return subprocess.Popen(
                    [exe, "--kiosk", url, "--new-window"],
                    close_fds=True,
                    cwd=os.path.expanduser("~"),
                )
    import webbrowser

    webbrowser.open(url)
    return None


def feed_viz_from_pcm_int16(
    viz: JarvisThreeJsVizState | None, data: np.ndarray, samplerate: int, frame: int = 1024
) -> None:
    if viz is None:
        return
    d = np.asarray(data, dtype=np.int16)
    if d.ndim == 1:
        mono = d
    else:
        mono = d.mean(axis=1).astype(np.int16)
    if mono.size == 0:
        return
    step = frame
    for i in range(0, int(mono.shape[0]), step):
        sl = mono[i : i + step]
        if sl.size < step:
            sl = np.pad(sl.astype(np.int64), (0, step - sl.size)).astype(np.int16)
        viz.feed_mono_pcm(sl, samplerate)
