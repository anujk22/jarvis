#!/usr/bin/env python3
"""
Futuristic orange voice visualizer overlay for Jarvis (standalone).

Captures *system / speaker playback* via a loopback-style input (Stereo Mix,
"What U Hear", etc.) so bars track Jarvis TTS and WAV output on the default
device. Run this in a separate terminal while Jarvis is running.

Dependencies: numpy, sounddevice (same as the main project).

Environment (optional):
  JARVIS_VIZ_DEVICE      Device index (int) or substring of the device name.
  JARVIS_VIZ_GAIN        Linear gain for sensitivity (default 4.0).
  JARVIS_VIZ_BLOCK       Callback block size (default 1024); smaller = snappier.
  JARVIS_VIZ_FULL        Set to 1 to use the full screen height for the viz.
  JARVIS_VIZ_TRANSPARENT Set to 0 to disable magenta chroma-key (uses dark semi-transparent window).
  JARVIS_VIZ_ALPHA       Whole-window opacity 0.0–1.0 if transparent color fails (default 0.92).

Controls: Escape or Q — quit.

If nothing moves: enable Stereo Mix (or similar) in Windows sound settings, or
set JARVIS_VIZ_DEVICE to the correct capture device index from:
  python -c "import sounddevice as sd; print(sd.query_devices())"
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk

import numpy as np
import sounddevice as sd

# --- Orange theme (Jarvis-style) ---
ORANGE_CORE = "#ff7a1a"
ORANGE_HOT = "#ffb347"
ORANGE_DIM = "#cc4a00"
GLOW_BG = "#ff00ff"  # chroma key (magenta) -> transparent on Windows


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _parse_device_spec(raw: str | None) -> int | str | None:
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return s


def resolve_loopback_device() -> int:
    """
    Pick an input device that can hear speaker output.
    Preference: explicit env, then known loopback names, then Stereo Mix, etc.
    """
    spec = _parse_device_spec(os.environ.get("JARVIS_VIZ_DEVICE"))
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    def score_name(name: str) -> int:
        n = name.lower()
        if "stereo mix" in n or "what u hear" in n:
            return 100
        if "loopback" in n:
            return 95
        if "wave out" in n and "mix" in n:
            return 90
        if "cable output" in n:  # VB-Audio virtual cable
            return 85
        return 0

    candidates: list[tuple[int, int, str]] = []
    for i, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0) or 0) < 1:
            continue
        name = str(dev.get("name", ""))
        api = hostapis[int(dev["hostapi"])]["name"]
        pri = score_name(name)
        # Prefer WASAPI / WDM-KS for lower latency when names tie.
        if "WASAPI" in api:
            pri += 3
        elif "WDM-KS" in api:
            pri += 2
        elif "DirectSound" in api:
            pri += 1
        candidates.append((pri, i, name))

    if isinstance(spec, int):
        sd.check_input_settings(device=spec)
        return spec

    if isinstance(spec, str):
        needle = spec.lower()
        for pri, i, name in sorted(candidates, key=lambda x: (-x[0], x[1])):
            if needle in name.lower():
                return i
        raise RuntimeError(f"No input device name contains {spec!r}. Set JARVIS_VIZ_DEVICE to a valid index.")

    if not candidates:
        raise RuntimeError("No input devices found.")

    best = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
    if best[0] < 85:
        raise RuntimeError(
            "No loopback-style capture device found (Stereo Mix, What U Hear, CABLE output, etc.).\n"
            "Enable Stereo Mix in Windows sound settings (Recording → show disabled devices), or install a\n"
            "virtual cable, then re-run. You can also force a device: JARVIS_VIZ_DEVICE=<index or name substring>\n"
            "Use: python -c \"import sounddevice as sd; print(sd.query_devices())\""
        )
    return best[1]


def log_spaced_fft_bands(mag: np.ndarray, n_bands: int) -> np.ndarray:
    """Map rFFT magnitudes to n_bands log-spaced groups (speech-friendly)."""
    n = mag.shape[0]
    if n < 8:
        return np.zeros(n_bands, dtype=np.float64)
    # log edges in bin index space
    edges = np.logspace(0, np.log10(n - 1), n_bands + 1)
    edges = np.clip(np.round(edges).astype(np.int64), 0, n - 1)
    out = np.empty(n_bands, dtype=np.float64)
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        if hi < lo:
            hi = lo
        out[b] = float(np.mean(mag[lo : hi + 1]))
    return out


class AudioCapture(threading.Thread):
    def __init__(self, device: int, frame_q: "queue.Queue[tuple[float, np.ndarray]]", blocksize: int) -> None:
        super().__init__(daemon=True)
        self.device = device
        self.frame_q = frame_q
        self.blocksize = blocksize
        self._stop = threading.Event()
        info = sd.query_devices(device, "input")
        self.samplerate = int(float(info["default_samplerate"]))
        self.channels = min(2, int(info["max_input_channels"]))

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        gain = _env_float("JARVIS_VIZ_GAIN", 4.0)
        win = np.hanning(self.blocksize).astype(np.float64)

        def callback(indata, frames, time_info, status) -> None:
            if status:
                pass
            x = np.asarray(indata, dtype=np.float64)
            if x.ndim == 2 and x.shape[1] > 1:
                mono = x.mean(axis=1)
            else:
                mono = x.reshape(-1)
            if mono.shape[0] < self.blocksize:
                return
            # use last blocksize samples
            block = mono[-self.blocksize :] * win
            rms = float(np.sqrt(np.mean(block**2)) * gain)
            spec = np.abs(np.fft.rfft(block))
            bands = log_spaced_fft_bands(spec, 48)
            # voice-ish emphasis: boost mid band energy slightly
            if bands.shape[0] > 12:
                bands[4:28] *= 1.35
            bands = np.clip(bands * (gain * 0.015), 0.0, 1.0)
            try:
                self.frame_q.put_nowait((rms, bands.astype(np.float32, copy=False)))
            except queue.Full:
                try:
                    _ = self.frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_q.put_nowait((rms, bands.astype(np.float32, copy=False)))
                except queue.Full:
                    pass

        with sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype=np.float32,
            callback=callback,
        ):
            while not self._stop.is_set():
                self._stop.wait(0.2)


def main() -> None:
    device = resolve_loopback_device()
    blocksize = max(256, _env_int("JARVIS_VIZ_BLOCK", 1024))
    frame_q: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=3)

    cap = AudioCapture(device, frame_q, blocksize=blocksize)
    cap.start()

    root = tk.Tk()
    root.title("")
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    use_alpha = os.environ.get("JARVIS_VIZ_TRANSPARENT", "1").strip() not in ("0", "false", "no")
    if use_alpha:
        try:
            root.configure(bg=GLOW_BG)
            root.wm_attributes("-transparentcolor", GLOW_BG)
        except tk.TclError:
            use_alpha = False
    if not use_alpha:
        root.attributes("-alpha", _env_float("JARVIS_VIZ_ALPHA", 0.92))
        root.configure(bg="#0a0608")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{sw}x{sh}+0+0")

    full = os.environ.get("JARVIS_VIZ_FULL", "").strip() in ("1", "true", "yes")
    strip_frac = 1.0 if full else 0.28

    canvas = tk.Canvas(root, highlightthickness=0, bg=GLOW_BG if use_alpha else "#0a0608")
    canvas.pack(fill=tk.BOTH, expand=True)

    n_bars = 40
    bar_ids_core: list[int] = []
    bar_ids_glow: list[int] = []
    arc_id = None

    def setup_drawing(_evt: tk.Event | None = None) -> None:
        nonlocal arc_id
        canvas.delete("all")
        bar_ids_core.clear()
        bar_ids_glow.clear()
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10:
            return
        base_y = int(h * (1.0 - 0.02))
        strip_top = int(h * (1.0 - strip_frac))
        cx = w // 2
        for _ in range(n_bars):
            g = canvas.create_rectangle(0, 0, 0, 0, fill=ORANGE_DIM, width=0, tags="bars")
            c = canvas.create_rectangle(0, 0, 0, 0, fill=ORANGE_CORE, width=0, tags="bars")
            bar_ids_glow.append(g)
            bar_ids_core.append(c)
        r = min(w, h) // 6
        arc_id = canvas.create_arc(
            cx - r,
            strip_top + (base_y - strip_top) // 2 - r,
            cx + r,
            strip_top + (base_y - strip_top) // 2 + r,
            start=200,
            extent=140,
            style=tk.ARC,
            outline=ORANGE_HOT,
            width=3,
            tags="arc",
        )

    canvas.bind("<Configure>", setup_drawing)

    # Envelope followers (smoothed for natural motion)
    env_rms = 0.0
    env_bands = np.zeros(n_bars, dtype=np.float64)
    attack = 0.42
    release = 0.08

    def smooth_toward(cur: float, target: float) -> float:
        a = attack if target > cur else release
        return cur + (target - cur) * a

    def tick() -> None:
        nonlocal env_rms, env_bands
        got_rms: float | None = None
        got_b48: np.ndarray | None = None
        try:
            while True:
                got_rms, got_b48 = frame_q.get_nowait()
        except queue.Empty:
            pass

        if got_b48 is not None and got_b48.shape[0] > 0 and got_rms is not None:
            rms = got_rms
            idx = (np.linspace(0, got_b48.shape[0] - 1, n_bars)).astype(np.int64)
            tgt_b = np.clip(got_b48[idx].astype(np.float64), 0.0, 1.0)
        else:
            rms = env_rms * 0.92
            tgt_b = np.clip(env_bands * 0.93, 0.0, 1.0)

        env_rms = smooth_toward(env_rms, float(np.clip(rms, 0.0, 1.2)))
        for i in range(n_bars):
            env_bands[i] = smooth_toward(env_bands[i], tgt_b[i])

        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w > 20 and h > 20 and bar_ids_core:
            base_y = int(h * (1.0 - 0.02))
            strip_top = int(h * (1.0 - strip_frac))
            mid_y = strip_top + (base_y - strip_top) // 2
            cx = w // 2
            span = w * 0.42
            half = n_bars // 2
            bw = span / max(half, 1)
            pulse = env_rms * min(w, h) * 0.12

            for i in range(half):
                t = env_bands[i]
                h_up = (base_y - strip_top) * (0.12 + 0.88 * t)
                xr0 = cx + i * bw
                xr1 = xr0 + bw * 0.78
                canvas.coords(bar_ids_glow[i], xr0, base_y, xr1, base_y - h_up * 1.08)
                canvas.coords(bar_ids_core[i], xr0 + bw * 0.05, base_y, xr1 - bw * 0.04, base_y - h_up)
                jl = half + i
                xl0 = cx - (i + 1) * bw
                xl1 = xl0 + bw * 0.78
                h_up2 = (base_y - strip_top) * (0.12 + 0.88 * t)
                canvas.coords(bar_ids_glow[jl], xl0, base_y, xl1, base_y - h_up2 * 1.08)
                canvas.coords(bar_ids_core[jl], xl0 + bw * 0.05, base_y, xl1 - bw * 0.04, base_y - h_up2)

            if arc_id is not None:
                r = min(w, h) // 6 + int(pulse)
                canvas.coords(arc_id, cx - r, mid_y - r, cx + r, mid_y + r)
                canvas.itemconfig(arc_id, outline=ORANGE_HOT, width=2 + int(env_rms * 6))

        root.after(16, tick)

    def quit_app(_: tk.Event | None = None) -> None:
        cap.stop()
        cap.join(timeout=2.0)
        root.destroy()

    root.bind("<Escape>", quit_app)
    root.bind("q", quit_app)
    root.bind("Q", quit_app)

    setup_drawing()
    root.after(16, tick)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"jarvis_visualizer_overlay: {e}", file=sys.stderr)
        sys.exit(1)
