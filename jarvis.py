from __future__ import annotations

import argparse
import json
import queue
import random
import re
import socket
import os
import sys
import subprocess
import threading
import time
import wave
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pyautogui
import pyttsx3
import sounddevice as sd
from rich.console import Console
from rich.text import Text
import numpy as np
from faster_whisper import WhisperModel

from jarvis_threejs_viz import (
    JarvisThreeJsVizState,
    ensure_audiovisualizer_dist,
    feed_viz_from_pcm_int16,
    open_kiosk_visualizer,
    start_viz_http_server,
)

APP_NAME = "J.A.R.V.I.S."
PROMPT = "JARVIS > "  # bottom input line (matches status dock)

# Theme: orange + off-white only (no blues / browns / reds)
T_ACCENT = "#FFB84D"  # labels, active HUD, prompt, banner
T_TEXT = "#F5F0E8"    # body text
T_MUTED = "#C9C0B4"   # timestamps, arrows, help, thinking, loading line
T_INACTIVE = "#8A8278"  # inactive HUD chips (soft gray, still warm)
T_RULE = "#D4A574"    # turn separator (muted orange-gold)
T_ERR = "#E8A060"     # errors (amber-orange, not red)
WAKE_PHRASES = ("wake up", "wake-up", "wakeup", "jarvis")
ARM_SECONDS = 8
VOICE_SESSION_SECONDS = 300  # keep voice active after wake
# Direct jarvis.py (no --connect): mic stays in-session until "sleep" (no wake phrase required).
_MIC_SESSION_ALWAYS_UNTIL_SLEEP = float("inf")

# UI → daemon: same TCP socket, control lines (not user transcripts) to pause STT while the UI runs a command.
DAEMON_MIC_BUSY_TOKEN = "__JARVIS_MIC_BUSY__"
DAEMON_MIC_IDLE_TOKEN = "__JARVIS_MIC_IDLE__"


class MicInputSuppression:
    """Process-local depth counter: when >0, mic pipeline discards audio and skips Whisper enqueue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0

    def acquire(self) -> None:
        with self._lock:
            self._depth += 1

    def release(self) -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)

    def reset(self) -> None:
        with self._lock:
            self._depth = 0

    def is_active(self) -> bool:
        with self._lock:
            return self._depth > 0

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "jarvis_config.json"

VOICE_DIR = ROOT / "voices"
VOICE_TEMPLATE = VOICE_DIR / "template.wav"
LLAMA_COMPLETION = ROOT / "bin" / "llama" / "llama-completion.exe"
_DEFAULT_LLM = ROOT / "models" / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
# Qwen2.x ChatML end-of-turn token (spell as concat so editors don't mangle it).
_QWEN_IM_END = "<|" + "im_end" + "|>"

# Mic → one Whisper run per utterance: buffer until this many seconds of quiet (RMS below threshold).
_JARVIS_UTT_END_SILENCE_SEC = float(os.environ.get("JARVIS_UTT_END_SILENCE_SEC", "0.6"))
_JARVIS_UTT_MIN_SEC = float(os.environ.get("JARVIS_UTT_MIN_SEC", "0.35"))
_JARVIS_UTT_MAX_SEC = float(os.environ.get("JARVIS_UTT_MAX_SEC", "28.0"))
_JARVIS_UTT_RMS = float(os.environ.get("JARVIS_UTT_RMS", "0.011"))
# Do not put questions like "what time is it" here — Whisper repeats them on silence/noise (speaker→mic loop).
_WHISPER_INITIAL_PROMPT = "English short voice commands: open notepad, launch app, search google, wake up jarvis."

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.03


@dataclass
class AppState:
    armed_until: float = 0.0
    voice_session_until: float = 0.0
    # Direct jarvis.py: accept all voice as commands until "sleep" (avoids timer/inf edge cases).
    local_mic_open_until_sleep: bool = False
    mic_level: float = 0.0
    last_error: str = ""
    tts_mode: str = "sapi"  # used only as fallback if a clip is missing
    # Voice UI: timeline phases idle|armed|listening; stack overrides with processing|speaking
    ui_phase_stack: list[str] = field(default_factory=list)
    ui_phase_lock: threading.Lock = field(default_factory=threading.Lock)
    _last_printed_hud_eff: str = field(default="", repr=False)
    # Drop mic transcripts while Jarvis is playing audio (stops TTS being re-heard as "commands").
    playback_depth: int = 0
    playback_lock: threading.Lock = field(default_factory=threading.Lock)
    # While >0: command is running (LLM/TTS pipeline); discard mic input so overlap cannot queue another turn.
    command_busy_depth: int = 0
    mic_control_socket: socket.socket | None = None
    mic_control_send_lock: threading.Lock = field(default_factory=threading.Lock)
    # Refcount for daemon mic suppression: command + each playback so IDLE is never sent mid-speech.
    daemon_mic_hold_depth: int = 0
    daemon_hold_lock: threading.Lock = field(default_factory=threading.Lock)
    threejs_viz: JarvisThreeJsVizState | None = None


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def resolved_whisper_model_name() -> str:
    cfg = load_config()
    return os.environ.get("JARVIS_WHISPER", cfg.get("whisper_model", "small.en")).strip() or "small.en"


def whisper_device_options() -> tuple[str, str]:
    dev = os.environ.get("JARVIS_WHISPER_DEVICE", "cpu").strip().lower()
    if dev == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


def resolved_llm_gguf_path() -> Path:
    env = os.environ.get("JARVIS_LLM_MODEL", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    fallbacks = [
        _DEFAULT_LLM,
        ROOT / "models" / "llm" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    ]
    for c in fallbacks:
        if c.is_file():
            return c
    folder = ROOT / "models" / "llm"
    if folder.is_dir():
        found = list(folder.glob("*.gguf"))
        if found:
            return max(found, key=lambda x: x.stat().st_size)
    return _DEFAULT_LLM


def is_garbage_llm_input(cmd: str) -> bool:
    """Typing '4' or other noise should not invoke the LLM."""
    s = (cmd or "").strip()
    if not s:
        return True
    if re.fullmatch(r"\d+", s):
        return True
    if len(s) == 1:
        return True
    if len(s) <= 2 and not re.search(r"[a-zA-Z]", s):
        return True
    if len(set(s.replace(" ", ""))) == 1 and len(s) <= 4:
        return True
    return False


def _chat_time() -> str:
    return time.strftime("%H:%M")


def print_chat_user(console: Console, text: str, *, via: str) -> None:
    """Log a user turn: timestamp, YOU, ▶, message (no rule before)."""
    _ = via  # reserved for future (e.g. mic icon)
    ts = _chat_time()
    line = Text()
    line.append(f"{ts}  ", style=T_MUTED)
    line.append("YOU", style=f"bold {T_ACCENT}")
    line.append("  ▶  ", style=T_MUTED)
    line.append(text, style=T_TEXT)
    console.print(line)


def strip_llm_stream_artifacts(text: str) -> str:
    """Remove llama.cpp / model markers that leak into user-visible text."""
    t = (text or "").strip()
    if not t:
        return t
    t = re.sub(r"\[end of text\]", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*\[end of text\]\s*$", "", t, flags=re.IGNORECASE).strip()
    return t


def _assistant_turn_separator(console: Console) -> None:
    """One light line (Rich Rule can stack visually on Windows; long replies + typewriter felt like many bars)."""
    try:
        w = min(58, max(28, (console.size.width or 80) - 6))
    except Exception:
        w = 48
    console.print(Text("─" * w, style=T_RULE))


def print_assistant_message(console: Console, message: str) -> None:
    """Jarvis reply: timestamp, label, full body at once (typewriter desyncs from TTS on long answers)."""
    message = strip_llm_stream_artifacts(message)
    ts = _chat_time()
    prefix = Text()
    prefix.append(f"{ts}  ", style=T_MUTED)
    prefix.append("J.A.R.V.I.S.", style=f"bold {T_ACCENT}")
    prefix.append("  ▶  ", style=T_MUTED)
    console.print(prefix, end="")
    console.print(Text(message, style=T_TEXT))
    _assistant_turn_separator(console)


def _under_thinking_status(console: Console | None, fn):
    """
    Animated status while Pocket-TTS generates (not a chat line). Label uses capital T.
    """
    if console is None:
        return fn()
    label = Text("Thinking", style=f"italic {T_MUTED}")
    with console.status(
        label,
        spinner="dots12",
        spinner_style=T_ACCENT,
        speed=1.2,
    ):
        return fn()


def voice_timeline_phase(state: AppState) -> str:
    now = time.time()
    if state.local_mic_open_until_sleep:
        return "listening"
    if now < state.voice_session_until:
        return "listening"
    if now < state.armed_until:
        return "armed"
    return "idle"


def effective_voice_ui_phase(state: AppState) -> str:
    with state.ui_phase_lock:
        if state.ui_phase_stack:
            return state.ui_phase_stack[-1]
    return voice_timeline_phase(state)


def push_voice_phase(state: AppState, console: Console | None, phase: str) -> None:
    if not console:
        return
    with state.ui_phase_lock:
        state.ui_phase_stack.append(phase)


def pop_voice_phase(state: AppState, console: Console | None) -> None:
    if not console:
        return
    with state.ui_phase_lock:
        if state.ui_phase_stack:
            state.ui_phase_stack.pop()


def print_voice_status_line(console: Console, state: AppState, *, force: bool = False) -> None:
    """Bottom dock: JARVIS + phase (listening / idle / …), not all-caps chips."""
    with state.ui_phase_lock:
        override = state.ui_phase_stack[-1] if state.ui_phase_stack else None
    base = voice_timeline_phase(state)
    eff = override if override else base
    if not force and eff == state._last_printed_hud_eff:
        return
    state._last_printed_hud_eff = eff

    # (symbol, state word, optional dim hint)
    specs: dict[str, tuple[str, str, str]] = {
        "idle": ("◇", "idle", 'say “wake up” to arm'),
        "armed": ("●", "armed", "say your command"),
        "listening": ("◉", "listening", ""),
        "processing": ("◎", "processing", ""),
        "speaking": ("▶", "speaking", ""),
    }
    sym, word, hint = specs.get(eff, ("·", "ready", ""))
    line = Text()
    line.append("── ", style=T_RULE)
    line.append(f"{sym}  ", style=f"bold {T_ACCENT}")
    line.append("JARVIS", style=f"bold {T_ACCENT}")
    line.append("  ·  ", style=T_MUTED)
    line.append(word, style=f"italic {T_MUTED}")
    if hint:
        line.append("  —  ", style=T_MUTED)
        line.append(hint, style=T_MUTED)
    line.append("  ──", style=T_RULE)
    console.print(line)


def print_prompt_block(console: Console, state: AppState) -> None:
    """Bottom dock: phase line, then branded prompt (utmost bottom is the input cursor)."""
    print_voice_status_line(console, state)
    console.print(Text(PROMPT.strip() + " ", style=f"bold {T_ACCENT}"), end="")


def refresh_terminal_prompt(console: Console, state: AppState) -> None:
    """
    Redraw status + JARVIS > after the speech thread prints (main thread stays inside input()).
    Without this, voice-only turns leave a blank tail with no visible prompt.
    """
    console.print()
    print_voice_status_line(console, state, force=True)
    console.print(Text(PROMPT.strip() + " ", style=f"bold {T_ACCENT}"), end="")


def is_qwen_gguf(path: Path) -> bool:
    return "qwen" in path.name.lower()


def build_llm_prompt(user_msg: str) -> tuple[str, str | None]:
    """
    Returns (prompt_for_llama, reverse_prompt_for_multiturn_stop or None).
    """
    path = resolved_llm_gguf_path()
    sys_rules = (
        "You are Jarvis, a real Windows voice assistant (not fiction, not Marvel or Avengers). "
        "Refer to yourself as Jarvis (one word), never as J.A.R.V.I.S. with periods. "
        "Reply in at most 2 short sentences. Be factual; if you do not know, say so. "
        "Answer ONLY this user message. Do not role-play additional User questions, "
        "do not write 'User:' or 'Assistant:' dialogue, and do not continue an imaginary conversation."
    )
    u = (user_msg or "").strip()
    if is_qwen_gguf(path):
        prompt = (
            f"<|im_start|>system\n{sys_rules}{_QWEN_IM_END}\n"
            f"<|im_start|>user\n{u}{_QWEN_IM_END}\n"
            f"<|im_start|>assistant\n"
        )
        return prompt, "<|im_start|>user"
    full_prompt = f"{sys_rules}\n\nUser: {u}\nAssistant:"
    return full_prompt, "\nUser:"


def sanitize_assistant_spoken_name(text: str) -> str:
    """Dotted acronym reads badly through TTS; normalize to one spoken name."""
    t = text or ""
    t = re.sub(r"(?i)J\.\s*A\.\s*R\.\s*V\.\s*I\.\s*S\.?", "Jarvis", t)
    return t


def sanitize_llm_reply(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    if _QWEN_IM_END in t:
        t = t.split(_QWEN_IM_END)[0].strip()
    im_start = "<|im_start|>"
    if im_start in t:
        t = t.split(im_start)[0].strip()
    t = re.split(r"(?i)\n\s*(?:User|Human)\s*:\s*", t, maxsplit=1)[0].strip()
    t = re.split(r"(?i)\s+User\s*:\s*", t, maxsplit=1)[0].strip()
    t = re.sub(r"(?i)^(Assistant|Jarvis)\s*:\s*", "", t).strip()
    # Drop fake multiturn on one line: "foo. User: bar"
    t = re.split(r"(?i)\sUser\s*:\s*", t, maxsplit=1)[0].strip()
    t = strip_llm_stream_artifacts(t)
    max_chars = int(os.environ.get("JARVIS_LLM_MAX_REPLY_CHARS", "420"))
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0] + "…"
    t = sanitize_assistant_spoken_name(t)
    return t.strip()


def create_whisper_model() -> WhisperModel:
    wmodel = resolved_whisper_model_name()
    wdev, wcomp = whisper_device_options()
    try:
        return WhisperModel(wmodel, device=wdev, compute_type=wcomp)
    except Exception:
        return WhisperModel(wmodel, device="cpu", compute_type="int8")


def transcribe_utterance_audio(
    model: WhisperModel, audio: np.ndarray, samplerate: int = 16000
) -> str:
    """
    Transcribe one full utterance. Whisper's own VAD is off so words at the end aren't dropped
    inside a single clip (fixed short chunks + VAD were cutting sentences).
    """
    prompt = os.environ.get("JARVIS_WHISPER_INITIAL_PROMPT", _WHISPER_INITIAL_PROMPT).strip()
    kw: dict = {
        "language": "en",
        "task": "transcribe",
        "beam_size": 5,
        "best_of": 1,
        "temperature": 0.0,
        "without_timestamps": True,
    }
    if prompt:
        kw["initial_prompt"] = prompt
    segments, _info = model.transcribe(audio, **kw)
    return " ".join(s.text.strip() for s in segments).strip()


def mic_utterances_to_queue(
    text_q: "queue.Queue[str]",
    stop_evt: threading.Event,
    whisper: WhisperModel,
    *,
    state: AppState | None,
    mic_suppressed: Callable[[], bool] | None = None,
) -> None:
    """
    Read mic until the user pauses (~end silence), run Whisper once on the whole buffer, enqueue text.
    Optional AppState for mic level and for skipping/dropping audio while Jarvis runs a command or plays TTS.
    When ``mic_suppressed`` is set (e.g. daemon UI-busy gate), it is combined with that AppState logic.
    """
    def suppressed() -> bool:
        if mic_suppressed is not None and mic_suppressed():
            return True
        if state is not None:
            with state.playback_lock:
                if state.playback_depth > 0 or state.command_busy_depth > 0:
                    return True
        return False

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

    def callback(indata, frames, time_info, status):
        if status:
            return
        if stop_evt.is_set():
            return
        mono = indata[:, 0].copy()
        if state is not None:
            state.mic_level = float(np.max(np.abs(mono))) if mono.size else 0.0
        try:
            audio_q.put_nowait(mono)
        except queue.Full:
            pass

    sr = 16000
    end_silence = _JARVIS_UTT_END_SILENCE_SEC
    min_sec = _JARVIS_UTT_MIN_SEC
    max_sec = _JARVIS_UTT_MAX_SEC
    rms_thresh = _JARVIS_UTT_RMS
    min_samples = int(sr * min_sec)
    max_samples = int(sr * max_sec)

    utt = np.zeros((0,), dtype=np.float32)
    last_voice_t: float | None = None

    with sd.InputStream(
        samplerate=sr,
        dtype="float32",
        channels=1,
        callback=callback,
        device=None,
    ):
        while not stop_evt.is_set():
            try:
                piece = audio_q.get(timeout=0.1)
            except queue.Empty:
                piece = None

            if suppressed():
                utt = np.zeros((0,), dtype=np.float32)
                last_voice_t = None
                continue

            now = time.time()
            if piece is not None:
                rms = float(np.sqrt(np.mean(np.square(piece))))
                if rms >= rms_thresh:
                    last_voice_t = now
                utt = np.concatenate([utt, piece.astype(np.float32, copy=False)])

            if utt.size > max_samples:
                # If we never detected voice above RMS threshold, this is almost certainly
                # background noise. Transcribing it can produce hallucinations such as the
                # Whisper initial_prompt leaking through.
                if last_voice_t is None:
                    utt = np.zeros((0,), dtype=np.float32)
                    continue
                to_send = utt.copy()
                utt = np.zeros((0,), dtype=np.float32)
                last_voice_t = None
                _enqueue_transcription(whisper, to_send, text_q, state=state, suppressed=suppressed)
                continue

            if (
                utt.size >= min_samples
                and last_voice_t is not None
                and (now - last_voice_t) >= end_silence
            ):
                to_send = utt
                utt = np.zeros((0,), dtype=np.float32)
                last_voice_t = None
                _enqueue_transcription(whisper, to_send, text_q, state=state, suppressed=suppressed)


def _enqueue_transcription(
    whisper: WhisperModel,
    audio_f32: np.ndarray,
    text_q: "queue.Queue[str]",
    *,
    state: AppState | None,
    suppressed: Callable[[], bool],
) -> None:
    try:
        if suppressed():
            return
        text = transcribe_utterance_audio(whisper, audio_f32, 16000)
        if not text:
            return
        # Guard: on near-silence, Whisper can emit the initial prompt itself.
        if normalize_text(text) == normalize_text(os.environ.get("JARVIS_WHISPER_INITIAL_PROMPT", _WHISPER_INITIAL_PROMPT)):
            return
        if suppressed():
            return
        text_q.put(text)
    except Exception as e:
        if state is not None:
            state.last_error = str(e)
        time.sleep(0.05)


class Speaker:
    def __init__(self, state: AppState):
        self.state = state
        self.status_console: Console | None = None
        self._sapi = pyttsx3.init()
        self._sapi.setProperty("rate", 185)

        self._audio_lock = threading.Lock()
        self._audio_stream: sd.OutputStream | None = None
        self._audio_rate = 24000  # common for many voice samples
        self._audio_channels = 1

    def _ensure_audio_stream(self, samplerate: int, channels: int) -> sd.OutputStream:
        # Keep one warm output stream to avoid "first play" lag.
        if self._audio_stream is None or self._audio_rate != samplerate or self._audio_channels != channels:
            if self._audio_stream is not None:
                try:
                    self._audio_stream.stop()
                    self._audio_stream.close()
                except Exception:
                    pass
            self._audio_rate = samplerate
            self._audio_channels = channels
            self._audio_stream = sd.OutputStream(
                samplerate=self._audio_rate,
                channels=self._audio_channels,
                dtype="int16",
                blocksize=0,
                latency="high",
            )
            self._audio_stream.start()

            # Warmup: prime the device buffer so the first real clip is not clipped (Windows DAC often eats a few ms).
            warm_s = float(os.environ.get("JARVIS_AUDIO_STREAM_WARMUP_SEC", "0.12"))
            if warm_s > 0:
                silence = np.zeros(
                    (int(self._audio_rate * warm_s), self._audio_channels), dtype=np.int16
                )
                self._audio_stream.write(silence)
        return self._audio_stream

    def _play_pcm_int16(self, data: np.ndarray, samplerate: int) -> None:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        ch = int(data.shape[1])
        # Lead-in silence so the start of speech isn't cut off after stream open or clip changes (set JARVIS_AUDIO_LEAD_MS=0 to disable).
        lead_ms = float(os.environ.get("JARVIS_AUDIO_LEAD_MS", "55"))
        if lead_ms > 0:
            n_lead = int(samplerate * (lead_ms / 1000.0))
            if n_lead > 0:
                pad = np.zeros((n_lead, ch), dtype=np.int16)
                data = np.concatenate([pad, data.astype(np.int16, copy=False)], axis=0)
        daemon_mic_hold_change(self.state, 1)
        with self.state.playback_lock:
            self.state.playback_depth += 1
        try:
            # Drive the visualizer in sync with playback: a single bulk FFT before stream.write()
            # finishes in milliseconds while audio plays for seconds, so /api/viz decayed to ~0 mid-utterance.
            pb_chunk = max(256, int(os.environ.get("JARVIS_VIZ_PLAYBACK_CHUNK", "1024")))
            with self._audio_lock:
                stream = self._ensure_audio_stream(samplerate=samplerate, channels=ch)
                out = data.astype(np.int16, copy=False)
                nfrm = int(out.shape[0])
                for start in range(0, nfrm, pb_chunk):
                    part = out[start : start + pb_chunk]
                    if part.size == 0:
                        break
                    feed_viz_from_pcm_int16(self.state.threejs_viz, part, samplerate)
                    stream.write(part)
        finally:
            with self.state.playback_lock:
                self.state.playback_depth -= 1
            daemon_mic_hold_change(self.state, -1)

    def _play_pcm_int16_async(self, data: np.ndarray, samplerate: int) -> None:
        t = threading.Thread(target=self._play_pcm_int16, args=(data, samplerate), daemon=True)
        t.start()

    def speak_key(self, key: str, fallback_text: str = "", *, blocking: bool = True) -> None:
        """
        Prefer prerecorded voice pack clips from ./voices/<key>.wav.
        If missing (or playback fails), optionally fall back to TTS/SAPI.
        """
        # Support small spelling variations for existing filenames.
        aliases = {
            "confirm_yes": "confirm",
            "yes_sir": "confirm",
        }
        resolved = aliases.get(key, key)

        # Try a few common filename variants (including historical typo).
        candidates = [
            resolved,
            key,
        ]
        if resolved in {"didnt_understand", "didnt_understood"} or key in {"didnt_understand", "didnt_understood"}:
            candidates.extend(["didnt_understand", "didnt_understood", "didnt_udnerstand"])

        for cand in dict.fromkeys(candidates):  # de-dupe, preserve order
            wav = VOICE_DIR / f"{cand}.wav"
            if not wav.exists():
                continue
            try:
                data, rate = self._load_wav_int16(wav)
                if blocking:
                    push_voice_phase(self.state, self.status_console, "speaking")
                    try:
                        self._play_pcm_int16(data, rate)
                    finally:
                        pop_voice_phase(self.state, self.status_console)
                else:
                    self._play_pcm_int16_async(data, rate)
                return
            except Exception:
                # If playback fails, fall back to text-to-speech.
                break

        if fallback_text:
            # Avoid pocket-tts + assistant transcript during voice keys (flickers console while input() waits).
            if blocking:
                self._say_sapi(fallback_text)
            else:
                threading.Thread(target=self._say_sapi, args=(fallback_text,), daemon=True).start()

    def play_thinking_clip(self, *, blocking: bool = False) -> None:
        """Random thinking_1 / thinking_2 clip while Pocket-TTS is generating."""
        opts = [k for k in ("thinking_1", "thinking_2") if (VOICE_DIR / f"{k}.wav").is_file()]
        if not opts:
            return
        self.speak_key(random.choice(opts), "", blocking=blocking)

    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        speak_with_pocket_tts(text, voice_path=str(VOICE_TEMPLATE), speaker=self, console=self.status_console)

    def _say_sapi(self, text: str) -> None:
        daemon_mic_hold_change(self.state, 1)
        with self.state.playback_lock:
            self.state.playback_depth += 1
        try:
            self._sapi.say(text)
            self._sapi.runAndWait()
        finally:
            with self.state.playback_lock:
                self.state.playback_depth -= 1
            daemon_mic_hold_change(self.state, -1)

    def _play_wav(self, path: Path) -> None:
        """Play a WAV file synchronously using the warmed stream."""
        data, rate = self._load_wav_int16(path)
        self._play_pcm_int16(data, rate)

    def _load_wav_int16(self, path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if width != 2:
            raise RuntimeError("Only 16-bit PCM wav is supported.")

        data = np.frombuffer(frames, dtype="<i2")
        if channels > 1:
            data = data.reshape(-1, channels)
        return data, rate


def orange_banner() -> Text:
    banner_path = ROOT / "jarvis.txt"
    txt = banner_path.read_text(encoding="utf-8", errors="replace").rstrip("\n") if banner_path.exists() else "JARVIS"
    t = Text(txt)
    t.stylize(f"bold {T_ACCENT}")
    return t


def configure_windows_utf8() -> None:
    """
    Make Windows terminals render unicode (avoids '???' for block ASCII art).
    Safe to call multiple times.
    """
    if os.name != "nt":
        return
    try:
        # Set UTF-8 code page for the current console.
        os.system("chcp 65001 >NUL")
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def contains_wake_phrase(s: str) -> bool:
    s_norm = normalize_text(s)
    s_norm = s_norm.replace("-", " ")
    s_norm = re.sub(r"\s+", " ", s_norm).strip()
    if any(p in s_norm for p in WAKE_PHRASES):
        return True
    compact = re.sub(r"\s+", "", s_norm)
    return "wakeup" in compact


def strip_wake_phrase(s: str) -> str:
    s_norm = normalize_text(s)
    s_norm = s_norm.replace("-", " ")
    s_norm = re.sub(r"\s+", " ", s_norm).strip()
    for phrase in WAKE_PHRASES:
        if s_norm.startswith(phrase):
            return s_norm[len(phrase) :].strip(" ,:-")
    for phrase in WAKE_PHRASES:
        pat = rf"^\b{re.escape(phrase)}\b[\s,:\-]*"
        if re.match(pat, s_norm):
            return re.sub(pat, "", s_norm).strip()
    return s_norm


def is_probably_non_speech_noise_transcript(s: str) -> bool:
    """
    Whisper sometimes emits short non-speech words from incidental sounds (mouse clicks, taps).
    Filter those so they don't show up as user turns or trigger commands.
    """
    t = normalize_text(s).strip(" .,!?:;\"'`")
    if not t:
        return True
    # Common "click" hallucinations from mouse / trackpad.
    if re.fullmatch(r"(click|clique|tick|tock)(\s+(click|clique|tick|tock))*", t):
        return True
    return False


def open_url(url: str) -> None:
    if not re.match(r"^https?://", url):
        url = "https://" + url
    webbrowser.open(url, new=1)


def open_search(query: str) -> None:
    q = re.sub(r"\s+", "+", query.strip())
    webbrowser.open(f"https://www.google.com/search?q={q}", new=1)


def open_app(app: str) -> None:
    app = app.strip().lower()
    mapping = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
        "cmd": "cmd.exe",
        "terminal": "wt.exe",
        "chrome": "chrome.exe",
        "edge": "msedge.exe",
    }
    exe = mapping.get(app, app)
    import subprocess

    subprocess.Popen(["cmd", "/c", "start", "", exe], shell=False)


def type_text(text: str) -> None:
    pyautogui.typewrite(text, interval=0.01)


def expand_spoken_command(cmd: str) -> str:
    """
    Map common paraphrases to the fixed command phrases handle_command expects,
    so Whisper output like "please launch notepad" still runs actions without an LLM router.
    """
    s = normalize_text(cmd)
    if not s:
        return s

    s = re.sub(r"^(please|can you|could you|will you|hey)\s+", "", s).strip()
    s = re.sub(r"^(i\s+(?:would\s+)?like\s+to|i\s+(?:want|need)\s+to)\s+", "", s).strip()

    s = re.sub(r"^(launch|start)\s+", "open ", s)

    m = re.match(r"^(i\s+(?:want|need)\s+to\s+)(?:open|launch|start)\s+(.+)$", s)
    if m:
        return f"open {m.group(2).strip()}"

    m = re.match(r"^(open|launch|start)\s+(.+)$", s)
    if m:
        rest = re.sub(r"^(the|a)\s+", "", m.group(2).strip())
        return f"open {rest}"

    if not re.match(r"^(open|go to|navigate to|search for|type|ask|chat)\s", s):
        m = re.match(r"^(write|type)\s+(.+)$", s)
        if m:
            return f"type {m.group(2).strip()}"

    if not re.match(r"^(open|go to|navigate to|search for|type|ask|chat)\s", s):
        if not re.search(r"\b(what|who|why|how|when|which|define|explain|capital of)\b", s):
            m = re.search(r"\b(?:google|search(?:\s+for)?|look\s+up)\s+(?:for\s+)?(.+)$", s)
            if m:
                q = m.group(1).strip()
                q = re.sub(r"^(the|a)\s+", "", q)
                if len(q) > 1:
                    return f"search for {q}"

    return s


def dispatch_open_target(target: str, speaker: Speaker) -> None:
    target = target.strip()
    if not target:
        return
    quick_sites = {
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "gmail": "https://mail.google.com",
        "github": "https://github.com",
    }
    key = target.lower()
    if key in quick_sites:
        open_url(quick_sites[key])
        speaker.speak_key("opening", f"Opening {key}.")
        return
    if re.search(r"\.|https?://", target):
        open_url(target)
        speaker.speak_key("opening", "Opening it.")
        return
    open_app(target)
    speaker.speak_key("opening", f"Opening {target}.")


def _llama_postprocess_stdout(out: str) -> str:
    lines = []
    for ln in (out or "").strip().splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("load_", "main:", "llama_", "common_", "print_info:", "system_info:", "sampler", "generate:")):
            continue
        if re.fullmatch(r"\[end of text\]", s, flags=re.I):
            continue
        lines.append(s)
    text = " ".join(lines).strip()
    if " Q:" in text:
        text = text.split(" Q:", 1)[0].strip()
    return strip_llm_stream_artifacts(text)


def run_llama_prompt(
    prompt: str,
    *,
    n_predict: int = 192,
    temp: float = 0.2,
    reverse_prompt: str | None = None,
) -> str:
    model_path = resolved_llm_gguf_path()
    if not LLAMA_COMPLETION.exists() or not model_path.is_file():
        return ""

    threads = os.environ.get("JARVIS_LLM_THREADS", "8").strip() or "8"
    args = [
        str(LLAMA_COMPLETION),
        "-m",
        str(model_path),
        "-p",
        prompt,
        "--temp",
        str(temp),
        "--top-p",
        "0.85",
        "--n-predict",
        str(n_predict),
        "-no-cnv",
        "--no-display-prompt",
        "--color",
        "off",
        "--threads",
        threads,
    ]
    if reverse_prompt:
        args.extend(["--reverse-prompt", reverse_prompt])

    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        return _llama_postprocess_stdout(p.stdout or "")
    except Exception:
        return ""


def _split_tts_chunks(text: str, max_len: int = 380) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_len:
            buf = f"{buf} {p}".strip() if buf else p
        else:
            if buf:
                out.append(buf)
            if len(p) <= max_len:
                buf = p
            else:
                for i in range(0, len(p), max_len):
                    out.append(p[i : i + max_len])
                buf = ""
    if buf:
        out.append(buf)
    return [x for x in out if x]


def _notify_daemon_mic_busy(state: AppState, busy: bool) -> None:
    """Send one BUSY/IDLE transition to jarvis_daemon (tokens are refcounted by daemon_mic_hold_change)."""
    s = state.mic_control_socket
    if s is None:
        return
    tok = DAEMON_MIC_BUSY_TOKEN if busy else DAEMON_MIC_IDLE_TOKEN
    payload = (tok + "\n").encode("utf-8")
    with state.mic_control_send_lock:
        try:
            s.sendall(payload)
        except OSError:
            pass


def daemon_mic_hold_change(state: AppState, delta: int) -> None:
    """
    Refcount holds sent to the daemon so the mic stays off for the whole command *and* for every
    audio output (pocket-tts, WAV clips, SAPI), including clips that run outside handle_command.
    """
    if state.mic_control_socket is None or delta == 0:
        return
    with state.daemon_hold_lock:
        prev = state.daemon_mic_hold_depth
        state.daemon_mic_hold_depth = max(0, state.daemon_mic_hold_depth + delta)
        cur = state.daemon_mic_hold_depth
    if prev == 0 and cur > 0:
        _notify_daemon_mic_busy(state, True)
    elif prev > 0 and cur == 0:
        _notify_daemon_mic_busy(state, False)


def begin_voice_command(state: AppState) -> None:
    with state.playback_lock:
        state.command_busy_depth += 1
    daemon_mic_hold_change(state, 1)


def end_voice_command(state: AppState) -> None:
    with state.playback_lock:
        state.command_busy_depth = max(0, state.command_busy_depth - 1)
    daemon_mic_hold_change(state, -1)


def handle_command(state: AppState, speaker: Speaker, cmd: str, console: Console) -> None:
    cmd = normalize_text(cmd)
    cmd = expand_spoken_command(cmd)
    # Users often dictate commands with trailing punctuation (e.g. "exit.").
    cmd = cmd.strip(" \t\r\n.,!?;:")
    if not cmd:
        return

    begin_voice_command(state)
    try:
        push_voice_phase(state, console, "processing")
        try:
            _handle_command_body(state, speaker, cmd, console)
        finally:
            pop_voice_phase(state, console)
    finally:
        end_voice_command(state)


def _handle_command_body(state: AppState, speaker: Speaker, cmd: str, console: Console) -> None:
    m = re.match(r"^(ask|chat)\s+(.+)$", cmd)
    if m:
        question = m.group(2).strip()
        if is_garbage_llm_input(question):
            speaker.speak_key("didnt_understand", "I did not catch that, sir.")
            return
        answer = llm_answer(question)
        speak_with_pocket_tts(answer, voice_path=str(VOICE_TEMPLATE), speaker=speaker, console=console)
        return

    if cmd in {"sleep", "go to sleep", "stand by", "stop listening"}:
        state.voice_session_until = 0.0
        state.armed_until = 0.0
        state.local_mic_open_until_sleep = False
        speaker.speak_key("listening_off", "Standing by.")
        return

    if cmd in {"exit", "quit"}:
        speaker.speak_key("goodbye", "Goodbye, sir.")
        raise SystemExit(0)

    if cmd in {"clear", "cls"}:
        console.clear()
        console.print(orange_banner())
        console.print()
        return

    if cmd in {"test sound", "test audio"}:
        speaker.speak_key("boot", "Systems online. Welcome, sir.")
        return

    m = re.match(r"^(open)\s+(.+)$", cmd)
    if m:
        dispatch_open_target(m.group(2).strip(), speaker)
        return

    m = re.match(r"^(go to|navigate to)\s+(.+)$", cmd)
    if m:
        open_url(m.group(2).strip())
        speaker.speak_key("done", "Done.")
        return

    m = re.match(r"^(search for)\s+(.+)$", cmd)
    if m:
        open_search(m.group(2).strip())
        speaker.speak_key("searching", "Searching.")
        return

    m = re.match(r"^(type)\s+(.+)$", cmd)
    if m:
        type_text(m.group(2))
        speaker.speak_key("done", "Done.")
        return

    handle_natural_language(state, speaker, console, cmd)


def llm_answer(prompt: str) -> str:
    """
    Local GGUF via llama.cpp. Default: Qwen2.5-1.5B-Instruct (see setup_llm.ps1) or JARVIS_LLM_MODEL.
    """
    if not LLAMA_COMPLETION.exists() or not resolved_llm_gguf_path().is_file():
        return "LLM not installed yet. Run scripts\\setup_llm.ps1."

    full_prompt, rev = build_llm_prompt(prompt)
    n_pred = int(os.environ.get("JARVIS_LLM_N_PREDICT", "140"))
    answer = run_llama_prompt(full_prompt, n_predict=n_pred, temp=0.2, reverse_prompt=rev)
    return sanitize_llm_reply(answer if answer else "I don't have a response.")


def handle_natural_language(state: AppState, speaker: Speaker, console: Console, cmd: str) -> None:
    """
    Conversational fallback when no scripted command matched: local LLM answer + pocket-tts voice.
    Action-style phrases are handled by expand_spoken_command + fixed matchers first.
    """
    if is_garbage_llm_input(cmd):
        speaker.speak_key("didnt_understand", "I did not catch that, sir.")
        return

    reply = llm_answer(cmd)
    if reply.startswith("LLM error:") or "not installed" in reply.lower():
        speaker.speak_key("didnt_understand", "Sorry, I did not understand that.")
        if reply:
            console.print(Text(reply, style=T_MUTED))
        return
    speak_with_pocket_tts(reply, voice_path=str(VOICE_TEMPLATE), speaker=speaker, console=console)


def _prune_ephemeral_pocket_cache(cache_dir: Path) -> None:
    """Remove prior on-the-spot Pocket-TTS outputs; keep only fixed names like warmup."""
    keep = frozenset({"_jarvis_pocket_warmup.wav"})
    try:
        for p in cache_dir.glob("pocket_*_*.wav"):
            if p.is_file() and p.name not in keep:
                try:
                    p.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def _pocket_tts_generate_wav(text_chunk: str, vpath: Path, out_path: Path) -> bool:
    device = os.environ.get("JARVIS_POCKET_DEVICE", "cpu").strip() or "cpu"
    max_tok = os.environ.get("JARVIS_POCKET_MAX_TOKENS", "200").strip() or "200"
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pocket_tts",
            "generate",
            "-q",
            "--voice",
            str(vpath),
            "--text",
            text_chunk,
            "--output-path",
            str(out_path),
            "--device",
            device,
            "--max-tokens",
            max_tok,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return out_path.is_file() and out_path.stat().st_size > 400


def speak_with_pocket_tts(
    text: str,
    voice_path: str,
    speaker: Speaker,
    *,
    console: Console | None = None,
) -> None:
    """
    Voice-clone via Pocket-TTS. Starts audio generation in a background thread, then (when console
    is passed) prints the assistant line so text appears while the first clip renders.
    """
    text = strip_llm_stream_artifacts((text or "").strip())
    text = sanitize_assistant_spoken_name(text)
    if not text:
        return

    c = console if console is not None else speaker.status_console
    if c:
        push_voice_phase(speaker.state, c, "speaking")
    try:
        _speak_with_pocket_tts_impl(text, voice_path, speaker, console=c)
    finally:
        if c:
            pop_voice_phase(speaker.state, c)


def _speak_with_pocket_tts_impl(
    text: str,
    voice_path: str,
    speaker: Speaker,
    *,
    console: Console | None,
) -> None:
    vpath = Path(voice_path)
    if not vpath.is_file():
        if console:
            print_assistant_message(console, text)
        speaker._say_sapi(text)
        return

    chunk_max = int(os.environ.get("JARVIS_TTS_CHUNK_CHARS", "520"))

    try:
        cache_dir = ROOT / ".cache"
        cache_dir.mkdir(exist_ok=True)
        _prune_ephemeral_pocket_cache(cache_dir)
        chunks = _split_tts_chunks(text, max_len=chunk_max)
        if not chunks:
            return

        if len(chunks) == 1:
            out_path = cache_dir / f"pocket_one_{time.time_ns() & 0xFFFFFFFF}.wav"
            done = threading.Event()
            ok_box: list[bool] = [False]

            def gen_one() -> None:
                ok_box[0] = _pocket_tts_generate_wav(chunks[0], vpath, out_path)
                done.set()

            threading.Thread(target=gen_one, daemon=True).start()
            _under_thinking_status(console, lambda: done.wait(timeout=180))
            if ok_box[0] and out_path.is_file():
                th = threading.Thread(target=lambda: speaker._play_wav(out_path), daemon=True)
                th.start()
                if console:
                    print_assistant_message(console, text)
                th.join(timeout=600)
            else:
                if console:
                    print_assistant_message(console, text)
                speaker._say_sapi(text)
            return

        q: "queue.Queue[tuple[int, Path | None]]" = queue.Queue()

        def producer() -> None:
            for i, ch in enumerate(chunks):
                outp = cache_dir / f"pocket_{i}_{time.time_ns() & 0xFFFFFFFF}.wav"
                if _pocket_tts_generate_wav(ch, vpath, outp):
                    q.put((i, outp))
                else:
                    q.put((i, None))
                    return

        threading.Thread(target=producer, daemon=True).start()
        _idx0, path0 = _under_thinking_status(console, lambda: q.get())
        if path0 is None:
            if console:
                print_assistant_message(console, text)
            speaker._say_sapi(text)
            return

        def play_sequential() -> None:
            speaker._play_wav(path0)
            for expect_i in range(1, len(chunks)):
                _i2, p2 = q.get()
                if p2 is None:
                    speaker._say_sapi(" ".join(chunks[expect_i:]).strip() or text)
                    return
                speaker._play_wav(p2)

        th = threading.Thread(target=play_sequential, daemon=True)
        th.start()
        if console:
            print_assistant_message(console, text)
        th.join(timeout=600)
        return
    except Exception:
        pass

    if console:
        print_assistant_message(console, text)
    speaker._say_sapi(text)


def speech_worker(
    text_q: "queue.Queue[str]",
    stop_evt: threading.Event,
    state: AppState,
    console: Console | None = None,
) -> None:
    _ = console
    whisper = create_whisper_model()
    mic_utterances_to_queue(text_q, stop_evt, whisper, state=state)


def ensure_whisper_warm(console: Console) -> None:
    """
    Trigger model download/cache early so the first command isn't slow.
    """
    wmodel = resolved_whisper_model_name()
    wdev, _ = whisper_device_options()
    console.print(
        Text(
            f"Loading speech model {wmodel!r} ({wdev}) — first run may download…",
            style=T_MUTED,
        )
    )
    _ = create_whisper_model()


def socket_reader_loop(sock: socket.socket, out_q: "queue.Queue[str]", stop_evt: threading.Event) -> None:
    buf = b""
    try:
        while not stop_evt.is_set():
            chunk = sock.recv(4096)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    out_q.put(text)
    except Exception:
        return


def run_ui(connect: tuple[str, int] | None, *, visualize: bool = False) -> None:
    configure_windows_utf8()
    # LLM output can contain "["; disable markup parsing. legacy_windows=False prefers UTF-8 VT output on Win10+.
    console = Console(markup=False, highlight=False, legacy_windows=False)
    console.clear()
    console.print(orange_banner())
    console.print(Text("    -----  J.A.R.V.I.S.  -----", style=f"bold {T_ACCENT}"))
    console.print(Text("Just A Rather Very Intelligent System", style=T_MUTED))
    console.print()

    state = AppState()
    if visualize:
        try:
            dist = ensure_audiovisualizer_dist(ROOT)
            viz = JarvisThreeJsVizState()
            httpd, vport = start_viz_http_server(viz, dist)
            state.threejs_viz = viz
            setattr(state, "_viz_httpd", httpd)  # retain server for process lifetime
            vurl = f"http://127.0.0.1:{vport}/"
            viz_proc = open_kiosk_visualizer(vurl)
            setattr(state, "_viz_proc", viz_proc)
            time.sleep(0.35)
            console.print(Text(f"Visualizer (audio from this run): {vurl}", style=f"bold {T_ACCENT}"))
            console.print(
                Text(
                    "Use this kiosk window only — Parcel, file://, or an old tab will not get live audio (sphere stays smooth).",
                    style=T_MUTED,
                )
            )
            console.print(
                Text(
                    "Close the browser window or Alt+F4 to leave kiosk; Jarvis keeps running in this console.",
                    style=T_MUTED,
                )
            )
            console.print()
        except Exception as e:
            state.threejs_viz = None
            console.print(Text(f"[error] Visualizer failed to start: {e}", style=T_ERR))
            console.print()

    try:
        if connect is None:
            # Terminal is already "live": accept voice commands without a wake phrase until sleep.
            state.voice_session_until = _MIC_SESSION_ALWAYS_UNTIL_SLEEP
            state.armed_until = _MIC_SESSION_ALWAYS_UNTIL_SLEEP
            state.local_mic_open_until_sleep = True

        if connect is None:
            ensure_whisper_warm(console)

        speaker = Speaker(state)
        speaker.status_console = console

        def pocket_tts_warmup_background() -> None:
            if not VOICE_TEMPLATE.is_file():
                return

            def run() -> None:
                cache_dir = ROOT / ".cache"
                cache_dir.mkdir(exist_ok=True)
                outp = cache_dir / "_jarvis_pocket_warmup.wav"
                try:
                    dev = os.environ.get("JARVIS_POCKET_DEVICE", "cpu").strip() or "cpu"
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pocket_tts",
                            "generate",
                            "-q",
                            "--voice",
                            str(VOICE_TEMPLATE),
                            "--device",
                            dev,
                            "--text",
                            "Ready.",
                            "--output-path",
                            str(outp),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                except Exception:
                    pass

            threading.Thread(target=run, daemon=True).start()

        pocket_tts_warmup_background()

        # Print immediately, play greeting async for seamless feel.
        console.print(Text("Welcome, Sir.", style=f"bold {T_ACCENT}"))
        console.print(Text("How may I assist you today?", style=T_TEXT))
        if connect is None:
            # Startup: only voices/wake.wav (no default SAPI — wrong voice / odd phrases). Opt-in: JARVIS_WAKE_SAPI=1.
            if (VOICE_DIR / "wake.wav").is_file():
                speaker.speak_key("wake", "", blocking=False)
            elif os.environ.get("JARVIS_WAKE_SAPI", "").strip().lower() in ("1", "true", "yes"):
                threading.Thread(
                    target=speaker._say_sapi,
                    args=("Welcome, sir. How may I assist you today.",),
                    daemon=True,
                ).start()
        console.print()
        console.print(
            Text(
                "I can open websites, apps, type text, and answer questions via the local LLM.",
                style=T_MUTED,
            )
        )
        console.print(
            Text(
                "Say things like “launch notepad”, “google …”, or ask anything after wake.",
                style=T_MUTED,
            )
        )
        console.print(Text("Commands: clear, exit, test sound, sleep", style=T_MUTED))
        console.print()

        stop_evt = threading.Event()
        speech_q: "queue.Queue[str]" = queue.Queue()

        if connect is None:
            speech_thread = threading.Thread(
                target=speech_worker, args=(speech_q, stop_evt, state, console), daemon=True
            )
            speech_thread.start()
        else:
            host, port = connect
            try:
                s = socket.create_connection((host, port), timeout=10)
                state.mic_control_socket = s
                t = threading.Thread(target=socket_reader_loop, args=(s, speech_q, stop_evt), daemon=True)
                t.start()
                # If we were launched by the daemon on a "wake up" phrase, the wake phrase
                # may have occurred before this UI connected. Auto-arm on connect.
                now = time.time()
                state.armed_until = now + ARM_SECONDS
                state.voice_session_until = now + VOICE_SESSION_SECONDS
                # Same clip as local startup (wake.wav), not confirm.wav.
                speaker.speak_key("wake", "", blocking=False)
            except Exception as e:
                console.print(Text(f"[error] Failed to connect to daemon: {e}", style=T_ERR))
                console.print(Text("Starting local microphone mode instead.", style=T_MUTED))
                connect = None
                state.voice_session_until = _MIC_SESSION_ALWAYS_UNTIL_SLEEP
                state.armed_until = _MIC_SESSION_ALWAYS_UNTIL_SLEEP
                state.local_mic_open_until_sleep = True
                ensure_whisper_warm(console)
                speech_thread = threading.Thread(
                    target=speech_worker, args=(speech_q, stop_evt, state, console), daemon=True
                )
                speech_thread.start()

        def speech_loop():
            last_final: str = ""
            last_final_at: float = 0.0

            def normalize_phrase(s: str) -> str:
                s = normalize_text(s)
                # common Whisper artifacts for commands
                s = re.sub(r"^(the|a)\s+", "", s).strip()
                s = s.replace(" dot ", ".")
                s = s.replace(" slash ", "/")
                s = s.replace(" colon ", ":")
                s = re.sub(r"\s+", " ", s).strip()
                return s

            def log_voice_turn(phrase: str) -> None:
                print_chat_user(console, phrase, via="microphone")

            while not stop_evt.is_set():
                try:
                    heard = speech_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                now = time.time()
                phrase = (heard or "").strip()
                if not phrase:
                    continue

                phrase_norm = normalize_phrase(phrase)
                if is_probably_non_speech_noise_transcript(phrase_norm):
                    continue
                if phrase_norm == last_final and (now - last_final_at) < 1.2:
                    continue
                last_final = phrase_norm
                last_final_at = now

                woke = contains_wake_phrase(phrase_norm)
                in_session = now < state.voice_session_until
                armed = state.local_mic_open_until_sleep or in_session or (now < state.armed_until)
                if woke:
                    state.armed_until = now + ARM_SECONDS
                    state.voice_session_until = now + VOICE_SESSION_SECONDS
                    cmd = strip_wake_phrase(phrase_norm)
                    if not cmd:
                        speaker.speak_key("confirm", "Yes, sir?")
                        refresh_terminal_prompt(console, state)
                        continue
                elif not armed:
                    continue
                else:
                    cmd = phrase_norm

                log_voice_turn(phrase)
                try:
                    handle_command(state, speaker, cmd, console)
                    # successful command keeps the voice session alive
                    state.voice_session_until = time.time() + VOICE_SESSION_SECONDS
                except SystemExit:
                    stop_evt.set()
                    os._exit(0)
                except Exception as e:
                    state.last_error = str(e)
                    console.print(Text(f"[error] {e}", style=T_ERR))
                finally:
                    refresh_terminal_prompt(console, state)

        bg = threading.Thread(target=speech_loop, daemon=True)
        bg.start()

        try:
            while True:
                try:
                    print_prompt_block(console, state)
                    cmd = input()
                except (EOFError, KeyboardInterrupt):
                    raise SystemExit(0)
                cmd_stripped = (cmd or "").strip()
                if cmd_stripped:
                    # Typed text already appeared on the JARVIS > line; skip duplicate YOU log.
                    console.print()
                handle_command(state, speaker, cmd, console)
        except SystemExit:
            stop_evt.set()
            sys.exit(0)
    finally:
        # Ensure kiosk visualizer closes when Jarvis exits.
        try:
            httpd = getattr(state, "_viz_httpd", None)
            if httpd is not None:
                try:
                    httpd.shutdown()
                except Exception:
                    pass
                try:
                    httpd.server_close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            proc = getattr(state, "_viz_proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="", help="Connect to daemon at host:port for speech input")
    ap.add_argument(
        "--visualize",
        action="store_true",
        help="Open the Three.js fullscreen visualizer driven by Jarvis speech (PCM output).",
    )
    args = ap.parse_args()

    connect = None
    if args.connect:
        host, port_s = args.connect.split(":", 1)
        connect = (host, int(port_s))

    visualize = args.visualize or (
        os.environ.get("JARVIS_VISUALIZE", "").strip().lower() in ("1", "true", "yes")
    )

    # Migrate old root-level config if present
    old_cfg = ROOT / "jarvis_config.json"
    if old_cfg.exists() and not CONFIG_PATH.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        old_cfg.replace(CONFIG_PATH)

    run_ui(connect, visualize=visualize)


if __name__ == "__main__":
    main()

