from __future__ import annotations

import argparse
import json
import queue
import re
import socket
import os
import sys
import subprocess
import threading
import time
import wave
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import pyautogui
import pyttsx3
import sounddevice as sd
from rich.console import Console
from rich.rule import Rule
from rich.text import Text
import numpy as np
from faster_whisper import WhisperModel

APP_NAME = "J.A.R.V.I.S."
PROMPT = "J.A.R.V.I.S. > "
WAKE_PHRASES = ("wake up", "jarvis")
ARM_SECONDS = 8
VOICE_SESSION_SECONDS = 300  # keep voice active after wake

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "jarvis_config.json"

VOICE_DIR = ROOT / "voices"
VOICE_TEMPLATE = VOICE_DIR / "template.wav"
LLAMA_COMPLETION = ROOT / "bin" / "llama" / "llama-completion.exe"
_DEFAULT_LLM = ROOT / "models" / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
# Qwen2.x ChatML end-of-turn token (spell as concat so editors don't mangle it).
_QWEN_IM_END = "<|" + "im_end" + "|>"

# Longer buffers + small.en (or JARVIS_WHISPER=base.en|medium.en) help accuracy vs tiny chunks on tiny.en.
_WHISPER_CHUNK_SEC = float(os.environ.get("JARVIS_WHISPER_CHUNK_SEC", "2.2"))
_WHISPER_INITIAL_PROMPT = (
    "Jarvis, wake up. Open notepad, search for weather, launch calculator, google maps. "
    "How is the weather? What time is it?"
)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.03


@dataclass
class AppState:
    armed_until: float = 0.0
    voice_session_until: float = 0.0
    mic_device_index: int | None = None
    mic_device_name: str = ""
    mic_level: float = 0.0
    last_error: str = ""
    tts_mode: str = "sapi"  # used only as fallback if a clip is missing


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


def print_user_message(console: Console, text: str, *, via: str) -> None:
    """via is 'microphone' or 'keyboard'."""
    lbl = "microphone" if via == "microphone" else "keyboard"
    console.print(Rule(style="dim"))
    console.print(Text.assemble(("You ", "bold cyan"), (f"({lbl})", "cyan"), (" · ", "dim"), (text, "white")))


def print_assistant_message(console: Console, message: str) -> None:
    console.print(Text.assemble(("Jarvis", "bold green"), (" · ", "dim"), (message, "white")))
    console.print()


def is_qwen_gguf(path: Path) -> bool:
    return "qwen" in path.name.lower()


def build_llm_prompt(user_msg: str) -> tuple[str, str | None]:
    """
    Returns (prompt_for_llama, reverse_prompt_for_multiturn_stop or None).
    """
    path = resolved_llm_gguf_path()
    sys_rules = (
        "You are J.A.R.V.I.S., a real Windows voice assistant (not fiction, not Marvel or Avengers). "
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
    max_chars = int(os.environ.get("JARVIS_LLM_MAX_REPLY_CHARS", "420"))
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0] + "…"
    return t.strip()


def create_whisper_model() -> WhisperModel:
    wmodel = resolved_whisper_model_name()
    wdev, wcomp = whisper_device_options()
    try:
        return WhisperModel(wmodel, device=wdev, compute_type=wcomp)
    except Exception:
        return WhisperModel(wmodel, device="cpu", compute_type="int8")


def transcribe_audio_chunk(model: WhisperModel, chunk: np.ndarray) -> str:
    segments, _info = model.transcribe(
        chunk,
        language="en",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 250},
        initial_prompt=_WHISPER_INITIAL_PROMPT,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def list_mics() -> list[tuple[int, str]]:
    devices = sd.query_devices()
    out: list[tuple[int, str]] = []
    for i, d in enumerate(devices):
        try:
            if int(d.get("max_input_channels", 0)) > 0:
                out.append((i, str(d.get("name", f"device-{i}"))))
        except Exception:
            continue
    return out


def pick_mic_interactive(console: Console, state: AppState) -> None:
    mics = list_mics()
    if not mics:
        state.last_error = "No microphone devices detected."
        return

    console.print("[bold]Microphones[/bold]")
    for i, name in mics:
        console.print(f"  {i}: {name}")

    console.print("\nPick a mic index (Enter for default).")
    raw = input("> ").strip()
    if not raw:
        state.mic_device_index = None
        state.mic_device_name = "(default)"
        return

    try:
        idx = int(raw)
    except ValueError:
        state.last_error = "Invalid mic index; using default."
        state.mic_device_index = None
        state.mic_device_name = "(default)"
        return

    mic_dict = dict(mics)
    if idx not in mic_dict:
        state.last_error = "Mic index not found; using default."
        state.mic_device_index = None
        state.mic_device_name = "(default)"
        return

    state.mic_device_index = idx
    state.mic_device_name = mic_dict[idx]


class Speaker:
    def __init__(self, state: AppState):
        self.state = state
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
            )
            self._audio_stream.start()

            # Warmup: write a tiny bit of silence so the first real clip has no lag.
            silence = np.zeros((int(self._audio_rate * 0.03), self._audio_channels), dtype=np.int16)
            self._audio_stream.write(silence)
        return self._audio_stream

    def _play_pcm_int16(self, data: np.ndarray, samplerate: int) -> None:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        with self._audio_lock:
            stream = self._ensure_audio_stream(samplerate=samplerate, channels=int(data.shape[1]))
            stream.write(data.astype(np.int16, copy=False))

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
                    self._play_pcm_int16(data, rate)
                else:
                    self._play_pcm_int16_async(data, rate)
                return
            except Exception:
                # If playback fails, fall back to text-to-speech.
                break

        if fallback_text:
            self.say(fallback_text)

    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        speak_with_pocket_tts(text, voice_path=str(VOICE_TEMPLATE), speaker=self)

    def _say_sapi(self, text: str) -> None:
        self._sapi.say(text)
        self._sapi.runAndWait()

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
    t.stylize("bold #ff8c00")  # orange
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
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def contains_wake_phrase(s: str) -> bool:
    s_norm = normalize_text(s)
    return any(p in s_norm for p in WAKE_PHRASES)


def strip_wake_phrase(s: str) -> str:
    s_norm = normalize_text(s)
    for phrase in WAKE_PHRASES:
        if s_norm.startswith(phrase):
            return s_norm[len(phrase) :].strip(" ,:-")
    for phrase in WAKE_PHRASES:
        pat = rf"^\b{re.escape(phrase)}\b[\s,:\-]*"
        if re.match(pat, s_norm):
            return re.sub(pat, "", s_norm).strip()
    return s_norm


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
        lines.append(s)
    text = " ".join(lines).strip()
    if " Q:" in text:
        text = text.split(" Q:", 1)[0].strip()
    return text


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


def handle_command(state: AppState, speaker: Speaker, cmd: str, console: Console) -> None:
    cmd = normalize_text(cmd)
    cmd = expand_spoken_command(cmd)
    if not cmd:
        return

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

    if cmd in {"list mics", "list microphones"}:
        mics = list_mics()
        console.print("[bold]Microphones[/bold]")
        for i, name in mics:
            console.print(f"  {i}: {name}")
        return

    m = re.match(r"^(use mic|set mic)\s+(\d+)$", cmd)
    if m:
        idx = int(m.group(2))
        mic_dict = dict(list_mics())
        if idx not in mic_dict:
            speaker.speak_key("error", "That microphone index does not exist.")
            return
        state.mic_device_index = idx
        state.mic_device_name = mic_dict[idx]
        cfg = load_config()
        cfg["mic_device_index"] = idx
        save_config(cfg)
        speaker.speak_key("ok", "Microphone updated. Restart Jarvis.")
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
            console.print(Text(reply, style="dim"))
        return
    speak_with_pocket_tts(reply, voice_path=str(VOICE_TEMPLATE), speaker=speaker, console=console)


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
    text = (text or "").strip()
    if not text:
        return

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
            if console:
                print_assistant_message(console, text)
            done.wait(timeout=180)
            if ok_box[0] and out_path.is_file():
                speaker._play_wav(out_path)
            else:
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
        if console:
            print_assistant_message(console, text)

        for expect_i in range(len(chunks)):
            _idx, path = q.get()
            if path is None:
                speaker._say_sapi(" ".join(chunks[expect_i:]).strip() or text)
                return
            speaker._play_wav(path)

        return
    except Exception:
        pass

    if console:
        print_assistant_message(console, text)
    speaker._say_sapi(text)


def speech_worker(text_q: "queue.Queue[str]", stop_evt: threading.Event, state: AppState) -> None:
    whisper = create_whisper_model()

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

    def callback(indata, frames, time_info, status):
        if status:
            return
        if stop_evt.is_set():
            return
        mono = indata[:, 0].copy()
        state.mic_level = float(np.max(np.abs(mono))) if mono.size else 0.0
        try:
            audio_q.put_nowait(mono)
        except queue.Full:
            pass

    sr = 16000
    chunk_seconds = _WHISPER_CHUNK_SEC
    target_len = int(sr * chunk_seconds)
    buf = np.zeros((0,), dtype=np.float32)

    with sd.InputStream(
        samplerate=sr,
        dtype="float32",
        channels=1,
        callback=callback,
        device=state.mic_device_index,
    ):
        while not stop_evt.is_set():
            try:
                piece = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            buf = np.concatenate([buf, piece])
            if buf.shape[0] < target_len:
                continue

            chunk = buf[:target_len]
            buf = buf[target_len:]

            try:
                text = transcribe_audio_chunk(whisper, chunk)
                if text:
                    text_q.put(text)
            except Exception as e:
                state.last_error = str(e)
                time.sleep(0.1)


def ensure_whisper_warm(console: Console) -> None:
    """
    Trigger model download/cache early so the first command isn't slow.
    """
    wmodel = resolved_whisper_model_name()
    wdev, _ = whisper_device_options()
    console.print(f"[dim]Loading speech model {wmodel!r} ({wdev}) — first run may download…[/dim]")
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


def run_ui(connect: tuple[str, int] | None) -> None:
    configure_windows_utf8()
    console = Console()
    console.clear()
    console.print(orange_banner())
    console.print(Text("    -----  J.A.R.V.I.S.  -----", style="bold white"))
    console.print(Text("Just A Rather Very Intelligent System", style="dim"))
    console.print()

    state = AppState()
    cfg = load_config()
    state.mic_device_index = cfg.get("mic_device_index", None)
    if state.mic_device_index is not None:
        try:
            state.mic_device_name = sd.query_devices(state.mic_device_index).get("name", "")
        except Exception:
            state.mic_device_name = ""

    if connect is None and state.mic_device_index is None:
        pick_mic_interactive(console, state)
        cfg["mic_device_index"] = state.mic_device_index
        save_config(cfg)

    if connect is None:
        ensure_whisper_warm(console)

    speaker = Speaker(state)

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
    console.print(Text("Welcome, Sir.", style="bold white"))
    console.print(Text("How may I assist you today?", style="bold white"))
    speaker.speak_key("wake", "Welcome, sir. How may I assist you today?", blocking=False)
    console.print()
    console.print(Text("I can open websites, apps, type text, and answer questions via the local LLM.", style="dim"))
    console.print(Text("Say things like “launch notepad”, “google …”, or ask anything after wake.", style="dim"))
    console.print(Text("Commands: clear, exit, list mics, use mic <n>, test sound", style="dim"))
    if connect is not None:
        console.print(Text("Speech: connected (wake up to arm)", style="dim"))
    console.print()

    stop_evt = threading.Event()
    speech_q: "queue.Queue[str]" = queue.Queue()

    if connect is None:
        speech_thread = threading.Thread(target=speech_worker, args=(speech_q, stop_evt, state), daemon=True)
        speech_thread.start()
    else:
        host, port = connect
        try:
            s = socket.create_connection((host, port), timeout=10)
            t = threading.Thread(target=socket_reader_loop, args=(s, speech_q, stop_evt), daemon=True)
            t.start()
            # If we were launched by the daemon on a "wake up" phrase, the wake phrase
            # may have occurred before this UI connected. Auto-arm on connect.
            now = time.time()
            state.armed_until = now + ARM_SECONDS
            state.voice_session_until = now + VOICE_SESSION_SECONDS
        except Exception as e:
            console.print(Text(f"[error] Failed to connect to daemon: {e}", style="red"))
            console.print(Text("Starting local microphone mode instead.", style="dim"))
            connect = None
            ensure_whisper_warm(console)
            speech_thread = threading.Thread(target=speech_worker, args=(speech_q, stop_evt, state), daemon=True)
            speech_thread.start()

    def speech_loop():
        pending: list[str] = []
        pending_started_at = 0.0
        last_piece_at = 0.0
        last_final: str = ""
        last_final_at = 0.0

        def flush_pending() -> str:
            nonlocal pending, pending_started_at, last_piece_at
            if not pending:
                return ""
            phrase = " ".join(pending).strip()
            pending = []
            pending_started_at = 0.0
            last_piece_at = 0.0
            return phrase

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
            console.print()
            print_user_message(console, phrase, via="microphone")

        while not stop_evt.is_set():
            try:
                heard = speech_q.get(timeout=0.2)
            except queue.Empty:
                # if we have partials and there's been a pause, flush as one phrase
                now = time.time()
                if pending and (now - last_piece_at) > 0.65:
                    phrase = flush_pending()
                    if phrase:
                        phrase_norm = normalize_phrase(phrase)
                        # de-dupe rapid repeats
                        if phrase_norm == last_final and (now - last_final_at) < 1.2:
                            continue
                        last_final = phrase_norm
                        last_final_at = now
                        now2 = time.time()
                        woke2 = contains_wake_phrase(phrase_norm)
                        in_session2 = now2 < state.voice_session_until
                        armed2 = (now2 < state.armed_until) or in_session2
                        if woke2:
                            state.armed_until = now2 + ARM_SECONDS
                            state.voice_session_until = now2 + VOICE_SESSION_SECONDS
                            cmd2 = strip_wake_phrase(phrase_norm)
                            if not cmd2:
                                speaker.speak_key("confirm", "Yes, sir?")
                                continue
                        elif not armed2:
                            continue
                        else:
                            cmd2 = phrase_norm
                        log_voice_turn(phrase)
                        try:
                            handle_command(state, speaker, cmd2, console)
                            state.voice_session_until = time.time() + VOICE_SESSION_SECONDS
                        except SystemExit:
                            stop_evt.set()
                            return
                        except Exception as e:
                            state.last_error = str(e)
                            console.print(Text(f"[error] {e}", style="red"))
                continue

            now = time.time()
            # accumulate pieces into a single phrase
            piece = heard.strip()
            if piece:
                if not pending:
                    pending_started_at = now
                pending.append(piece)
                last_piece_at = now

            # safety: don't let a phrase run too long
            if pending and (now - pending_started_at) > 2.5:
                phrase = flush_pending()
            else:
                # wait for pause to flush
                phrase = ""

            if not phrase:
                continue

            phrase = flush_pending()
            phrase_norm = normalize_phrase(phrase)
            if phrase_norm == last_final and (now - last_final_at) < 1.2:
                continue
            last_final = phrase_norm
            last_final_at = now

            woke = contains_wake_phrase(phrase_norm)
            in_session = now < state.voice_session_until
            armed = (now < state.armed_until) or in_session
            if woke:
                state.armed_until = now + ARM_SECONDS
                state.voice_session_until = now + VOICE_SESSION_SECONDS
                cmd = strip_wake_phrase(phrase_norm)
                if not cmd:
                    speaker.speak_key("confirm", "Yes, sir?")
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
                return
            except Exception as e:
                state.last_error = str(e)
                console.print(Text(f"[error] {e}", style="red"))

    bg = threading.Thread(target=speech_loop, daemon=True)
    bg.start()

    try:
        while True:
            try:
                cmd = input(PROMPT)
            except (EOFError, KeyboardInterrupt):
                raise SystemExit(0)
            cmd_stripped = (cmd or "").strip()
            if cmd_stripped:
                console.print()
                print_user_message(console, cmd_stripped, via="keyboard")
            handle_command(state, speaker, cmd, console)
    except SystemExit:
        stop_evt.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="", help="Connect to daemon at host:port for speech input")
    args = ap.parse_args()

    connect = None
    if args.connect:
        host, port_s = args.connect.split(":", 1)
        connect = (host, int(port_s))

    # Migrate old root-level config if present
    old_cfg = ROOT / "jarvis_config.json"
    if old_cfg.exists() and not CONFIG_PATH.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        old_cfg.replace(CONFIG_PATH)

    run_ui(connect)


if __name__ == "__main__":
    main()

