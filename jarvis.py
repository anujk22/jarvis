from __future__ import annotations

import argparse
import json
import queue
import re
import socket
import threading
import time
import wave
import webbrowser
import winsound
from dataclasses import dataclass
from pathlib import Path

import pyautogui
import pyttsx3
import sounddevice as sd
from rich.console import Console
from rich.text import Text
from vosk import KaldiRecognizer, Model

APP_NAME = "J.A.R.V.I.S."
PROMPT = "J.A.R.V.I.S. > "
WAKE_PHRASES = ("wake up", "jarvis")
ARM_SECONDS = 8

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "jarvis_config.json"

DEFAULT_PIPER_MODEL = ROOT / "models" / "piper" / "jarvis-medium.onnx"
DEFAULT_PIPER_MODEL_CONFIG = ROOT / "models" / "piper" / "jarvis-medium.onnx.json"

VOSK_MODEL_DIR = ROOT / "models" / "vosk" / "model"

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.03


@dataclass
class AppState:
    armed_until: float = 0.0
    mic_device_index: int | None = None
    mic_device_name: str = ""
    mic_level: float = 0.0
    last_error: str = ""
    tts_mode: str = "sapi"  # "sapi" or "piper"


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

        self._piper_voice = None
        model_path, cfg_path = pick_local_piper_model()
        if model_path.exists() and cfg_path.exists():
            try:
                from piper.voice import PiperVoice  # type: ignore

                self._piper_voice = PiperVoice.load(str(model_path))
                self.state.tts_mode = "piper"
            except Exception:
                self._piper_voice = None

    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.state.tts_mode == "piper" and self._piper_voice is not None:
            self._say_piper(text)
        else:
            self._say_sapi(text)

    def _say_sapi(self, text: str) -> None:
        self._sapi.say(text)
        self._sapi.runAndWait()

    def _say_piper(self, text: str) -> None:
        tmp = ROOT / ".cache"
        tmp.mkdir(exist_ok=True)
        out_wav = tmp / "tts.wav"
        try:
            if self._piper_voice is None:
                raise RuntimeError("Piper voice not loaded")
            # piper-tts returns AudioChunk(s); we must write the WAV ourselves.
            with wave.open(str(out_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(self._piper_voice.config.sample_rate))
                for chunk in self._piper_voice.synthesize(text):
                    wf.writeframes(chunk.audio_int16_bytes)
            try:
                winsound.PlaySound(str(out_wav), winsound.SND_FILENAME | winsound.SND_SYNC)
            except Exception:
                # Fallback: PowerShell SoundPlayer
                import subprocess

                subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        f"(New-Object Media.SoundPlayer '{out_wav}').PlaySync();",
                    ],
                    check=False,
                    capture_output=True,
                )
        except Exception as e:
            self.state.tts_mode = "sapi"
            self.state.last_error = f"Piper failed; falling back to SAPI: {e}"
            self._say_sapi(text)


def orange_banner() -> Text:
    banner_path = ROOT / "jarvis.txt"
    txt = banner_path.read_text(encoding="utf-8", errors="replace").rstrip("\n") if banner_path.exists() else "JARVIS"
    t = Text(txt)
    t.stylize("bold #ff8c00")  # orange
    return t


def pick_local_piper_model() -> tuple[Path, Path]:
    """
    Prefer the user's local model inside ./jarvis/... if present.
    Fallback to ./models/piper.
    """
    preferred = ROOT / "jarvis" / "en" / "en_GB" / "jarvis" / "high" / "jarvis-high.onnx"
    preferred_cfg = preferred.with_suffix(preferred.suffix + ".json")
    if preferred.exists() and preferred_cfg.exists():
        return preferred, preferred_cfg

    preferred2 = ROOT / "jarvis" / "en" / "en_GB" / "jarvis" / "medium" / "jarvis-medium.onnx"
    preferred2_cfg = preferred2.with_suffix(preferred2.suffix + ".json")
    if preferred2.exists() and preferred2_cfg.exists():
        return preferred2, preferred2_cfg

    return DEFAULT_PIPER_MODEL, DEFAULT_PIPER_MODEL_CONFIG


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


def handle_command(state: AppState, speaker: Speaker, cmd: str, console: Console) -> None:
    cmd = normalize_text(cmd)
    if not cmd:
        return

    if cmd in {"exit", "quit"}:
        speaker.say("Goodbye, sir.")
        raise SystemExit(0)

    if cmd in {"clear", "cls"}:
        console.clear()
        console.print(orange_banner())
        console.print()
        return

    if cmd in {"test sound", "test audio"}:
        speaker.say("Welcome, sir. How may I assist you today?")
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
            speaker.say("That microphone index does not exist.")
            return
        state.mic_device_index = idx
        state.mic_device_name = mic_dict[idx]
        cfg = load_config()
        cfg["mic_device_index"] = idx
        save_config(cfg)
        speaker.say("Microphone updated. Restart Jarvis.")
        return

    m = re.match(r"^(open)\s+(.+)$", cmd)
    if m:
        target = m.group(2).strip()
        quick_sites = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "gmail": "https://mail.google.com",
            "github": "https://github.com",
        }
        if target in quick_sites:
            open_url(quick_sites[target])
            speaker.say(f"Opening {target}.")
            return
        if re.search(r"\.|https?://", target):
            open_url(target)
            speaker.say("Opening it.")
            return
        open_app(target)
        speaker.say(f"Opening {target}.")
        return

    m = re.match(r"^(go to|navigate to)\s+(.+)$", cmd)
    if m:
        open_url(m.group(2).strip())
        speaker.say("Done.")
        return

    m = re.match(r"^(search for)\s+(.+)$", cmd)
    if m:
        open_search(m.group(2).strip())
        speaker.say("Searching.")
        return

    m = re.match(r"^(type)\s+(.+)$", cmd)
    if m:
        type_text(m.group(2))
        speaker.say("Done.")
        return

    speaker.say("Sorry, I did not understand that.")


def speech_worker(text_q: "queue.Queue[str]", stop_evt: threading.Event, state: AppState) -> None:
    if not (VOSK_MODEL_DIR.exists() and any(VOSK_MODEL_DIR.iterdir())):
        state.last_error = "Missing Vosk model. Run jarvis once to auto-download it (or re-run setup)."
        return

    model = Model(str(VOSK_MODEL_DIR))
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(False)

    def callback(indata, frames, time_info, status):
        if status:
            return
        try:
            import array

            a = array.array("h", bytes(indata))
            if len(a) > 0:
                peak = max(abs(x) for x in a)
                state.mic_level = min(1.0, peak / 32768.0)
        except Exception:
            pass

        if rec.AcceptWaveform(bytes(indata)):
            try:
                res = json.loads(rec.Result())
                txt = (res.get("text") or "").strip()
                if txt:
                    text_q.put(txt)
            except Exception:
                return

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
        device=state.mic_device_index,
    ):
        while not stop_evt.is_set():
            time.sleep(0.05)


def ensure_vosk_model_auto(console: Console) -> None:
    # lightweight auto-install (same as earlier script, but inline)
    if VOSK_MODEL_DIR.exists() and any(VOSK_MODEL_DIR.iterdir()):
        return

    console.print("[dim]Downloading offline speech model (first run)…[/dim]")
    import zipfile
    from io import BytesIO

    import requests

    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(BytesIO(r.content))
    tmp = ROOT / ".cache" / "vosk_extract"
    tmp.mkdir(parents=True, exist_ok=True)
    z.extractall(tmp)
    top_dirs = [p for p in tmp.iterdir() if p.is_dir()]
    if not top_dirs:
        raise RuntimeError("Unexpected Vosk zip layout")
    model_root = top_dirs[0]
    VOSK_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if VOSK_MODEL_DIR.exists():
        # best-effort clear
        for p in sorted(VOSK_MODEL_DIR.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                try:
                    p.rmdir()
                except OSError:
                    pass
    model_root.replace(VOSK_MODEL_DIR)


def ensure_vosk_model_silent() -> None:
    if VOSK_MODEL_DIR.exists() and any(VOSK_MODEL_DIR.iterdir()):
        return
    import zipfile
    from io import BytesIO

    import requests

    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(BytesIO(r.content))
    tmp = ROOT / ".cache" / "vosk_extract"
    tmp.mkdir(parents=True, exist_ok=True)
    z.extractall(tmp)
    top_dirs = [p for p in tmp.iterdir() if p.is_dir()]
    if not top_dirs:
        raise RuntimeError("Unexpected Vosk zip layout")
    model_root = top_dirs[0]
    VOSK_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if VOSK_MODEL_DIR.exists():
        for p in sorted(VOSK_MODEL_DIR.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                try:
                    p.rmdir()
                except OSError:
                    pass
    model_root.replace(VOSK_MODEL_DIR)


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
        ensure_vosk_model_auto(console)

    speaker = Speaker(state)
    speaker.say("Welcome, sir. How may I assist you today?")
    console.print(Text("Welcome, Sir.", style="bold white"))
    console.print(Text("How may I assist you today?", style="bold white"))
    console.print()
    console.print(Text("I can open websites, apps, and type text.", style="dim"))
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
        except Exception as e:
            console.print(Text(f"[error] Failed to connect to daemon: {e}", style="red"))
            console.print(Text("Starting local microphone mode instead.", style="dim"))
            connect = None
            ensure_vosk_model_auto(console)
            speech_thread = threading.Thread(target=speech_worker, args=(speech_q, stop_evt, state), daemon=True)
            speech_thread.start()

    def speech_loop():
        while not stop_evt.is_set():
            try:
                heard = speech_q.get(timeout=0.2)
            except queue.Empty:
                continue

            now = time.time()
            woke = contains_wake_phrase(heard)
            armed = now < state.armed_until
            if woke:
                state.armed_until = now + ARM_SECONDS
                cmd = strip_wake_phrase(heard)
                if not cmd:
                    speaker.say("Yes, sir?")
                    continue
            elif not armed:
                continue
            else:
                cmd = normalize_text(heard)

            console.print(Text(f"[heard] {heard}", style="dim"))
            try:
                handle_command(state, speaker, cmd, console)
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

