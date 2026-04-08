from __future__ import annotations

import queue
import socket
import subprocess
import threading
import time
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model

import jarvis

ROOT = Path(__file__).resolve().parent


def start_server() -> tuple[socket.socket, int]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def launch_ui(port: int) -> None:
    # Opens a new visible terminal window.
    cmd = f"cd /d \"{ROOT}\"; .\\.venv\\Scripts\\python.exe .\\jarvis.py --connect 127.0.0.1:{port}"
    subprocess.Popen(
        ["cmd", "/c", "start", "", "powershell", "-NoExit", "-Command", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def accept_client(srv: socket.socket, client_box: dict[str, socket.socket], stop_evt: threading.Event) -> None:
    while not stop_evt.is_set():
        try:
            c, _ = srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # replace existing client if any
            old = client_box.get("client")
            if old:
                try:
                    old.close()
                except Exception:
                    pass
            client_box["client"] = c
        except Exception:
            time.sleep(0.2)


def send_line(client_box: dict[str, socket.socket], text: str) -> None:
    c = client_box.get("client")
    if not c:
        return
    try:
        c.sendall((text.strip() + "\n").encode("utf-8", errors="ignore"))
    except Exception:
        try:
            c.close()
        except Exception:
            pass
        client_box.pop("client", None)


def speech_loop(stop_evt: threading.Event, heard_q: "queue.Queue[str]", mic_device_index: int | None) -> None:
    jarvis.ensure_vosk_model_silent()
    model = Model(str(jarvis.VOSK_MODEL_DIR))
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(False)

    def callback(indata, frames, time_info, status):
        if status:
            return
        if rec.AcceptWaveform(bytes(indata)):
            try:
                import json

                res = json.loads(rec.Result())
                txt = (res.get("text") or "").strip()
                if txt:
                    heard_q.put(txt)
            except Exception:
                return

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
        device=mic_device_index,
    ):
        while not stop_evt.is_set():
            time.sleep(0.05)


def main() -> None:
    cfg = jarvis.load_config()
    mic_device_index = cfg.get("mic_device_index", None)

    srv, port = start_server()
    stop_evt = threading.Event()
    client_box: dict[str, socket.socket] = {}

    t_accept = threading.Thread(target=accept_client, args=(srv, client_box, stop_evt), daemon=True)
    t_accept.start()

    heard_q: "queue.Queue[str]" = queue.Queue()
    t_speech = threading.Thread(target=speech_loop, args=(stop_evt, heard_q, mic_device_index), daemon=True)
    t_speech.start()

    ui_running = False
    try:
        while True:
            heard = heard_q.get()
            send_line(client_box, heard)

            if jarvis.contains_wake_phrase(heard) and not ui_running:
                ui_running = True
                launch_ui(port)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        try:
            srv.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

