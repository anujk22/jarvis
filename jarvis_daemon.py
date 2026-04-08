from __future__ import annotations

import queue
import socket
import subprocess
import threading
import time
from pathlib import Path

import sounddevice as sd
import numpy as np
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
    whisper = jarvis.create_whisper_model()

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

    def callback(indata, frames, time_info, status):
        if status:
            return
        mono = indata[:, 0].copy()
        try:
            audio_q.put_nowait(mono)
        except queue.Full:
            pass

    sr = 16000
    chunk_seconds = jarvis._WHISPER_CHUNK_SEC
    target_len = int(sr * chunk_seconds)
    buf = np.zeros((0,), dtype=np.float32)

    with sd.InputStream(
        samplerate=sr,
        dtype="float32",
        channels=1,
        callback=callback,
        device=mic_device_index,
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
                text = jarvis.transcribe_audio_chunk(whisper, chunk)
                if text:
                    heard_q.put(text)
            except Exception:
                time.sleep(0.1)


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

