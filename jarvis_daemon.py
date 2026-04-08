from __future__ import annotations

import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import sounddevice as sd
import numpy as np
import jarvis

ROOT = Path(__file__).resolve().parent

# Only one daemon per machine (this port must stay bound). Running 2+ daemons = many windows on one "wake up".
_DAEMON_LOCK_PORT = int(os.environ.get("JARVIS_DAEMON_LOCK_PORT", "38471"))


class _UiLaunchInProgress:
    """Reserved in child_proc so burst of 'wake up' transcripts only spawn one console."""


_UI_LAUNCHING = _UiLaunchInProgress()

# After one wake launch, ignore further launches for this long (Whisper may enqueue many "wake" lines;
# a child that exits instantly would otherwise spawn one window per queued line).
_WAKE_LAUNCH_COOLDOWN_SEC = float(os.environ.get("JARVIS_DAEMON_WAKE_COOLDOWN_SEC", "15"))


def start_server() -> tuple[socket.socket, int]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def launch_ui(port: int) -> subprocess.Popen | None:
    """Spawn Jarvis in a new console. Returns None if the process could not be started."""
    py_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py_exe.is_file():
        py_exe = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    args = [str(py_exe), str(ROOT / "jarvis.py"), "--connect", f"127.0.0.1:{port}"]
    kwargs: dict = {"cwd": str(ROOT)}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    try:
        return subprocess.Popen(args, **kwargs)
    except OSError:
        return None


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


def speech_loop(stop_evt: threading.Event, heard_q: "queue.Queue[str]") -> None:
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
        device=None,
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


def _acquire_singleton_lock() -> socket.socket:
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", _DAEMON_LOCK_PORT))
    except OSError:
        print(
            "\n*** Another Jarvis daemon is already running (or port "
            f"{_DAEMON_LOCK_PORT} is in use).\n"
            "*** STOP: only ONE jarvis_daemon should run. Extra copies each open a new window on wake.\n"
            "*** Kill stray Python: Task Manager -> Python, or in PowerShell:\n"
            "    Get-Process python*, pythonw* | Stop-Process -Force\n"
            "*** Then start a single: .\\.venv\\Scripts\\pythonw.exe .\\jarvis_daemon.py\n",
            file=sys.stderr,
        )
        sys.exit(1)
    lock.listen(1)
    return lock


def main() -> None:
    singleton_lock = _acquire_singleton_lock()
    srv, port = start_server()
    stop_evt = threading.Event()
    client_box: dict[str, socket.socket] = {}

    t_accept = threading.Thread(target=accept_client, args=(srv, client_box, stop_evt), daemon=True)
    t_accept.start()

    heard_q: "queue.Queue[str]" = queue.Queue()
    t_speech = threading.Thread(target=speech_loop, args=(stop_evt, heard_q), daemon=True)
    t_speech.start()

    child_proc: dict[str, subprocess.Popen | None | _UiLaunchInProgress] = {"p": None}
    last_wake_launch_at: float = 0.0

    def ui_alive() -> bool:
        p = child_proc["p"]
        if p is None:
            return False
        if p is _UI_LAUNCHING:
            return True
        if not isinstance(p, subprocess.Popen):
            child_proc["p"] = None
            return False
        if p.poll() is not None:
            child_proc["p"] = None
            return False
        return True

    def watch_ui(proc: subprocess.Popen) -> None:
        proc.wait()
        if child_proc.get("p") is proc:
            child_proc["p"] = None
        # UI died — drop socket so we are not stuck thinking a client is still connected.
        old = client_box.get("client")
        if old:
            try:
                old.close()
            except Exception:
                pass
            client_box.pop("client", None)

    try:
        while True:
            heard = heard_q.get()
            send_line(client_box, heard)

            now = time.monotonic()
            if jarvis.contains_wake_phrase(heard):
                in_cooldown = (now - last_wake_launch_at) < _WAKE_LAUNCH_COOLDOWN_SEC
                if not in_cooldown and not ui_alive():
                    last_wake_launch_at = now
                    child_proc["p"] = _UI_LAUNCHING
                    try:
                        time.sleep(0.05)
                        proc = launch_ui(port)
                        if proc:
                            child_proc["p"] = proc
                            threading.Thread(target=watch_ui, args=(proc,), daemon=True).start()
                            # Same utterance often produces many transcripts; don't handle them as separate wakes.
                            for _ in range(128):
                                try:
                                    extra = heard_q.get_nowait()
                                except queue.Empty:
                                    break
                                send_line(client_box, extra)
                        else:
                            child_proc["p"] = None
                    except Exception:
                        child_proc["p"] = None
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        try:
            srv.close()
        except Exception:
            pass
        try:
            singleton_lock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

