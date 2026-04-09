from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if os.name == "nt":
    _VENV_PY = _ROOT / ".venv" / "Scripts" / "python.exe"
else:
    _VENV_PY = _ROOT / ".venv" / "bin" / "python"


def _reexec_in_venv_if_needed() -> None:
    """Running `python jarvis_daemon.py` with system Python misses deps; use .venv automatically."""
    if __name__ != "__main__":
        return
    if not _VENV_PY.is_file():
        return
    try:
        if Path(sys.executable).resolve() == _VENV_PY.resolve():
            return
    except OSError:
        return
    argv = [str(_VENV_PY), "-u", str(_ROOT / "jarvis_daemon.py"), *sys.argv[1:]]
    os.execv(str(_VENV_PY), argv)


_reexec_in_venv_if_needed()

ROOT = _ROOT

import queue
import socket
import subprocess
import threading
import time

try:
    import jarvis
except ModuleNotFoundError as e:
    if __name__ == "__main__":
        print(
            f"Missing dependency ({e}). Use the project venv:\n"
            f"  cd /d \"{ROOT}\"\n"
            f"  python -m venv .venv\n"
            f"  .\\.venv\\Scripts\\pip install -r requirements.txt\n"
            f"  .\\.venv\\Scripts\\python.exe jarvis_daemon.py\n",
            file=sys.stderr,
        )
    raise

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


def _resolve_ui_python_exe() -> Path | None:
    """
    Must be python.exe (real console). pythonw.exe has no console — Jarvis "runs" in Task Manager
    but no window appears.
    """
    p = ROOT / ".venv" / "Scripts" / "python.exe"
    if p.is_file():
        return p
    # Rare: venv only has py launcher; still avoid pythonw for UI.
    se = Path(sys.executable)
    if se.name.lower() == "python.exe" and se.is_file():
        return se
    return None


def launch_ui(port: int) -> subprocess.Popen | None:
    """Spawn Jarvis in a new visible console. Returns None if the process could not be started."""
    py_exe = _resolve_ui_python_exe()
    if py_exe is None:
        print(
            "ERROR: Need .venv\\Scripts\\python.exe to show the Jarvis window (not pythonw). "
            "Recreate venv: python -m venv .venv",
            file=sys.stderr,
        )
        return None
    jarvis_py = ROOT / "jarvis.py"
    if not jarvis_py.is_file():
        print(f"ERROR: Missing {jarvis_py}", file=sys.stderr)
        return None
    args = [str(py_exe), str(jarvis_py), "--connect", f"127.0.0.1:{port}"]
    if os.environ.get("JARVIS_VISUALIZE", "").strip().lower() in ("1", "true", "yes"):
        args.append("--visualize")
    kwargs: dict = {"cwd": str(ROOT)}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        # Nudge Windows to show the new console (some setups leave it backgrounded).
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = getattr(subprocess, "SW_SHOW", 5)  # Win32 SW_SHOW
        kwargs["startupinfo"] = si
    try:
        return subprocess.Popen(args, **kwargs)
    except OSError as e:
        print(f"ERROR: Could not start Jarvis UI: {e}", file=sys.stderr)
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
    jarvis.mic_utterances_to_queue(heard_q, stop_evt, whisper, state=None)


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

