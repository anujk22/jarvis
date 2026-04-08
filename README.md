# Jarvis (Windows voice assistant)

Local Jarvis-style assistant that:

- **Speech-to-text**: Whisper (`faster-whisper`, `tiny.en`) on CPU (better than Vosk, still lightweight)
- **Speech output**: your **prerecorded** `voices/*.wav` clips (SAPI fallback only if a clip is missing)
- **Actions**: open websites/apps, search, type text

## Setup (one time)

```powershell
cd d:\Coding\Cursor\Jarvis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

### Normal (terminal appears immediately)
```powershell
cd d:\Coding\Cursor\Jarvis
.\.venv\Scripts\python.exe .\jarvis.py
```

### Silent daemon (say “wake up” to pop the terminal)
```powershell
cd d:\Coding\Cursor\Jarvis
.\.venv\Scripts\pythonw.exe .\jarvis_daemon.py
```

## Optional: blazing-fast local LLM (1B) + voice-cloned replies

This uses:
- **LLM**: `TinyLlama-1.1B-Chat` (GGUF `Q4_K_M`) via **prebuilt llama.cpp** binaries (no compiler needed)
- **TTS**: `pocket-tts` voice cloning using your `voices/template.wav`

One-time setup:

```powershell
cd d:\Coding\Cursor\Jarvis
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\scripts\setup_llm.ps1
```

Usage:
- Type or say: `ask what is 2+2`
- Type or say: `chat tell me a fun fact`

## Voice usage

1. Say **“wake up”** (or “jarvis”) to start a voice session.
2. Then say commands like:
   - “open google dot com”
   - “search for pizza”
   - “open notepad”
   - “exit”

Typed commands also work at the prompt:
`list mics`, `use mic <n>`, `test sound`, `sleep`, `exit`

## Voice clips

Jarvis plays `voices/<key>.wav` when available. Common keys:
`wake`, `confirm`, `opening`, `searching`, `done`, `goodbye`, `didnt_understand`, `error`, `ok`

## Notes
- For safety, typing uses the active window; say “open notepad” first if you want a safe target.

