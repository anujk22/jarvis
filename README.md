# Jarvis (Windows voice assistant)

Local, lightweight desktop assistant that:
- Listens to your microphone (offline speech-to-text via Vosk)
- Runs simple Windows desktop tasks (open a browser URL, open apps, type text)
- Shows a Jarvis-like terminal greeting and prompt
- Speaks back using **either** Windows SAPI (default) **or** an optional local Piper voice model


## Setup (Windows)

### 1) Create venv + install deps
```powershell
cd d:\Coding\Cursor\Jarvis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) (Optional) Download the Jarvis-style Piper voice
This project supports running Piper **directly in Python** (via `piper-tts`) and downloading the
Hugging Face voice model into `models\piper\`.

Run:
```powershell
python .\scripts\download_voice.py --repo jgkawell/jarvis --filename en/en_GB/jarvis/medium/jarvis-medium.onnx --out models\piper\jarvis-medium.onnx
python .\scripts\download_voice.py --repo jgkawell/jarvis --filename en/en_GB/jarvis/medium/jarvis-medium.onnx.json --out models\piper\jarvis-medium.onnx.json
```

Model source: `https://huggingface.co/jgkawell/jarvis`

### 3) Run
```powershell
python .\jarvis.py
```

## Voice commands (examples)
- Say “wake up” then:
  - “open youtube”
  - “go to https://news.ycombinator.com”
  - “search for best lightweight speech recognition”
  - “open notepad”
  - “type hello world”
  - “exit”

## Notes
- If Piper isn’t configured, it will fall back to Windows SAPI voices (pyttsx3).
- For safety, typing uses the active window; say “open notepad” first if you want a safe target.

