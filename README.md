# Jarvis (Windows voice assistant)

Local, lightweight “Jarvis”-style assistant that:

- Listens to your microphone (offline speech-to-text via Vosk)
- Uses your voice model (Piper in Python) for speech output
- Executes simple desktop actions (open sites/apps, type text)
- Prints recognized speech to the terminal once you say the wake phrase

## Layout

- Voice model (Piper): `jarvis/en/en_GB/jarvis/high/` (or `jarvis/.../medium/`)
  - Jarvis will prefer `jarvis-high.onnx` if present, otherwise `jarvis-medium.onnx`
- Speech-to-text (Vosk): `models/vosk/model`
  - Automatically downloaded on first run
- Config: `config/jarvis_config.json`
  - Stores which mic index you selected

If you already have the `jarvis/` folder, you are good.

## Setup (one time)

1. Create venv + install dependencies
```powershell
cd d:\Coding\Cursor\Jarvis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) If `jarvis/en/en_GB/jarvis/high/` is missing
The repo includes a tiny downloader that writes files into whatever `--out` path you give it.

Example (downloads “high” model into the expected folder structure):
```powershell
python .\scripts\download_voice.py --repo jgkawell/jarvis --filename en/en_GB/jarvis/high/jarvis-high.onnx --out jarvis\en\en_GB\jarvis\high\jarvis-high.onnx
python .\scripts\download_voice.py --repo jgkawell/jarvis --filename en/en_GB/jarvis/high/jarvis-high.onnx.json --out jarvis\en\en_GB\jarvis\high\jarvis-high.onnx.json
```

Model source: `https://huggingface.co/jgkawell/jarvis`

## Run

### Normal mode (terminal appears immediately)
```powershell
cd d:\Coding\Cursor\Jarvis
.\.venv\Scripts\python.exe .\jarvis.py
```

### “Perfect world” mode (silent until wake up)
```powershell
cd d:\Coding\Cursor\Jarvis
.\.venv\Scripts\pythonw.exe .\jarvis_daemon.py
```

Note: the silent daemon uses the saved mic index from `jarvis_config.json`. If the wrong mic is selected, run normal mode once, then use `use mic <n>`.

## Usage

1. Say **“wake up”** (or “jarvis”) to arm Jarvis for ~8 seconds.
2. Then say a command, for example:

  - “open youtube”
  - “go to https://news.ycombinator.com”
  - “search for pizza”
  - “open notepad”
  - “type hello world”
  - “exit”

### Commands (typed or voice)

- `test sound` (spoken check)
- `list mics`
- `use mic <n>` (re-select mic index; restart recommended)
- `clear`
- `exit`

## Troubleshooting (no sound)

1. Run `test sound`
2. If you still hear nothing:
   - verify Windows output volume is not muted
   - check that `./.cache/tts.wav` is larger than ~44 bytes

