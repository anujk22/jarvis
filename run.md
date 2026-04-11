One-time setup
    cd D:\Coding\Cursor\Jarvis
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
Normal run (terminal shows right away)
    cd D:\Coding\Cursor\Jarvis
    .\.venv\Scripts\python.exe .\jarvis.py
Silent background (say “wake up” to open the UI)
    cd D:\Coding\Cursor\Jarvis
    .\.venv\Scripts\pythonw.exe .\jarvis_daemon.py
With the full-screen visualizer
    cd D:\Coding\Cursor\Jarvis
    .\.venv\Scripts\python.exe .\jarvis.py --visualize
Use your real repo path if it isn’t D:\Coding\Cursor\Jarvis. If you meant how to run any command in PowerShell, say what you want to run and we can spell out the exact line.