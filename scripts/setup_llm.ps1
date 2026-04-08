param(
  [string]$ModelDir = ".\\models\\llm",
  [string]$ModelRepo = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
  [string]$ModelFile = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path ".\\bin\\llama" | Out-Null

Write-Host "1) Downloading llama.cpp Windows CPU build..."

$rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
$zipUrl = ($rel.assets | Where-Object { $_.name -like "*bin-win-cpu-x64.zip" } | Select-Object -First 1).browser_download_url
if (-not $zipUrl) { throw "Could not find a win-cpu-x64 zip in latest release assets." }

$zipPath = ".\\.cache\\llama-bin.zip"
New-Item -ItemType Directory -Force -Path ".\\.cache" | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri $zipUrl -OutFile $zipPath

Write-Host "2) Extracting llama.cpp binaries..."
$extractDir = ".\\.cache\\llama-bin"
Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue
Expand-Archive -Force $zipPath $extractDir

$cli = Get-ChildItem -Recurse $extractDir -Filter "llama-cli.exe" | Select-Object -First 1
if (-not $cli) { throw "llama-cli.exe not found in extracted archive." }

$cliDir = Split-Path -Parent $cli.FullName
Copy-Item -Force -Recurse (Join-Path $cliDir "*") ".\\bin\\llama\\"

Write-Host "3) Downloading Qwen2.5-1.5B-Instruct GGUF (Q4_K_M)..."
Write-Host "   If this fails with 401/403: accept the model license on Hugging Face, then run: huggingface-cli login"
Write-Host "   Or set `$env:HF_TOKEN before running this script."

& .\\.venv\\Scripts\\python.exe -c "from huggingface_hub import hf_hub_download; import pathlib; p=hf_hub_download(repo_id='$ModelRepo', filename='$ModelFile'); out=pathlib.Path(r'$ModelDir')/r'$ModelFile'; out.write_bytes(pathlib.Path(p).read_bytes()); print('Saved', out)"

Write-Host "Done."
