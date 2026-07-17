# AURORA dev launcher — starts the FastAPI backend in a new window, waits for
# it, then runs the Streamlit frontend in this one.
#
#   .\scripts\dev.ps1                 # default data dir (<repo>\data)
#   .\scripts\dev.ps1 -DataDir D:\x   # override AURORA_DATA_DIR
param(
    [string]$DataDir = "",
    [int]$ApiPort = 8000
)

$repo = Split-Path -Parent $PSScriptRoot
if ($DataDir -ne "") { $env:AURORA_DATA_DIR = $DataDir }
$env:AURORA_API_URL = "http://localhost:$ApiPort"

Write-Host "Starting backend on port $ApiPort..."
Start-Process python -ArgumentList "-m", "uvicorn", "main:app", "--app-dir", "$repo\backend", "--port", "$ApiPort" -WorkingDirectory $repo

$up = $false
foreach ($i in 1..30) {
    try {
        Invoke-RestMethod "http://localhost:$ApiPort/ping" -TimeoutSec 2 | Out-Null
        $up = $true; break
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $up) {
    Write-Error "Backend did not come up on port $ApiPort — check the uvicorn window."
    exit 1
}
Write-Host "Backend up — API docs: http://localhost:$ApiPort/docs"

streamlit run "$repo\frontend\app.py"
