param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $icon = Join-Path $PSScriptRoot 'transcriber.ico'
    & $Python -m PyInstaller --noconfirm --onefile --windowed --noupx --name Transcriber `
        --icon $icon --add-data ($icon + ':.') --distpath dist --workpath build --specpath build transcriber.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
