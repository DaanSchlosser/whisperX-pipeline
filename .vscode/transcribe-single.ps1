# Transcribe a single audio file with all pipeline options.
# Invoked by the "2. Transcribe: Single File (all options)" VSCode task,
# which passes VSCode `${input:...}` values as positional arguments.
#
# Expected arguments (in order):
#   Audio filename inside Audio/
#   Whisper model name
#   Language code ('auto' to auto-detect)
#   Min speakers (empty to skip)
#   Max speakers (empty to skip)
#   Batch size
#   Compute type
#   Output directory

param(
    [Parameter(Mandatory=$true)][string]$AudioFile,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Language,
    [Parameter(Mandatory=$false)][string]$MinSpeakers = '',
    [Parameter(Mandatory=$false)][string]$MaxSpeakers = '',
    [Parameter(Mandatory=$true)][string]$BatchSize,
    [Parameter(Mandatory=$true)][string]$ComputeType,
    [Parameter(Mandatory=$true)][string]$OutputDir
)

if (-not (Test-Path '.env')) {
    Write-Error 'Missing .env file. Copy .env.example to .env and set HF_TOKEN first.'
    exit 1
}

$envText = Get-Content '.env' -Raw
if ($envText -notmatch '(?m)^HF_TOKEN\s*=\s*.+$') {
    Write-Error 'HF_TOKEN is missing in .env. Update .env before running transcription.'
    exit 1
}

$audioPath = Join-Path 'Audio' $AudioFile
if (-not (Test-Path $audioPath)) {
    Write-Error ('Audio file not found: ' + $audioPath)
    exit 1
}

$pyArgs = @(
    'transcribe.py',
    $audioPath,
    '--model', $Model,
    '--compute-type', $ComputeType,
    '--batch-size', $BatchSize,
    '--output-dir', $OutputDir
)
if ($Language -and $Language -ne 'auto') {
    $pyArgs += @('--language', $Language)
}
if ($MinSpeakers) {
    $pyArgs += @('--min-speakers', $MinSpeakers)
}
if ($MaxSpeakers) {
    $pyArgs += @('--max-speakers', $MaxSpeakers)
}

Write-Host ('Running: .venv/Scripts/python.exe ' + ($pyArgs -join ' '))
& '.venv/Scripts/python.exe' @pyArgs
exit $LASTEXITCODE
