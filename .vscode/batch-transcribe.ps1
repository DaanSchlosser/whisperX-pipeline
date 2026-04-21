# Batch transcribe all audio files in Audio/ directory
# Collects speaker counts for all files first, then processes sequentially.

$audioExtensions = '.wav', '.mp3', '.flac', '.ogg', '.m4a'

# Validate .env exists and has HF_TOKEN
if (-not (Test-Path '.env')) {
    Write-Error 'Missing .env file. Copy .env.example to .env and set HF_TOKEN first.'
    exit 1
}

$envText = Get-Content '.env' -Raw
if ($envText -notmatch '(?m)^HF_TOKEN\s*=\s*.+$') {
    Write-Error 'HF_TOKEN is missing in .env. Update .env before running transcription.'
    exit 1
}

# Get all audio files
$audioFiles = Get-ChildItem -Path 'Audio' -File |
    Where-Object { $audioExtensions -contains $_.Extension.ToLowerInvariant() } |
    Sort-Object Name

if (-not $audioFiles) {
    Write-Host 'No audio files found in Audio/' -ForegroundColor Yellow
    exit 0
}

# Phase 1 - collect shared options, then per-file speaker inputs
Write-Host "`nFound $($audioFiles.Count) file(s)." -ForegroundColor Cyan

$batchSizeInput = Read-Host 'Batch size? (press Enter for default 8, lower if GPU runs out of memory)'
$batchSize = if ($batchSizeInput) { $batchSizeInput } else { '8' }

Write-Host "`nEnter speaker counts per file (press Enter to auto-detect):`n" -ForegroundColor Cyan

$speakerArgs = @{}

foreach ($file in $audioFiles) {
    $response = Read-Host "  $($file.Name) - speakers? (N or 'min max')"

    if ($response) {
        $parts = $response -split '\s+' | Where-Object { $_ }
        if ($parts.Count -eq 1) {
            $speakerArgs[$file.FullName] = @('--min-speakers', $parts[0], '--max-speakers', $parts[0])
        } elseif ($parts.Count -eq 2) {
            $speakerArgs[$file.FullName] = @('--min-speakers', $parts[0], '--max-speakers', $parts[1])
        } else {
            Write-Error "Invalid input for '$($file.Name)'. Use one number or 'min max'."
            exit 1
        }
    } else {
        $speakerArgs[$file.FullName] = @()
    }
}

# Phase 2 - process each file
Write-Host "`nStarting transcription...`n" -ForegroundColor Cyan

foreach ($file in $audioFiles) {
    Write-Host "--- $($file.Name) ---" -ForegroundColor Cyan

    $pyArgs = @('transcribe.py', $file.FullName, '--batch-size', $batchSize) + $speakerArgs[$file.FullName]

    & '.venv\Scripts\python.exe' @pyArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "`nAll files processed." -ForegroundColor Green
