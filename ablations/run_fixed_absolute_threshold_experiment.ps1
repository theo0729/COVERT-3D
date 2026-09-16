[CmdletBinding()]
param(
    [string]$OutputDir = (Join-Path $PSScriptRoot 'fixed_absolute_threshold_verification'),
    [string]$DatasetRoot = 'data\real3d_ad',
    [string]$MvtecRoot = 'data\mvtec_3d_ad_converted',
    [int]$SampleWorkers = 4,
    [int]$QueryWorkers = 1
)

$ErrorActionPreference = 'Stop'
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
[System.IO.Directory]::CreateDirectory($OutputDir) | Out-Null
$StatePath = Join-Path $OutputDir 'runner_state.json'
$LogPath = Join-Path $OutputDir 'formal_run.log'
$StartedAt = [DateTimeOffset]::UtcNow

@{
    status = 'running'
    pid = $PID
    started_at_utc = $StartedAt.ToString('o')
    command = 'fit, freeze, replay-check, 618 discovered defective, 598 held-out aggregation, 297 good'
} | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8

try {
    Push-Location (Split-Path $PSScriptRoot -Parent)
    try {
        & python `
            ablations\fixed_absolute_threshold_experiment.py run `
            --output-dir $OutputDir `
            --dataset-root $DatasetRoot `
            --mvtec-root $MvtecRoot `
            --sample-workers $SampleWorkers `
            --query-workers $QueryWorkers `
            --resume 2>&1 | Tee-Object -FilePath $LogPath -Append
        $ExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($ExitCode -ne 0) {
        throw "Experiment exited with code $ExitCode."
    }
    @{
        status = 'completed'
        pid = $PID
        started_at_utc = $StartedAt.ToString('o')
        completed_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
        exit_code = 0
        log_path = $LogPath
    } | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
}
catch {
    @{
        status = 'failed'
        pid = $PID
        started_at_utc = $StartedAt.ToString('o')
        failed_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
        exit_code = if ($null -eq $ExitCode) { 1 } else { $ExitCode }
        error = $_.Exception.Message
        log_path = $LogPath
    } | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
    throw
}
