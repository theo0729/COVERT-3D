[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $ProjectRoot "ablations_main8_full_run_$Timestamp.log"
$Modes = @(
    "full",
    "laplacian_restoration",
    "wo_va_hks",
    "wo_structural_suppression",
    "wo_seed_growth",
    "wo_counterfactual_verification"
)
$TranscriptStarted = $false

Push-Location -LiteralPath $ProjectRoot
try {
    Start-Transcript -LiteralPath $LogPath | Out-Null
    $TranscriptStarted = $true

    foreach ($Mode in $Modes) {
        Write-Host "========================================"
        Write-Host "START ABLATION: $Mode"
        Write-Host "timestamp: $(Get-Date -Format 'o')"
        Write-Host "========================================"

        & python ablations/covert_ablation_batch.py `
            --ablation-mode $Mode `
            --sample-workers 4 `
            --query-workers 1 `
            --good-samples `
            --resume

        if ($LASTEXITCODE -ne 0) {
            throw "Ablation '$Mode' failed with exit code $LASTEXITCODE."
        }

        Write-Host "FINISH ABLATION: $Mode"
        Write-Host "timestamp: $(Get-Date -Format 'o')"
    }

    Write-Host "ALL ABLATIONS FINISHED"
    Write-Host "timestamp: $(Get-Date -Format 'o')"
}
finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
    Pop-Location
}
