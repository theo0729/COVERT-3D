[CmdletBinding()]
param(
    [switch]$Smoke
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SensitivityRoot = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $SensitivityRoot
$ValidationRoot = Join-Path $SensitivityRoot "validation"
$FingerprintScript = Join-Path $ValidationRoot "print_fingerprint.py"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SampleWorkers = 4
$QueryWorkers = 1
$SmokeSampleKey = "fish/275_bulge"
$SmokeRows = @()
$ExitCode = 0
$TranscriptStarted = $false

$Experiments = @(
    [PSCustomObject]@{
        Name = "01_graph_geometry_coupling"
        Entrypoint = "sensitivity_analysis/01_graph_geometry_coupling/run_sensitivity.py"
        SmokeConfig = "local_k14_L12-24-48"
    },
    [PSCustomObject]@{
        Name = "02_vahks_modulation"
        Entrypoint = "sensitivity_analysis/02_vahks_modulation/run_sensitivity.py"
        SmokeConfig = "gamma_0"
    },
    [PSCustomObject]@{
        Name = "03_ac_boundary"
        Entrypoint = "sensitivity_analysis/03_ac_boundary/run_sensitivity.py"
        SmokeConfig = "ac_90"
    },
    [PSCustomObject]@{
        Name = "04_candidate_recovery"
        Entrypoint = "sensitivity_analysis/04_candidate_recovery/run_sensitivity.py"
        SmokeConfig = "tauQ_030_H5"
    },
    [PSCustomObject]@{
        Name = "05_gtr_local_calibration"
        Entrypoint = "sensitivity_analysis/05_gtr_local_calibration/run_sensitivity.py"
        SmokeConfig = "tauRel_025"
    }
)

function Get-ImplementationFingerprint {
    $Raw = & python $FingerprintScript
    $NativeExitCode = $LASTEXITCODE
    if ($NativeExitCode -ne 0) {
        throw "Fingerprint helper failed with exit code $NativeExitCode"
    }
    return (($Raw -join [Environment]::NewLine) | ConvertFrom-Json)
}

function Write-SmokeSummary {
    if (-not $Smoke) {
        return
    }
    $CsvPath = Join-Path $ValidationRoot "PS1_SMOKE_SUMMARY.csv"
    $JsonPath = Join-Path $ValidationRoot "PS1_SMOKE_SUMMARY.json"
    $SmokeRows | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Encoding UTF8
    ConvertTo-Json -InputObject @($SmokeRows) -Depth 20 | Set-Content -LiteralPath $JsonPath -Encoding UTF8
}

if ($Smoke) {
    $TranscriptRoot = Join-Path $ValidationRoot "ps1_smoke"
    $TranscriptName = "sensitivity_smoke_run_$Timestamp.log"
}
else {
    $TranscriptRoot = Join-Path $SensitivityRoot "logs"
    $TranscriptName = "sensitivity_main8_run_$Timestamp.log"
}
New-Item -ItemType Directory -Path $TranscriptRoot -Force | Out-Null
$TranscriptPath = Join-Path $TranscriptRoot $TranscriptName

Push-Location $ProjectRoot
try {
    Start-Transcript -LiteralPath $TranscriptPath | Out-Null
    $TranscriptStarted = $true

    $FingerprintBefore = Get-ImplementationFingerprint
    if ([int]$FingerprintBefore.implementation_fingerprint_schema -ne 2) {
        throw "Expected implementation_fingerprint_schema=2"
    }
    $RequiredDependencies = @(
        "vast/data/io.py",
        "vast/data/preprocess.py",
        "vast/evaluation/scoring.py",
        "vast/features/hks.py",
        "vast/features/surface_variation.py",
        "vast/graph/edge_weight.py",
        "vast/graph/laplacian.py",
        "vast/graph/mutual_knn.py"
    )
    foreach ($Dependency in $RequiredDependencies) {
        if ($FingerprintBefore.implementation_files -notcontains $Dependency) {
            throw "Implementation fingerprint is missing required dependency: $Dependency"
        }
    }
    Write-Host "IMPLEMENTATION FINGERPRINT: $($FingerprintBefore.implementation_hash)"
    Write-Host "IMPLEMENTATION SCHEMA: $($FingerprintBefore.implementation_fingerprint_schema)"
    Write-Host "IMPLEMENTATION FILE COUNT: $($FingerprintBefore.implementation_file_count)"
    Write-Host "MODE: $(if ($Smoke) { 'SMOKE' } else { 'FORMAL MAIN8' })"

    foreach ($Experiment in $Experiments) {
        $StartedAt = Get-Date -Format "o"
        Write-Host "====================================="
        Write-Host "START SENSITIVITY: $($Experiment.Name)"
        Write-Host "timestamp: $StartedAt"
        Write-Host "====================================="

        $Entrypoint = Join-Path $ProjectRoot $Experiment.Entrypoint
        $OutputRoot = $null
        $CheckpointPath = $null
        $CheckpointHashBefore = $null
        $CompletedBefore = $false

        if ($Smoke) {
            $OutputRoot = Join-Path $TranscriptRoot $Experiment.Name
            $CheckpointPath = Join-Path $OutputRoot "raw/$($Experiment.SmokeConfig)/samples/real3d/fish/275_bulge/metrics.json"
            $CompletedPath = Join-Path $OutputRoot "raw/$($Experiment.SmokeConfig)/completed.json"
            $CompletedBefore = Test-Path -LiteralPath $CompletedPath
            if (Test-Path -LiteralPath $CheckpointPath) {
                $CheckpointHashBefore = (Get-FileHash -LiteralPath $CheckpointPath -Algorithm SHA256).Hash
            }
            & python $Entrypoint `
                --quick `
                --only $Experiment.SmokeConfig `
                --categories fish `
                --sample-key $SmokeSampleKey `
                --output-root $OutputRoot `
                --sample-workers $SampleWorkers `
                --query-workers $QueryWorkers `
                --fail-fast
        }
        else {
            & python $Entrypoint `
                --sample-workers $SampleWorkers `
                --query-workers $QueryWorkers `
                --fail-fast
        }

        $ExperimentExitCode = $LASTEXITCODE
        if ($ExperimentExitCode -ne 0) {
            $FailedAt = Get-Date -Format "o"
            Write-Host "FAILED SENSITIVITY: $($Experiment.Name); exit_code=$ExperimentExitCode; timestamp=$FailedAt" -ForegroundColor Red
            if ($Smoke) {
                $SmokeRows += [PSCustomObject]@{
                    experiment = $Experiment.Name
                    config_id = $Experiment.SmokeConfig
                    sample_key = $SmokeSampleKey
                    status = "failure"
                    exit_code = $ExperimentExitCode
                    requested_parameters = ""
                    effective_parameters = ""
                    predicted_points = ""
                    precision = ""
                    recall = ""
                    f1 = ""
                    iou = ""
                    implementation_hash = $FingerprintBefore.implementation_hash
                    output_path = $OutputRoot
                    error = "experiment process failed"
                    checkpoint_reused = $false
                }
                Write-SmokeSummary
            }
            $ExitCode = $ExperimentExitCode
            break
        }

        if ($Smoke) {
            try {
                $ConfigPath = Join-Path $OutputRoot "raw/$($Experiment.SmokeConfig)/config.json"
                $MetricsPath = Join-Path $OutputRoot "raw/$($Experiment.SmokeConfig)/sample_metrics.json"
                $DiagnosticsPath = Join-Path $OutputRoot "raw/$($Experiment.SmokeConfig)/stage_diagnostics.json"
                foreach ($RequiredOutput in @($ConfigPath, $MetricsPath, $DiagnosticsPath, $CheckpointPath)) {
                    if (-not (Test-Path -LiteralPath $RequiredOutput)) {
                        throw "Missing smoke output: $RequiredOutput"
                    }
                }
                $Config = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
                $Metric = @((Get-Content -Raw -LiteralPath $MetricsPath | ConvertFrom-Json))[0]
                $Diagnostics = @((Get-Content -Raw -LiteralPath $DiagnosticsPath | ConvertFrom-Json))[0]
                if ($Metric.category -ne "fish" -or $Metric.sample_id -ne "275_bulge") {
                    throw "Smoke output used an unexpected sample"
                }
                if ($Metric.config_id -ne $Experiment.SmokeConfig -or $Config.config_id -ne $Experiment.SmokeConfig) {
                    throw "Smoke output used an unexpected configuration"
                }
                if ($Metric.status -ne "success") {
                    throw "Smoke sample status is not success"
                }
                if ([int]$Config.implementation_fingerprint_schema -ne 2) {
                    throw "Smoke config did not record fingerprint schema 2"
                }
                if ([int]$Config.implementation_file_count -ne [int]$FingerprintBefore.implementation_file_count) {
                    throw "Smoke config recorded the wrong implementation file count"
                }
                if ($Config.implementation_hash -ne $FingerprintBefore.implementation_hash) {
                    throw "Smoke config recorded the wrong implementation hash"
                }
                if ($null -eq $Diagnostics.final_positive_point_count) {
                    throw "Smoke stage diagnostics are incomplete"
                }

                $CheckpointHashAfter = (Get-FileHash -LiteralPath $CheckpointPath -Algorithm SHA256).Hash
                $CheckpointReused = [bool](
                    $CompletedBefore -and
                    $null -ne $CheckpointHashBefore -and
                    $CheckpointHashBefore -eq $CheckpointHashAfter
                )
                $SmokeRows += [PSCustomObject]@{
                    experiment = $Experiment.Name
                    config_id = $Experiment.SmokeConfig
                    sample_key = $SmokeSampleKey
                    status = "success"
                    exit_code = 0
                    requested_parameters = ($Config.requested_values | ConvertTo-Json -Compress -Depth 10)
                    effective_parameters = $Metric.effective_parameters_json
                    predicted_points = $Metric.predicted_positive_points
                    precision = $Metric.precision
                    recall = $Metric.recall
                    f1 = $Metric.f1
                    iou = $Metric.iou
                    implementation_hash = $Metric.implementation_hash
                    output_path = $OutputRoot
                    error = ""
                    checkpoint_reused = $CheckpointReused
                }
                Write-Host "SMOKE CHECKPOINT REUSED: $($Experiment.Name) = $CheckpointReused"
                Write-SmokeSummary
            }
            catch {
                $SmokeRows += [PSCustomObject]@{
                    experiment = $Experiment.Name
                    config_id = $Experiment.SmokeConfig
                    sample_key = $SmokeSampleKey
                    status = "failure"
                    exit_code = 1
                    requested_parameters = ""
                    effective_parameters = ""
                    predicted_points = ""
                    precision = ""
                    recall = ""
                    f1 = ""
                    iou = ""
                    implementation_hash = $FingerprintBefore.implementation_hash
                    output_path = $OutputRoot
                    error = $_.Exception.Message
                    checkpoint_reused = $false
                }
                Write-SmokeSummary
                Write-Host "FAILED SMOKE OUTPUT VALIDATION: $($Experiment.Name); $($_.Exception.Message)" -ForegroundColor Red
                $ExitCode = 1
                break
            }
        }

        Write-Host "FINISH SENSITIVITY: $($Experiment.Name)"
        Write-Host "timestamp: $(Get-Date -Format 'o')"
    }

    if ($ExitCode -eq 0) {
        $FingerprintAfter = Get-ImplementationFingerprint
        if ($FingerprintAfter.implementation_hash -ne $FingerprintBefore.implementation_hash) {
            throw "Production implementation hash changed during sensitivity sequence"
        }
        if ([int]$FingerprintAfter.implementation_fingerprint_schema -ne 2) {
            throw "Production implementation fingerprint schema changed during sequence"
        }
        if ($Smoke -and $SmokeRows.Count -ne 5) {
            throw "Smoke sequence produced $($SmokeRows.Count) summary rows instead of 5"
        }
        Write-Host "ALL SENSITIVITY EXPERIMENTS FINISHED"
        Write-Host "timestamp: $(Get-Date -Format 'o')"
    }
}
catch {
    if ($ExitCode -eq 0) {
        $ExitCode = 1
    }
    Write-Host "SENSITIVITY SEQUENCE STOPPED: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    if ($Smoke) {
        Write-SmokeSummary
    }
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
    Pop-Location
}

exit $ExitCode
