# Runs every verification offline (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\run_checks.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_checks.ps1 -Quick   # skip the crash test
param([switch]$Quick)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$env:HF_HUB_OFFLINE = "1"          # prove nothing needs the network
$env:PYTHONIOENCODING = "utf-8"

$runs = @(
    @{ name = "Part 1, variant 1"; args = @("tests\verify.py", "--variant", "1") },
    @{ name = "Part 1, variant 2"; args = @("tests\verify.py", "--variant", "2") },
    @{ name = "Part 1, variant 3"; args = @("tests\verify.py", "--variant", "3") },
    @{ name = "Part 2, HTTP API";  args = @("tests\verify_api.py") + $(if ($Quick) { @("--skip-crash") } else { @() }) }
)
$results = @()
foreach ($r in $runs) {
    Write-Host "`n==== $($r.name) ====" -ForegroundColor Cyan
    & $py @($r.args)
    $code = $LASTEXITCODE
    if ($code -eq 2) { Write-Host "`nStopped: see the message above (model or test audio missing)." -ForegroundColor Red; exit 2 }
    $results += [pscustomobject]@{ Check = $r.name; Result = $(if ($code -eq 0) { "all passed" } else { "some checks failed" }) }
}
Write-Host "`n==== Summary ====" -ForegroundColor Cyan
$results | Format-Table -AutoSize
if ($results.Result -contains "some checks failed") { exit 1 } else { exit 0 }
