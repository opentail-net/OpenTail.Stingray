# Start-up time for the same A/B: time from process start to the first /health 200, interleaved launches of the baseline and HEAD publishes
# (the same <Root>/bin-A and <Root>/bin-B as telemetry-overhead-ab.ps1).  pwsh -File scripts/telemetry-overhead-startup.ps1 -Root <Root> -Model <model.gguf>
param([string]$Root, [string]$Model, [int]$Runs = 12)
$ErrorActionPreference = 'Stop'
$http = [System.Net.Http.HttpClient]::new(); $http.Timeout = [TimeSpan]::FromSeconds(5)
function Launch($bin, $port) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new((Join-Path $bin 'OpenTail.Stingray.Server.Host.exe'))
    $psi.UseShellExecute = $false; $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
    $psi.Environment['STINGRAY_MODEL'] = $Model; $psi.Environment['ASPNETCORE_URLS'] = "http://127.0.0.1:$port"
    $psi.Environment['STINGRAY_CONFIG_DIR'] = Join-Path $Root 'cfg-startup'; $psi.Environment['STINGRAY_MODEL_HOME'] = Join-Path $Root 'home-startup'
    $sw = [Diagnostics.Stopwatch]::StartNew(); $p = [System.Diagnostics.Process]::Start($psi)
    $null = $p.StandardOutput.BaseStream.CopyToAsync([System.IO.Stream]::Null); $null = $p.StandardError.BaseStream.CopyToAsync([System.IO.Stream]::Null)
    $up = $false
    while (-not $up) { try { $up = $http.GetAsync("http://127.0.0.1:$port/health").GetAwaiter().GetResult().IsSuccessStatusCode } catch { Start-Sleep -Milliseconds 20 } }
    $ms = $sw.Elapsed.TotalMilliseconds; $p.Kill($true); $p.WaitForExit(); return $ms
}
$null = Launch (Join-Path $Root 'bin-A') 5201; $null = Launch (Join-Path $Root 'bin-B') 5202   # warm the disk cache
$a = @(); $b = @()
for ($i = 0; $i -lt $Runs; $i++) {
    if ($i % 2 -eq 0) { $a += Launch (Join-Path $Root 'bin-A') 5201; $b += Launch (Join-Path $Root 'bin-B') 5202 }
    else              { $b += Launch (Join-Path $Root 'bin-B') 5202; $a += Launch (Join-Path $Root 'bin-A') 5201 }
}
function Med($v) { $s = $v | Sort-Object; $n = $s.Count; ($s[[int](($n-1)/2)] + $s[[int]($n/2)]) / 2 }
"time to first /health 200, $Runs interleaved launches each:"
"baseline (pre-instrumentation): median {0:F0} ms, min {1:F0}, max {2:F0}" -f (Med $a), ($a | Measure-Object -Minimum).Minimum, ($a | Measure-Object -Maximum).Maximum
"HEAD:                           median {0:F0} ms, min {1:F0}, max {2:F0}" -f (Med $b), ($b | Measure-Object -Minimum).Minimum, ($b | Measure-Object -Maximum).Maximum
"HEAD / baseline: {0:F3}x" -f ((Med $b) / (Med $a))
