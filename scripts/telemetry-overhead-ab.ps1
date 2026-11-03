# Telemetry overhead A/B (OpenTelemetry plan section 8): interleaved requests across four Server.Host processes on one model.
#
# Build the two binaries first (plain Release publishes; both JIT, so the comparison is like for like):
#   git archive <commit before any instrumentation> | tar -x -C <Root>/baseline-src          (the plan used a261dfdb^ = 4925d974)
#   dotnet publish <Root>/baseline-src/src/OpenTail.Stingray.Server.Host -c Release -o <Root>/bin-A
#   dotnet publish src/OpenTail.Stingray.Server.Host -c Release -o <Root>/bin-B
# Run on a machine with nothing else going on and note the background CPU load next to the result:
#   pwsh -File scripts/telemetry-overhead-ab.ps1 -Root <Root> -Model <model.gguf> -Rounds 120
#
# Caveat the numbers must carry: A0 and B are two different commits, so A0 vs B includes every change between them, not only the instrumentation.
# The clean comparisons are A0 vs A1 (the same binary twice: the method's noise floor) and B vs C (the same binary with and without an exporter).
# Section 8 overhead A/B: interleaved requests across four servers, same model, prompt and settings.
#   A0 = baseline binary (built from a261dfdb^, before any instrumentation)          port 5101
#   A1 = a second instance of the same baseline binary: the A/A control (noise floor)  port 5104
#   B  = HEAD, no exporter, no listener                                                 port 5102
#   C  = HEAD with an OTLP/HTTP exporter (traces + metrics) pointed at a local sink    port 5103
param([string]$Root, [string]$Model, [int]$Rounds = 60, [int]$Warmup = 4, [int]$MaxTokens = 64)
$ErrorActionPreference = 'Stop'
$procs = @()
function Start-Server($name, $bin, $port, $extra) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new((Join-Path $bin 'OpenTail.Stingray.Server.Host.exe'))
    $psi.UseShellExecute = $false; $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
    $psi.Environment['STINGRAY_MODEL'] = $Model
    $psi.Environment['ASPNETCORE_URLS'] = "http://127.0.0.1:$port"
    $psi.Environment['STINGRAY_CONFIG_DIR'] = Join-Path $Root "cfg-$name"
    $psi.Environment['STINGRAY_MODEL_HOME'] = Join-Path $Root "home-$name"
    foreach ($k in $extra.Keys) { $psi.Environment[$k] = $extra[$k] }
    $p = [System.Diagnostics.Process]::Start($psi)
    # Drain the pipes so a chatty exporter cannot block on a full buffer.
    $null = $p.StandardOutput.BaseStream.CopyToAsync([System.IO.Stream]::Null)
    $null = $p.StandardError.BaseStream.CopyToAsync([System.IO.Stream]::Null)
    $script:procs += $p
    return $p
}
$sinkCount = Join-Path $Root 'sink.count'
$sink = Start-Process pwsh -ArgumentList '-NoProfile', '-File', (Join-Path $PSScriptRoot 'telemetry-overhead-sink.ps1'), $sinkCount -PassThru -WindowStyle Hidden
try {
    $servers = [ordered]@{
        A0 = @{ Port = 5101; Proc = (Start-Server 'A0' (Join-Path $Root 'bin-A') 5101 @{}) }
        A1 = @{ Port = 5104; Proc = (Start-Server 'A1' (Join-Path $Root 'bin-A') 5104 @{}) }
        B  = @{ Port = 5102; Proc = (Start-Server 'B'  (Join-Path $Root 'bin-B') 5102 @{}) }
        C  = @{ Port = 5103; Proc = (Start-Server 'C'  (Join-Path $Root 'bin-B') 5103 @{
                   OTEL_EXPORTER_OTLP_ENDPOINT = 'http://127.0.0.1:4318'; OTEL_EXPORTER_OTLP_PROTOCOL = 'http/protobuf' }) }
    }
    $http = [System.Net.Http.HttpClient]::new(); $http.Timeout = [TimeSpan]::FromSeconds(120)
    foreach ($s in $servers.Keys) {
        $port = $servers[$s].Port; $deadline = [DateTime]::UtcNow.AddSeconds(90); $up = $false
        while ([DateTime]::UtcNow -lt $deadline -and -not $up) {
            try { $r = $http.GetAsync("http://127.0.0.1:$port/health").GetAwaiter().GetResult(); $up = $r.IsSuccessStatusCode } catch { Start-Sleep -Milliseconds 500 }
        }
        if (-not $up) { throw "server $s did not come up on $port" }
    }
    $body = '{"model":"m","messages":[{"role":"user","content":"Explain in two sentences what a hash map is."}],"max_tokens":' + $MaxTokens + ',"temperature":0,"opentail_timing":true}'
    function Ask($port) {
        $content = [System.Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, 'application/json')
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $resp = $http.PostAsync("http://127.0.0.1:$port/v1/chat/completions", $content).GetAwaiter().GetResult()
        $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult(); $sw.Stop()
        if (-not $resp.IsSuccessStatusCode) { throw "request failed: $($resp.StatusCode) $text" }
        $j = $text | ConvertFrom-Json
        [pscustomobject]@{ Wall = $sw.Elapsed.TotalMilliseconds; Total = [double]$j.opentail_timing.total_ms; Decode = [double]$j.opentail_timing.decode_ms
                           Tokens = [int]$j.usage.completion_tokens; Text = $j.choices[0].message.content }
    }
    foreach ($s in $servers.Keys) { 1..$Warmup | ForEach-Object { $null = Ask $servers[$s].Port } }
    $perms = @(@('A0','A1','B','C'), @('A1','B','C','A0'), @('B','C','A0','A1'), @('C','A0','A1','B'), @('C','B','A1','A0'), @('B','A1','A0','C'))
    $rows = [System.Collections.Generic.List[object]]::new()
    for ($r = 0; $r -lt $Rounds; $r++) {
        foreach ($s in $perms[$r % $perms.Count]) {
            $x = Ask $servers[$s].Port
            $rows.Add([pscustomobject]@{ Round = $r; Server = $s; Wall = $x.Wall; Total = $x.Total; Decode = $x.Decode; Tokens = $x.Tokens; Text = $x.Text })
        }
    }
    $rows | Select-Object Round, Server, Wall, Total, Decode, Tokens | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $Root 'ab.csv')
    $texts = $rows | Select-Object -ExpandProperty Text | Sort-Object -Unique
    "distinct response texts across all servers and rounds: $(@($texts).Count)"
    function Med($v) { $s = $v | Sort-Object; $n = $s.Count; if ($n % 2) { $s[[int](($n-1)/2)] } else { ($s[$n/2-1] + $s[$n/2]) / 2 } }
    function Pct($v, $p) { $s = $v | Sort-Object; $s[[int][math]::Floor(($s.Count-1)*$p)] }
    foreach ($metric in 'Total','Decode','Wall') {
        "--- $metric (ms), $Rounds rounds, $MaxTokens tokens"
        foreach ($s in $servers.Keys) {
            $v = $rows | Where-Object Server -eq $s | ForEach-Object { $_.$metric }
            "{0,-3} median {1,8:F1}  p25 {2,8:F1}  min {3,8:F1}  max {4,8:F1}  mean {5,8:F1}" -f $s, (Med $v), (Pct $v 0.25), ($v | Measure-Object -Minimum).Minimum, ($v | Measure-Object -Maximum).Maximum, ($v | Measure-Object -Average).Average
        }
    }
    "sink received: $(if (Test-Path $sinkCount) { Get-Content $sinkCount } else { 'nothing' })"
}
finally {
    foreach ($p in $procs) { try { if (-not $p.HasExited) { $p.Kill($true) } } catch {} }
    try { if (-not $sink.HasExited) { $sink.Kill($true) } } catch {}
}
