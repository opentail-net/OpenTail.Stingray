# Re-check OUR side (C#, t/s) of PerformanceLeague.md for the LLM text rows: one results file, one model at a time.
#
# Sets:
#   quick   5 representative models, short + 500 + 2k scenarios         (about 13 minutes)
#   full    the curated League list (18 models), same scenarios
#   all     EVERY text-generation GGUF found under every drive's _models, K:\_other_models and the repo's models/
#           (scripts/list-checkpoints.ps1 builds the inventory; non-LLM architectures and mmproj files are listed as not-in-scope),
#           smallest first so the heavy hitters run last, 500-token scenario only, one run each (override with -Scenarios / -Runs)
#
# Scenarios (parsed from the CLI's "Prefill: <N> tokens, <X> t/s | Decode: <M> tokens, <Y> t/s"):
#     short  ~15-token prompt, 64 new tokens -> decode at short context
#     500    docs/reference/benchmark-prompt.txt (~490 tokens), 16 new tokens -> prefill (~500 tok) + decode @500
#     2k     that prompt x4 (~2,000 tokens), 24 new tokens -> prefill (~2k) + decode @2k (SnapKV auto-budget 2048 applies)
# on CPU (-g 0) and Vulkan (-g -1). The CPU 2k scenario needs -CpuLong.
#
# Results stream into -OutFile (one markdown row per scenario) and the LAST thing written is "<OutFile>.done" holding one summary
# line, so a watcher can wait for that file instead of waking per result:
#     until [ -f "<OutFile>.done" ]; do sleep 30; done
# -Resume skips (model, backend, scenario) rows that are already in -OutFile, so an interrupted run continues where it stopped.
#
# Rules this follows: never two model jobs at once (it refuses to start while a stingray / test process is running and runs strictly
# sequentially); default JIT settings (no DOTNET_TC_QuickJitForLoops=0, per the no-TC=0 rule); best-of-N because interference on this
# machine can only slow a run down; a model that is missing, too big for free memory, unsupported or that times out is RECORDED
# (SKIP / UNSUPPORTED / TIMEOUT), never silently dropped, and the script moves on.
#
# The numbers go into PerformanceLeague.md BY HAND, as dated rows, after reading them (rule 12: check the timings, not just that a
# row exists).
param(
    [ValidateSet('quick', 'full', 'all')][string]$Set = 'quick',
    [string[]]$Models = @(),                      # file stems (without .gguf); overrides -Set
    [ValidateSet('cpu', 'vulkan')][string[]]$Backends = @('cpu', 'vulkan'),
    [ValidateSet('short', '500', '2k')][string[]]$Scenarios = @(),   # default: short,500,2k for quick/full; 500 for all
    [int]$Runs = 0,                               # default: 2 for quick/full, 1 for all
    [switch]$CpuLong,                             # also run the ~2k scenario on CPU
    [double]$MaxGB = 0,                           # -Set all: skip checkpoints larger than this; 0 = no cap (the engine streams / splits oversized checkpoints)
    [int]$TimeoutMinutes = 25,                    # per run
    [switch]$Resume,
    [string]$OutFile = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$exe = Join-Path $repo 'src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe'
if (-not (Test-Path $exe)) { throw "Build the CLI in Release first: dotnet build src/OpenTail.Stingray.Cli -c Release" }

if ($Runs -le 0) { $Runs = if ($Set -eq 'all') { 1 } else { 2 } }
if ($Scenarios.Count -eq 0) { $Scenarios = if ($Set -eq 'all') { @('500') } else { @('short', '500', '2k') } }

$quick = @('SmolLM2-1.7B-Instruct-Q4_K_M', 'Qwen3-4B-Q4_K_M', 'Qwen3.5-4B-Q4_K_M', 'gemma-4-E4B-it-Q4_K_M', 'Mistral-7B-Instruct-v0.3-Q4_K_M')
$full = $quick + @('SmolLM2-135M-Instruct-Q4_K_M', 'SmolLM2-360M-Instruct-Q4_K_M', 'qwen2.5-0.5b-instruct-q4_k_m', 'qwen2.5-3b-instruct-q4_k_m',
    'Qwen3-0.6B-Q8_0', 'Qwen3-8B-Q4_K_M', 'Qwen3.5-9B-Q4_K_M', 'gemma-3-4b-it-Q4_K_M', 'gemma-4-12b-it-Q4_K_M',
    'OLMoE-1B-7B-0924-Instruct-Q4_K_M', 'gpt-oss-20b-MXFP4', 'Qwen2.5-VL-7B-Instruct-Q4_K_M', 'Phi-3-mini-4k-instruct-Q4_K_M')

function Find-Model([string]$stem) {
    foreach ($d in @('models', 'models/_models')) {
        $p = Join-Path $repo "$d/$stem.gguf"
        if (Test-Path $p) { return $p }
    }
    return $null
}

# Work list: objects { stem; path; gb; arch }
$work = @()
$notInScope = @()
if ($Models.Count -gt 0 -or $Set -ne 'all') {
    $stems = if ($Models.Count -gt 0) { $Models } elseif ($Set -eq 'full') { $full } else { $quick }
    foreach ($s in $stems) {
        $p = Find-Model $s
        $work += [pscustomobject]@{ stem = $s; path = $p; gb = $(if ($p) { [Math]::Round((Get-Item $p).Length / 1GB, 2) } else { 0 }); arch = '' }
    }
} else {
    $inv = & (Join-Path $PSScriptRoot 'list-checkpoints.ps1')
    foreach ($r in ($inv | Where-Object kind -eq 'llm' | Sort-Object { [double]$_.sizeGB })) {
        $work += [pscustomobject]@{ stem = [System.IO.Path]::GetFileNameWithoutExtension($r.name); path = $r.path; gb = [double]$r.sizeGB; arch = $r.arch }
    }
    $notInScope = @($inv | Where-Object { $_.kind -ne 'llm' })
}

if (-not $OutFile) { $OutFile = Join-Path ([System.IO.Path]::GetTempPath()) ("league-recheck-{0:yyyyMMdd-HHmm}.md" -f (Get-Date)) }
$doneFile = "$OutFile.done"
Remove-Item -Path $doneFile -ErrorAction SilentlyContinue

# Never two model jobs at once: wait (up to 15 min) for any other stingray / test process to finish, else refuse.
function Get-ModelJobs { Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'stingray*' -or $_.ProcessName -like 'OpenTail.Stingray.Tests*' -or $_.ProcessName -like 'sd-cli*' -or $_.ProcessName -like 'llama-*' } }
$waited = 0
while ((Get-ModelJobs).Count -gt 0) {
    if ($waited -ge 900) { throw "Another model job is running ($((Get-ModelJobs | Select-Object -First 1).ProcessName)); not starting a second one." }
    Start-Sleep -Seconds 15; $waited += 15
}

function Get-FreeGB { (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB }

# Prompt files (passed with -f: long prompts exceed the Windows command-line limit with -p).
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) 'league-recheck-prompts'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$pShort = Join-Path $tmp 'short.txt'
$p500 = Join-Path $repo 'docs/reference/benchmark-prompt.txt'
$p2k = Join-Path $tmp 'x4.txt'
Set-Content -Path $pShort -Value 'Explain in two sentences why the sky is blue.' -NoNewline -Encoding utf8
$base = (Get-Content -Raw -Path $p500)
Set-Content -Path $p2k -Value ($base + ' ' + $base + ' ' + $base + ' ' + $base) -NoNewline -Encoding utf8
$scenarioDefs = @{ 'short' = @($pShort, 64); '500' = @($p500, 16); '2k' = @($p2k, 24) }

function Get-BlockCount([string]$path) {
    $t = (& $exe list-metadata -m $path 2>&1 | Out-String)
    if ($t -match '\.block_count\s+(\d+)') { return [int]$Matches[1] }
    return 0
}

function Invoke-One([string]$model, [string]$backend, [string]$promptFile, [int]$n, [string]$nglOverride = "") {
    $ngl = if ($nglOverride) { $nglOverride } elseif ($backend -eq "vulkan") { "-1" } else { "0" }
    $psi = [System.Diagnostics.ProcessStartInfo]::new($exe)
    $argv = @("-m", $model, "-f", $promptFile, "-n", "$n", "--temp", "0", "-g", $ngl)
    # Oversized checkpoints are supported, so the CPU memory preflight (a refusal, not a capability limit) is bypassed for the harness.
    if ($backend -eq "cpu") { $argv += "--ignore-preflight" }
    foreach ($a in $argv) { [void]$psi.ArgumentList.Add($a) }
    $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $errTask = $proc.StandardError.ReadToEndAsync()
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutMinutes * 60 * 1000)) { try { $proc.Kill($true) } catch {}; return @{ ok = $false; kind = 'TIMEOUT'; why = "timeout after $TimeoutMinutes min" } }
    $text = $outTask.Result + "`n" + $errTask.Result
    if ($text -match 'Prefill:\s*(\d+)\s*tokens,\s*([\d.]+)\s*t/s\s*\|\s*Decode:\s*(\d+)\s*tokens,\s*([\d.]+)\s*t/s') {
        return @{ ok = $true; pt = [int]$Matches[1]; pf = [double]$Matches[2]; dt = [int]$Matches[3]; df = [double]$Matches[4] }
    }
    # Keep the first useful error line so an UNSUPPORTED row says why.
    $line = ($text -split "`n" | Where-Object { $_ -match 'rror|not supported|unsupported|refus|Unknown|cannot|failed|Exception' } | Select-Object -First 1)
    if (-not $line) { $line = "no Prefill/Decode line (exit $($proc.ExitCode))" }
    return @{ ok = $false; kind = 'UNSUPPORTED'; why = ($line.Trim() -replace '\|', '/' -replace '\s+', ' ') }
}

$commit = (git rev-parse --short HEAD 2>$null)
$dirty = if ((git status --porcelain 2>$null | Measure-Object).Count -gt 0) { '+dirty' } else { '' }
$started = Get-Date
$done = @{}
if ($Resume -and (Test-Path $OutFile)) {
    foreach ($l in (Get-Content $OutFile)) { if ($l -match '^\| (.+?) \| [\d.]+ \| .*? \| (cpu|vulkan) \| (short|500|2k) \|') { $done["$($Matches[1])|$($Matches[2])|$($Matches[3])"] = $true } }
} else {
@"
# League recheck, our side (C#), $($started.ToString('yyyy-MM-dd HH:mm')), set '$Set'

Commit ``$commit$dirty``; engine defaults (no env overrides); best of $Runs run(s) per row; scenarios: $($Scenarios -join ', ') (short ~15-40 tok / 500 ~490-580 tok / 2k ~2,000 tok; SnapKV auto-budget 2048 applies); per-run timeout $TimeoutMinutes min$(if ($Set -eq 'all' -and $MaxGB -gt 0) { "; checkpoints over $MaxGB GB skipped" }).
Machine: Ryzen 7 5700G + integrated Radeon (iGPU; rule 13: do not generalise to discrete GPUs). 63 GB RAM.

| Model | GB | Arch | Backend | Scenario | Prompt tok | Prefill t/s (best) | Decode t/s (best) | All runs (prefill / decode) |
|---|---:|---|---|---|---:|---:|---:|---|
"@ | Set-Content -Path $OutFile -Encoding utf8
}

$ok = 0; $bad = 0; $skipped = 0; $unsup = 0
foreach ($m in $work) {
    $stem = $m.stem
    if (-not $m.path) { Add-Content -Path $OutFile -Value "| $stem | - | - | - | - | - | SKIP | SKIP | checkpoint not present |"; $skipped++; continue }
    if ($Set -eq 'all' -and $MaxGB -gt 0 -and $m.gb -gt $MaxGB) { Add-Content -Path $OutFile -Value "| $stem | $($m.gb) | $($m.arch) | - | - | - | SKIP | SKIP | larger than -MaxGB $MaxGB |"; $skipped++; continue }
    # Memory guard: wait up to 3 minutes for enough free RAM for the mapped weights plus headroom, else record and skip.
    # Memory guard: only WAITS (up to 3 minutes) for free RAM to recover from the previous model; it never skips, because oversized checkpoints are
    # supported (layer split, expert streaming, mmap) and a slow or failed run is recorded as such.
    $needGB = [Math]::Min($m.gb + 4, 40); $w = 0
    while ((Get-FreeGB) -lt $needGB -and $w -lt 180) { Start-Sleep -Seconds 15; $w += 15 }
    $modelDead = @{}
    foreach ($backend in $Backends) {
        foreach ($scName in $Scenarios) {
            if ($scName -eq '2k' -and $backend -eq 'cpu' -and -not $CpuLong) { continue }
            if ($done.ContainsKey("$stem|$backend|$scName")) { continue }
            if ($modelDead[$backend]) { continue }
            $file = $scenarioDefs[$scName][0]; $n = [int]$scenarioDefs[$scName][1]
            $pf = @(); $df = @(); $tok = 0; $why = ''; $kind = ''; $suffix = ""
            for ($r = 0; $r -lt $Runs; $r++) {
                $res = Invoke-One $m.path $backend $file $n
                # Full Vulkan offload of a checkpoint larger than the placement budget runs out of host memory while uploading layers; the supported way to
                # run such a checkpoint is a partial offload, so retry once with -g N sized to about 12 GB of layers and label the row.
                if (-not $res.ok -and $backend -eq 'vulkan' -and $res.why -match 'OutOfHostMemory|ErrorOutOf|out of memory') {
                    $layers = Get-BlockCount $m.path
                    if ($layers -gt 0 -and $m.gb -gt 12) {
                        $gN = [Math]::Max(1, [int][Math]::Floor($layers * 12.0 / $m.gb))
                        $res = Invoke-One $m.path $backend $file $n "$gN"
                        $suffix = " [partial offload -g $gN]"
                    }
                }
                if ($res.ok) { $pf += $res.pf; $df += $res.df; $tok = $res.pt } else { $why = $res.why; $kind = $res.kind; break }
            }
            if ($pf.Count -eq 0) {
                Add-Content -Path $OutFile -Value "| $stem | $($m.gb) | $($m.arch) | $backend | $scName | - | $kind | $kind | $why |"
                if ($kind -eq 'UNSUPPORTED') { $unsup++ } else { $bad++ }
                $modelDead[$backend] = $true   # no point retrying the larger prompts on a backend that already failed
            } else {
                $bp = ($pf | Measure-Object -Maximum).Maximum; $bd = ($df | Measure-Object -Maximum).Maximum
                $runsTxt = (0..($pf.Count - 1) | ForEach-Object { '{0:F1}/{1:F1}' -f $pf[$_], $df[$_] }) -join ', '
                $runsTxt += $suffix
                Add-Content -Path $OutFile -Value ("| {0} | {1} | {2} | {3} | {4} | {5} | {6:F1} | {7:F1} | {8} |" -f $stem, $m.gb, $m.arch, $backend, $scName, $tok, $bp, $bd, $runsTxt); $ok++
            }
        }
    }
}

if ($notInScope.Count -gt 0) {
    Add-Content -Path $OutFile -Value "`n## Not in scope for this harness ($($notInScope.Count) files, $([Math]::Round(($notInScope | Measure-Object sizeGB -Sum).Sum, 1)) GB)`n`nEmbeddings / rerank, ASR, TTS, diffusion, vision projectors and non-GGUF checkpoints need their own harnesses (scripts/bench-audio.ps1, scripts/bench-image.ps1, the encoder benchmark tests).`n`n| Kind | Arch | File | GB |`n|---|---|---|---:|"
    foreach ($r in ($notInScope | Sort-Object kind, arch, name)) { Add-Content -Path $OutFile -Value "| $($r.kind) | $($r.arch) | $($r.name) | $($r.sizeGB) |" }
}

$elapsed = (Get-Date) - $started
$summary = "LEAGUE-RECHECK DONE set=$Set results=$ok unsupported=$unsup failed=$bad skipped=$skipped elapsed=$([int]$elapsed.TotalMinutes)min file=$OutFile"
Add-Content -Path $OutFile -Value "`n$summary"
Set-Content -Path $doneFile -Value $summary -Encoding utf8
Write-Output $summary
