# Repack A/B: is llama.cpp's repacked 2D Q4_K GEMM the remaining "boss tower"?
#
# Design rationale, source line references and how to read the result: docs/repack-gemm/README.md
#
# Hard-won constraints baked in -- do not "simplify" these away:
#   1. llama-BENCH cannot run this experiment. It builds params from llama_model_default_params()
#      and never calls common_init_from_params, so LLAMA_ARG_REPACK is silently ignored and both
#      arms run repacked.
#   2. llama-CLI is interactive-only as of b8585 ("--no-conversation is not supported by llama-cli,
#      please use llama-completion instead"). It answers the prompt then blocks on stdin forever.
#      Use llama-completion, and redirect stdin from an empty file so nothing can ever hang.
#   3. Phase 1 is a hard gate. If CPU_REPACK is absent when repack is enabled, or present when it
#      is disabled, the flag did not take effect and every timing below would be noise -> abort.
#   4. Weight-byte accounting: with repack ON, "Host ... model" still reports the whole mmap'd
#      file and CPU_REPACK is an ADDITIONAL buffer holding repacked copies. So the repacked
#      fraction is CPU_REPACK(on) / HostModel(off), NOT repack/(repack+host) -- the latter
#      double-counts and understates the share (42% vs the true ~72%).
#
# Crash-safety: every sample is appended to the markdown immediately, so an interrupted session
# loses at most one run. Raw stdout/stderr of every invocation is kept under docs/repack-gemm/raw/.
#
# Usage: pwsh -NoProfile -File scripts/repack-ab.ps1 [-Runs 3] [-TimeoutSec 600]

param(
    [int[]]$Words      = @(900, 4700),
    [int]$Runs         = 3,
    [int]$Threads      = 6,       # physical cores on the Zen 3 box; both arms identical
    [int]$Ctx          = 8192,
    [int]$TimeoutSec   = 600
)

$ErrorActionPreference = "Stop"

$root    = Split-Path -Parent (Split-Path -Parent $PSCommandPath)   # extensions/OpenTail.Stingray
$exe     = Join-Path $root "tools\llama.cpp\llama-completion.exe"
$model   = Join-Path $root "models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
$outDir  = Join-Path $root "docs\repack-gemm"
$rawDir  = Join-Path $outDir "raw"
$md      = Join-Path $outDir "ab-results.md"

foreach ($d in @($outDir, $rawDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

# Empty stdin: any tool that unexpectedly goes interactive gets EOF and exits instead of hanging.
$nulIn = Join-Path $rawDir "_empty-stdin.txt"
Set-Content -Path $nulIn -Value "" -NoNewline -Encoding utf8

# NOTE: 'Md' is a built-in PowerShell alias for mkdir, and aliases OUTRANK functions in command
# resolution -- a function named Md is silently shadowed. Hence Write-Md.
function Write-Md([string]$s = "") { Add-Content -Path $md -Value $s -Encoding utf8 }

function Invoke-Llama {
    param([string]$Tag, [string]$ExtraFlags = "", [string]$PromptFile = $null)

    $so = Join-Path $rawDir "$Tag.out.txt"
    $se = Join-Path $rawDir "$Tag.err.txt"

    # -st (--single-turn) is LOad-BEARING: without it llama-completion waits on stdin, hits EOF,
    # prints "Interrupted by user" and raises the console interrupt. With -NoNewWindow the child
    # shares our console, so that CTRL_C_EVENT reached the parent pwsh and killed the whole script
    # right after the first run -- silently, with no exception and no log line. --simple-io is the
    # documented subprocess-compatibility switch. -WindowStyle Hidden gives the child its own
    # console as defence in depth, so no future signal of its own can reach us.
    $a  = "-m `"$model`" -c $Ctx -t $Threads -n 1 -st --simple-io"
    if ($PromptFile) { $a += " -f `"$PromptFile`"" } else { $a += " -p `"hi`"" }
    if ($ExtraFlags) { $a += " $ExtraFlags" }

    $p = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru `
                       -RedirectStandardOutput $so -RedirectStandardError $se
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill($true) } catch { }
        Write-Warning "TIMEOUT after ${TimeoutSec}s: $Tag"
        return @{ Ok = $false; Reason = "timeout"; Text = "" }
    }
    $txt = ""
    foreach ($f in @($so, $se)) {
        if (Test-Path $f) { $txt += (Get-Content $f -Raw -ErrorAction SilentlyContinue) + "`n" }
    }
    return @{ Ok = $true; Text = $txt; ExitCode = $p.ExitCode }
}

# Parses the llama_memory_breakdown_print table:
#   |   - Host        |   2645 =  1005 +  1536 +  104 |    (total = model + context + compute)
#   |   - CPU_REPACK  |    729 =   729 +     0 +    0 |
function Get-BufMiB([string]$Text, [string]$Row) {
    if ($Text -match ("-\s*" + [regex]::Escape($Row) + "\s*\|\s*(\d+)\s*=\s*(\d+)\s*\+")) {
        return [double]$Matches[2]      # the "model" column
    }
    return $null
}

# ---------------------------------------------------------------- phase 0: preflight
if (-not (Test-Path $exe))   { throw "llama-completion.exe not found at $exe" }
if (-not (Test-Path $model)) { throw "model not found at $model" }

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Md ""
Write-Md "---"
Write-Md ""
Write-Md "# Repack A/B run - $stamp"
Write-Md ""
Write-Md "Runner: ``scripts/repack-ab.ps1``.  Interpretation: ``docs/repack-gemm/README.md``."
Write-Md ""
Write-Md "| control | value |"
Write-Md "|---|---|"
Write-Md "| exe | ``$(Split-Path -Leaf $exe)`` (llama-cli is interactive-only in b8585) |"
Write-Md "| model | ``$(Split-Path -Leaf $model)`` ($([math]::Round((Get-Item $model).Length/1MB,1)) MiB) |"
Write-Md "| ctx | $Ctx |"
Write-Md "| threads | $Threads (physical cores) |"
Write-Md "| runs per arm | $Runs, interleaved on/off |"
Write-Md "| warmup | enabled (timed prefill must not pay first-touch page faults) |"
Write-Md ""

Write-Host "=== phase 1: gate - does --no-repack actually take effect? ===" -ForegroundColor Cyan
Write-Md "## Phase 1 - gate: did the flag take effect?"
Write-Md ""

$gate = @{}
foreach ($arm in @("on", "off")) {
    $flags = if ($arm -eq "off") { "--no-repack" } else { "" }
    $r = Invoke-Llama -Tag "gate-$arm" -ExtraFlags $flags
    if (-not $r.Ok) { Write-Md "- **$arm**: FAILED ($($r.Reason))"; $gate[$arm] = $null; continue }

    $rep  = Get-BufMiB $r.Text "CPU_REPACK"
    $hostMiB = Get-BufMiB $r.Text "Host"
    $gate[$arm] = @{ Repack = $rep; Host = $hostMiB }

    Write-Md "- **repack $arm**: CPU_REPACK model = $(if ($null -ne $rep) { "$rep MiB" } else { "*absent*" }); Host model = $(if ($null -ne $hostMiB) { "$hostMiB MiB" } else { "*absent*" })"
    Write-Host ("  repack {0,-3}: CPU_REPACK={1}  Host={2}" -f $arm, $rep, $hostMiB)

    if ($arm -eq "on") {
        $census = [regex]::Matches($r.Text, '-\s+type\s+(\S+):\s+(\d+)\s+tensors')
        if ($census.Count -gt 0) {
            Write-Md ""
            Write-Md "Tensor type census (load-time):"
            Write-Md ""
            Write-Md '```'
            foreach ($m in $census) { Write-Md ("  {0,-8} {1,5} tensors" -f $m.Groups[1].Value, $m.Groups[2].Value) }
            Write-Md '```'
            Write-Md ""
        }
    }
}

# Gate: CPU_REPACK must be present with repack ON and absent with --no-repack.
$gateOk = ($null -ne $gate["on"]  -and $null -ne $gate["on"].Repack) -and
          ($null -ne $gate["off"] -and $null -eq $gate["off"].Repack)

if (-not $gateOk) {
    Write-Md ""
    Write-Md "> **GATE FAILED - run aborted.** Expected a ``CPU_REPACK`` buffer with repack enabled"
    Write-Md "> and none with ``--no-repack``. Without that, both arms run the same code and every"
    Write-Md "> timing below would be noise - exactly the failure that invalidated the earlier"
    Write-Md "> ``llama-bench`` attempt (README section 0). See ``docs/repack-gemm/raw/gate-*.txt``."
    Write-Md ""
    Write-Host "GATE FAILED - aborting before timing. See $md" -ForegroundColor Red
    exit 2
}

# Repacked layout is a permutation of the same bits (block_q4_Kx8 == 8x block_q4_K), so
# CPU_REPACK MiB == the original MiB of the tensors that changed code path.
$totalW   = $gate["off"].Host
$dilution = $gate["on"].Repack / $totalW
Write-Md ""
Write-Md ("**Gate passed.** Repacked tensors are $($gate['on'].Repack) MiB of $totalW MiB total weights = " +
          "**$([math]::Round($dilution * 100,1))%** of weight bytes. The remaining " +
          "$([math]::Round((1-$dilution) * 100,1))% (Q6_K / F32 - neither repacks on AVX2) takes the " +
          "identical path in both arms, so the raw ratio below is a **lower bound**.")
Write-Md ""
Write-Host ("  gate PASSED - repacked share = {0:P1}" -f $dilution) -ForegroundColor Green

# ---------------------------------------------------------------- phase 2: interleaved A/B
Write-Host "=== phase 2: interleaved A/B ===" -ForegroundColor Cyan
Write-Md "## Phase 2 - prefill A/B"
Write-Md ""
Write-Md "Only ``prompt eval time`` is used: the repacked GEMM is selected only for M > 3"
Write-Md "(``repack.cpp:4241``), so decode (M=1) runs GEMV and is not informative here."
Write-Md ""
Write-Md "| tokens | arm | run | prefill t/s |"
Write-Md "|---:|---|---:|---:|"

$samples = @{}
foreach ($w in $Words) {
    $pf = Join-Path $rawDir "prompt-$w.txt"
    Set-Content -Path $pf -Value (("data " * $w).Trim()) -NoNewline -Encoding utf8
    $samples[$w] = @{ on = @(); off = @() }
    $tokens = $w

    for ($r = 0; $r -lt $Runs; $r++) {
        foreach ($arm in @("on", "off")) {
            $flags = if ($arm -eq "off") { "--no-repack" } else { "" }
            $res = Invoke-Llama -Tag "ab-$w-$arm-r$r" -ExtraFlags $flags -PromptFile $pf

            $tps = $null
            if ($res.Ok -and $res.Text -match 'prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens[^\r\n]*?([\d.]+)\s*tokens per second') {
                $tokens = [int]$Matches[1]; $tps = [double]$Matches[2]
            } elseif ($res.Ok -and $res.Text -match 'Prompt:\s*([\d.]+)\s*t/s') {
                $tps = [double]$Matches[1]
            }

            if ($null -ne $tps) {
                $samples[$w][$arm] += $tps
                Write-Md ("| $tokens | $arm | $r | $([math]::Round($tps,2)) |")   # appended immediately
                Write-Host ("  {0,5}tok {1,-3} r{2}: {3,8:F2} t/s" -f $tokens, $arm, $r, $tps)
            } else {
                Write-Md ("| $w words | $arm | $r | *failed* |")
                Write-Host ("  {0,5}wd  {1,-3} r{2}: FAILED" -f $w, $arm, $r) -ForegroundColor Yellow
            }
        }
    }
}

# ---------------------------------------------------------------- phase 3: verdict
Write-Md ""
Write-Md "## Phase 3 - verdict"
Write-Md ""
Write-Md "| tokens | repack ON (best) | repack OFF (best) | ratio | un-diluted est. |"
Write-Md "|---:|---:|---:|---:|---:|"

foreach ($w in $Words) {
    $on  = $samples[$w].on
    $off = $samples[$w].off
    if ($on.Count -eq 0 -or $off.Count -eq 0) { Write-Md "| $w words | - | - | *insufficient samples* | - |"; continue }

    # Best-of, not mean: interference on this box is one-sided (it can only slow a run down),
    # so max observed throughput is the cleanest estimator of the code's real speed.
    $bOn   = ($on  | Measure-Object -Maximum).Maximum
    $bOff  = ($off | Measure-Object -Maximum).Maximum
    $ratio = $bOn / $bOff

    # Amdahl solved for the changed part: 1/ratio = (1-f) + f/s  =>  s = f / (1/ratio - (1-f)).
    # Uses byte share as a proxy for time share -- fair for a memory-bound quantised matmul,
    # but an estimate, not a measurement.
    $denom    = (1.0 / $ratio) - (1.0 - $dilution)
    $undilTxt = if ($denom -gt 1e-9) { "~{0:F2}x" -f ($dilution / $denom) } else { "n/a (exceeds Amdahl bound)" }

    Write-Md ("| $w | {0:F2} | {1:F2} | **{2:F2}x** | {3} |" -f $bOn, $bOff, $ratio, $undilTxt)
    Write-Host ("  {0,5}wd  ratio {1:F2}x  (un-diluted {2})" -f $w, $ratio, $undilTxt) -ForegroundColor Yellow
}

Write-Md ""
Write-Md "Reading (README section 6): >=2x means the repacked 2D GEMM is the boss tower and the"
Write-Md "phase-2 premise holds; <=1.2x relocates it to ordinary ``ggml_vec_dot_q4_K_q8_K`` codegen"
Write-Md "plus threading, a far cheaper target than a 1450-line kernel port."
Write-Md ""
Write-Host "done -> $md" -ForegroundColor Green
