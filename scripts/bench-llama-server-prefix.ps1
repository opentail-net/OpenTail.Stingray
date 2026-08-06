# Incumbent comparison for the session runtime plan's Milestone 0 §3.4:
# "Measure current llama-server slot/prefix behavior under an equal byte budget."
#
# Produces llama-server's warm-vs-cold prefill ratio on the SAME model and context OpenTail's
# session-bench uses, so the numbers sit directly beside §3.4.1's 24.0x (1K) / 71.1x (4K).
#
# Method: two /completion requests against one slot. The first sees a cold slot and must evaluate
# the whole prompt. The second sends the identical prefix plus a short suffix; with prompt caching
# on, llama-server should evaluate only the suffix. `timings.prompt_n` is the number of prompt
# tokens it ACTUALLY evaluated, which is what makes this measurable rather than inferred.
#
# The cold arm is forced cold by restarting the server, not by sending a different prompt — a
# different prompt would also change the work, confounding "cache miss" with "more tokens".
param(
    [string]$Model   = "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    [int]$Ctx        = 8192,
    [int[]]$Prefixes = @(1024, 4096),
    [int]$SuffixWords = 32,
    [int]$Port       = 8081,
    [switch]$SlotSave   # also time --slot-save-path save/restore (the reverse-proxy pattern)
)

$ErrorActionPreference = "Stop"
$exe = "tools/llama.cpp/llama-server.exe"
if (-not (Test-Path $exe))   { throw "llama-server not found at $exe (run scripts/setup-llamacpp.ps1)" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }

$slotDir = Join-Path ([System.IO.Path]::GetTempPath()) "opentail-llama-slots"
if ($SlotSave -and -not (Test-Path $slotDir)) { New-Item -ItemType Directory $slotDir | Out-Null }

function Start-Server {
    $serverArgs = @("-m", $Model, "-c", "$Ctx", "-np", "1", "--slots", "--port", "$Port", "-t", "12")
    if ($SlotSave) { $serverArgs += @("--slot-save-path", $slotDir) }
    $p = Start-Process -FilePath $exe -ArgumentList $serverArgs -PassThru -WindowStyle Hidden `
                       -RedirectStandardOutput "$env:TEMP\llama-server-out.txt" `
                       -RedirectStandardError  "$env:TEMP\llama-server-err.txt"
    # Poll /health rather than sleeping a guessed interval: model load time varies with page cache.
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($h.status -eq "ok") { return $p }
        } catch { }
    }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    throw "llama-server did not become healthy within 60s; see $env:TEMP\llama-server-err.txt"
}

function Complete($prompt) {
    $body = @{ prompt = $prompt; n_predict = 1; temperature = 0; cache_prompt = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
                           -ContentType "application/json" -Body $body -TimeoutSec 600
    [pscustomobject]@{ PromptN = $r.timings.prompt_n; PromptMs = $r.timings.prompt_ms }
}

Write-Host "=== llama-server prefix/slot behaviour ===" -ForegroundColor Cyan
Write-Host ("model {0}  ctx {1}  slots 1" -f (Split-Path $Model -Leaf), $Ctx)
Write-Host ""
Write-Host ("{0,10} {1,12} {2,12} {3,12} {4,12} {5,10}" -f "prefix", "cold tok", "cold ms", "warm tok", "warm ms", "speedup")

foreach ($n in $Prefixes) {
    $prefix = (("data " * $n).Trim())
    $suffix = " " + (("more " * $SuffixWords).Trim())

    # COLD: fresh server, so the slot has never seen this prefix.
    $srv = Start-Server
    try   { $cold = Complete ($prefix + $suffix) }
    finally { Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 800 }

    # WARM: same server, prefix already evaluated by the priming request.
    $srv = Start-Server
    try {
        [void](Complete $prefix)          # prime the slot
        $warm = Complete ($prefix + $suffix)
    }
    finally { Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 800 }

    $speedup = if ($warm.PromptMs -gt 0) { $cold.PromptMs / $warm.PromptMs } else { [double]::NaN }
    Write-Host ("{0,10} {1,12} {2,12:F1} {3,12} {4,12:F1} {5,9:F2}x" -f `
        $n, $cold.PromptN, $cold.PromptMs, $warm.PromptN, $warm.PromptMs, $speedup)
}

if ($SlotSave) {
    # The reverse-proxy save/restore pattern: prime a slot, persist its KV to disk, erase it, then
    # restore from the file. This is what a proxy in front of llama-server would do to survive a
    # restart or to multiplex more logical sessions than there are slots. OpenTail has no durable
    # equivalent yet (Milestone 3), so this measures a capability the incumbent has and we do not.
    Write-Host ""
    Write-Host "=== slot save / restore (reverse-proxy pattern) ===" -ForegroundColor Cyan
    Write-Host ("{0,10} {1,12} {2,12} {3,14} {4,14}" -f "prefix", "save ms", "restore ms", "file MiB", "post-restore tok")

    foreach ($n in $Prefixes) {
        $prefix = (("data " * $n).Trim())
        $suffix = " " + (("more " * $SuffixWords).Trim())
        $file = "slot-$n.bin"
        $srv = Start-Server
        try {
            [void](Complete $prefix)     # prime the slot

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/slots/0?action=save" -Method Post `
                -ContentType "application/json" -Body (@{ filename = $file } | ConvertTo-Json) `
                -TimeoutSec 600 | Out-Null
            $sw.Stop(); $saveMs = $sw.Elapsed.TotalMilliseconds

            # Erase so the restore cannot be satisfied from what is already resident.
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/slots/0?action=erase" -Method Post `
                -TimeoutSec 60 | Out-Null

            $sw.Restart()
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/slots/0?action=restore" -Method Post `
                -ContentType "application/json" -Body (@{ filename = $file } | ConvertTo-Json) `
                -TimeoutSec 600 | Out-Null
            $sw.Stop(); $restoreMs = $sw.Elapsed.TotalMilliseconds

            # Proof the restore actually took: prompt_n must collapse to the suffix, not the whole
            # prompt. Without this the timings above could be measuring a no-op.
            $after = Complete ($prefix + $suffix)

            $sizeMiB = (Get-Item (Join-Path $slotDir $file)).Length / 1MB
            Write-Host ("{0,10} {1,12:F1} {2,12:F1} {3,14:F1} {4,14}" -f `
                $n, $saveMs, $restoreMs, $sizeMiB, $after.PromptN)
        }
        finally { Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 800 }
    }
    Write-Host ""
    Write-Host "post-restore tok near the suffix length proves the restore was real, not a no-op."
}

Write-Host ""
Write-Host "prompt_n is the number of prompt tokens llama-server ACTUALLY evaluated."
Write-Host "A warm prompt_n near the suffix length means the prefix was reused from the slot."
Write-Host "Compare against session runtime plan §3.4.1 (OpenTail HotSession: 24.0x at 1K, 71.1x at 4K)."
