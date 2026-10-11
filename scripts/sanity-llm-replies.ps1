# Safety check after many engine changes: does every installed LLM checkpoint still return a COHERENT reply? No timing, smallest first, one process at a time.
# Memory guard: a checkpoint is skipped (not run) when its file size exceeds the available physical RAM minus -ReserveGB; the CLI's own preflight is NOT bypassed.
# Coherent = exit 0, a decode line, and the reply contains the expected word (Paris) for a trivial question. A failure shows the reply for a human to judge.
# Resumable: results are appended to -Csv; checkpoints already in it are skipped. Usage: pwsh scripts/sanity-llm-replies.ps1 -Inventory inventory.csv -Csv results.csv
param([string]$Inventory = '', [string]$Csv = '', [double]$ReserveGB = 6, [int]$TimeoutMinutes = 12, [int]$Tokens = 96, [string]$Backend = 'cpu')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Inventory) { $Inventory = Join-Path ([IO.Path]::GetTempPath()) 'sanity-inventory.csv'; & (Join-Path $PSScriptRoot 'list-checkpoints.ps1') -Out $Inventory | Out-Null }
if (-not $Csv) { $Csv = Join-Path ([IO.Path]::GetTempPath()) 'sanity-llm-results.csv' }
$exe = Join-Path $repo 'src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe'
if (-not (Test-Path $exe)) { throw "build first: dotnet build src/OpenTail.Stingray.Cli -c Release" }
$nonLlm = '^(bert|nomic-bert|xlm-roberta|jina-bert.*|neo-bert|modern-bert|clip|whisper|flux|wan|sd3?|stable-diffusion|qwen[_-]image|z-?image|ltxv|hunyuan_video|hyvid|parakeet|chatterbox.*|kokoro.*|parler.*|rvc|silero.*|fish.*|cosyvoice.*|f5|melo|piper|qwen3[-_]?asr|qwen3-tts.*|forced-aligner|dac|snac|mimi|wavtokenizer|t5encoder|stable-audio|acestep|audiocpp|canary-ctc|paraformer|asr|vocos|diffusion-gemma|deepseek2-ocr|paddleocr)$'
$prompt = 'What is the capital of France? Answer in one short sentence.'
$done = @{}
if (Test-Path $Csv) { foreach ($r in (Import-Csv $Csv)) { $done[$r.path] = 1 } } else { 'path,name,sizeGB,arch,status,note' | Set-Content $Csv -Encoding utf8 }
$rows = Import-Csv $Inventory | Where-Object { $_.ext -eq '.gguf' -and $_.name -notmatch '^mmproj|-mmproj' } | Sort-Object { [double]$_.sizeGB }
function Add-Row($r, $status, $note) {
    $note = ($note -replace '[\r\n]+', ' ' -replace '"', "'")
    if ($note.Length -gt 220) { $note = $note.Substring(0, 220) }
    '"{0}","{1}",{2},"{3}",{4},"{5}"' -f $r.path, $r.name, $r.sizeGB, $r.arch, $status, $note | Add-Content $Csv -Encoding utf8
}
foreach ($r in $rows) {
    if ($done.ContainsKey($r.path)) { continue }
    if ($r.arch -match $nonLlm) { Add-Row $r 'NOT-LLM' "arch $($r.arch)"; continue }
    $availGB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ([double]$r.sizeGB -gt ($availGB - $ReserveGB)) { Add-Row $r 'SKIPPED-MEM' ("size {0} GB > available {1:F1} GB - reserve {2} GB" -f $r.sizeGB, $availGB, $ReserveGB); continue }
    $psi = [Diagnostics.ProcessStartInfo]::new($exe)
    foreach ($a in @('-m', $r.path, '-p', $prompt, '-n', "$Tokens", '--temp', '0', '-g', $(if ($Backend -eq 'cpu') { '0' } else { '-1' }))) { [void]$psi.ArgumentList.Add($a) }
    $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false
    $p = [Diagnostics.Process]::Start($psi)
    $err = $p.StandardError.ReadToEndAsync(); $out = $p.StandardOutput.ReadToEndAsync()
    if (-not $p.WaitForExit($TimeoutMinutes * 60000)) { try { $p.Kill($true) } catch {}; Add-Row $r 'TIMEOUT' "after $TimeoutMinutes min"; continue }
    $text = $out.Result; $all = $text + "`n" + $err.Result
    $reply = (($text -split "`n") | Where-Object { $_ -notmatch '^\s*\[|Prefill:|Decode:' }) -join ' '
    $reply = ($reply -replace '\s+', ' ').Trim()
    if ($all -notmatch 'Decode:\s*\d+\s*tokens') {
        $line = ($all -split "`n" | Where-Object { $_ -match 'rror|not supported|unsupported|refus|Unknown|cannot|failed|Exception|preflight' } | Select-Object -First 1)
        Add-Row $r 'NO-REPLY' ($(if ($line) { $line.Trim() } else { "exit $($p.ExitCode)" })); continue
    }
    if ($reply -match 'Paris') { Add-Row $r 'PASS' $reply } else { Add-Row $r 'CHECK-REPLY' $reply }
}
Import-Csv $Csv | Group-Object status | ForEach-Object { "{0}: {1}" -f $_.Name, $_.Count }
