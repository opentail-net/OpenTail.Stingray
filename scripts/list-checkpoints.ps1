# Inventory every checkpoint under the known model roots (every drive's _models, K:\_other_models, the repo's models/) as CSV:
#   path, name, sizeGB, ext, arch (general.architecture read from the GGUF header), kind (llm | mmproj | other-gguf | non-gguf)
# De-duplicates by (file name, size): the repo's models/_models is the same data as F:\_models, and some files exist on several drives.
# Usage: pwsh scripts/list-checkpoints.ps1 [-Out inventory.csv]
param([string]$Out = '')

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

# Architectures that are not text-generation models (they need another harness: embeddings, ASR/TTS, diffusion, vision encoders).
# Anything else is attempted by the text CLI and recorded as UNSUPPORTED if it does not produce a Prefill line.
$nonLlm = @('bert', 'nomic-bert', 'xlm-roberta', 'jina-bert-v2', 'jina-bert-v3', 'neo-bert', 'modern-bert', 'clip', 'whisper', 'flux', 'wan', 'stable-diffusion',
    'sd', 'sd3', 'qwen_image', 'qwen-image', 'z-image', 'zimage', 'ltxv', 'hunyuan_video', 'hyvid', 'parakeet', 'chatterbox', 'kokoro', 'parler', 'rvc',
    'silero', 'fish', 'cosyvoice', 'f5', 'melo', 'piper', 'qwen3-asr', 'qwen3_asr', 'forced-aligner', 'dac', 'snac', 'mimi', 'wavtokenizer', 't5encoder', 'stable-audio', 'acestep',
    'audiocpp', 'fish-speech', 'qwen3-tts', 'qwen3-tts-tokenizer', 'kokoro-voice', 'parler-tts', 'cosyvoice3-2512', 'chatterbox-s3gen', 'canary-ctc', 'paraformer',
    'qwen3asr', 'silerovad', 'asr', 'vocos', 'diffusion-gemma', 'deepseek2-ocr', 'paddleocr')

function Read-GgufArch([string]$path) {
    try {
        $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        try {
            $br = [System.IO.BinaryReader]::new($fs)
            if ($br.ReadUInt32() -ne 0x46554747) { return '' }          # "GGUF"
            [void]$br.ReadUInt32()                                      # version
            [void]$br.ReadUInt64()                                      # tensor count
            $kv = $br.ReadUInt64()
            function Skip([int]$t) {
                switch ($t) {
                    { $_ -in 0, 1, 7 } { [void]$br.ReadBytes(1) }
                    { $_ -in 2, 3 } { [void]$br.ReadBytes(2) }
                    { $_ -in 4, 5, 6 } { [void]$br.ReadBytes(4) }
                    { $_ -in 10, 11, 12 } { [void]$br.ReadBytes(8) }
                    8 { $l = $br.ReadUInt64(); [void]$fs.Seek([long]$l, 'Current') }
                    9 { $it = $br.ReadUInt32(); $n = $br.ReadUInt64(); for ($i = 0; $i -lt $n; $i++) { Skip $it } }
                }
            }
            for ($k = 0; $k -lt [Math]::Min($kv, 64); $k++) {
                $kl = $br.ReadUInt64(); $key = [System.Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$kl))
                $t = $br.ReadUInt32()
                if ($key -eq 'general.architecture' -and $t -eq 8) {
                    $vl = $br.ReadUInt64(); return [System.Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$vl))
                }
                Skip $t
            }
            return ''
        } finally { $fs.Dispose() }
    } catch { return '' }
}

$roots = [System.Collections.Generic.List[string]]::new()
foreach ($d in (Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name.Length -eq 1 })) {
    foreach ($n in '_models', '_other_models') { $p = "$($d.Name):\$n"; if (Test-Path $p) { $roots.Add($p) } }
}
foreach ($p in @("$repo\models", "$repo\models\_models")) { if (Test-Path $p) { $roots.Add($p) } }

$seen = @{}
$rows = foreach ($r in ($roots | Select-Object -Unique)) {
    foreach ($f in Get-ChildItem -Path $r -Recurse -File -ErrorAction SilentlyContinue -Include *.gguf, *.safetensors, *.onnx) {
        if ($f.Name -match '-0000[2-9]-of-\d+\.gguf$') { continue }   # later shards: the first shard is the entry point
        $key = "$($f.Name)|$($f.Length)"
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        $arch = ''; $kind = 'non-gguf'
        if ($f.Extension -eq '.gguf') {
            $arch = Read-GgufArch $f.FullName
            $kind = if ($f.Name -match 'mmproj') { 'mmproj' } elseif ($arch -and ($nonLlm -contains $arch)) { 'other-gguf' } elseif ($arch -eq '') { 'other-gguf' } else { 'llm' }
        }
        [pscustomobject]@{ path = $f.FullName; name = $f.Name; sizeGB = [Math]::Round($f.Length / 1GB, 2); ext = $f.Extension; arch = $arch; kind = $kind }
    }
}
$rows = $rows | Sort-Object kind, sizeGB
if ($Out) { $rows | Export-Csv -Path $Out -NoTypeInformation -Encoding utf8; Write-Output "wrote $($rows.Count) rows to $Out" } else { $rows }
