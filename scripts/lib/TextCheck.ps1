# Text sanity check for scripts/bench-league-recheck.ps1: a run that decodes at a good speed can still print nothing or garbage.
# Dot-source this file; it only defines functions.

# The CLI prints "Loading model...", "Model loaded in ...", then the prompt echoed straight into the generated text, a blank line and
# the "Prefill: ... | Decode: ..." line. Returns just the generated text ('' when the layout is not recognised).
function Get-GeneratedText([string]$stdout, [string]$promptText) {
    $lines = $stdout -split "`r?`n"
    $start = -1; $end = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($start -lt 0 -and $lines[$i] -match '^Model loaded in ') { $start = $i + 1 }
        if ($lines[$i] -match '^Prefill:\s*\d+\s*tokens') { $end = $i; break }
    }
    if ($start -lt 0 -or $end -lt $start) { return '' }
    $body = (($lines[$start..($end - 1)]) -join "`n").Trim()
    $prompt = $promptText.Trim()
    if ($prompt.Length -gt 0 -and $body.StartsWith($prompt, [System.StringComparison]::Ordinal)) { $body = $body.Substring($prompt.Length) }
    return $body.Trim()
}

# '' when the text looks like text, else one flag: EMPTY, NON-TEXT (under half of the visible characters are letters or digits) or
# REPEATING (at least 6 words with under 35% distinct, or a long text of at most 3 distinct characters).
function Get-TextFlag([string]$text) {
    $t = $text.Trim()
    if ($t.Length -eq 0) { return 'EMPTY' }
    $visible = ($t -replace '\s', '')
    $alnum = ([regex]::Matches($visible, '[\p{L}\p{N}]')).Count
    if ($visible.Length -gt 0 -and $alnum / $visible.Length -lt 0.5) { return 'NON-TEXT' }
    if ($visible.Length -ge 16 -and ($visible.ToCharArray() | Select-Object -Unique).Count -le 3) { return 'REPEATING' }
    $words = @($t -split '\s+' | Where-Object { $_ })
    if ($words.Count -ge 6 -and (($words | ForEach-Object { $_.ToLowerInvariant() } | Select-Object -Unique).Count / $words.Count) -lt 0.35) { return 'REPEATING' }
    return ''
}

# The first 80 characters on one line, safe inside a markdown table cell.
function Format-TextStart([string]$text) {
    $s = ($text -replace '\s+', ' ').Trim()
    if ($s.Length -gt 80) { $s = $s.Substring(0, 80).TrimEnd() }
    return ($s -replace '\|', '/')
}
