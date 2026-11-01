# A minimal OTLP/HTTP receiver: accepts any POST on 127.0.0.1:4318 with an empty 200 and counts what it got.
param([string]$CountFile)
$l = [System.Net.HttpListener]::new()
$l.Prefixes.Add('http://127.0.0.1:4318/')
$l.Start()
$requests = 0; $bytes = 0L
while ($l.IsListening) {
    $ctx = $l.GetContext()
    $ms = [System.IO.MemoryStream]::new()
    $ctx.Request.InputStream.CopyTo($ms)
    $requests++; $bytes += $ms.Length
    $ctx.Response.StatusCode = 200
    $ctx.Response.ContentType = 'application/x-protobuf'
    $ctx.Response.ContentLength64 = 0
    $ctx.Response.Close()
    Set-Content -LiteralPath $CountFile -Value "$requests $bytes $($ctx.Request.Url.AbsolutePath)"
}
