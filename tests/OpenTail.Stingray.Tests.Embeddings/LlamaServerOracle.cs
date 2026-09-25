using System.Diagnostics;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text.Json;

namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// Runs the vendored <c>tools/llama.cpp/llama-server.exe</c> on a GGUF as an independent oracle for
/// embeddings (<c>--embedding</c>, OpenAI <c>/v1/embeddings</c>) and rerank scores (<c>--rerank</c>,
/// <c>/v1/rerank</c>). The server is started on a free local port and killed on dispose.
/// </summary>
internal sealed class LlamaServerOracle : IDisposable
{
    private readonly Process _process;
    private readonly HttpClient _http;

    private LlamaServerOracle(Process process, int port)
    {
        _process = process;
        _http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromMinutes(5) };
    }

    public static string? FindServerExe() =>
        BertEncoderOnnxParityTests.FindRepoDir("tools/llama.cpp") is { } d && File.Exists(Path.Combine(d, "llama-server.exe"))
            ? Path.Combine(d, "llama-server.exe")
            : null;

    /// <param name="mode">"--embedding" or "--rerank".</param>
    public static LlamaServerOracle Start(string exe, string gguf, string mode, string? pooling = null, IEnumerable<string>? extraArgs = null)
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();

        var psi = new ProcessStartInfo(exe)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (var a in new[] { "-m", gguf, mode, "--port", port.ToString(System.Globalization.CultureInfo.InvariantCulture),
                     "--host", "127.0.0.1", "-c", "8192", "-b", "8192", "-ub", "8192", "-ngl", "0", "--no-webui" })
            psi.ArgumentList.Add(a);
        if (pooling is not null) { psi.ArgumentList.Add("--pooling"); psi.ArgumentList.Add(pooling); }
        foreach (var a in extraArgs ?? []) psi.ArgumentList.Add(a);

        var process = Process.Start(psi) ?? throw new InvalidOperationException("llama-server did not start");
        process.OutputDataReceived += (_, _) => { };
        process.ErrorDataReceived += (_, _) => { };
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        var oracle = new LlamaServerOracle(process, port);
        var deadline = DateTime.UtcNow.AddMinutes(3);
        while (DateTime.UtcNow < deadline)
        {
            if (process.HasExited) throw new InvalidOperationException($"llama-server exited with {process.ExitCode}");
            try
            {
                using var r = oracle._http.GetAsync("/health").GetAwaiter().GetResult();
                if (r.IsSuccessStatusCode) return oracle;
            }
            catch (HttpRequestException) { }
            Thread.Sleep(250);
        }
        oracle.Dispose();
        throw new TimeoutException("llama-server did not become healthy");
    }

    private JsonDocument Post(string path, string json)
    {
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using var r = _http.PostAsync(path, content).GetAwaiter().GetResult();
        string body = r.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        if (!r.IsSuccessStatusCode) throw new InvalidOperationException($"{path} -> {(int)r.StatusCode}: {body}");
        return JsonDocument.Parse(body);
    }

    private static string JsonString(string s)
    {
        var sb = new StringBuilder("\"");
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", System.Globalization.CultureInfo.InvariantCulture));
                    else sb.Append(c);
                    break;
            }
        }
        return sb.Append('"').ToString();
    }

    /// <summary>OpenAI-style embeddings (llama-server L2-normalizes pooled embeddings here).</summary>
    public float[][] Embeddings(IReadOnlyList<string> inputs)
    {
        using var doc = Post("/v1/embeddings", "{\"input\":[" + string.Join(",", inputs.Select(JsonString)) + "]}");
        return doc.RootElement.GetProperty("data").EnumerateArray()
            .OrderBy(e => e.GetProperty("index").GetInt32())
            .Select(e => e.GetProperty("embedding").EnumerateArray().Select(v => v.GetSingle()).ToArray())
            .ToArray();
    }

    /// <summary>Embeds <paramref name="inputs"/> in one request and returns the server's reported prompt token count.</summary>
    public int EmbedAndCountTokens(IReadOnlyList<string> inputs)
    {
        using var doc = Post("/v1/embeddings", "{\"input\":[" + string.Join(",", inputs.Select(JsonString)) + "]}");
        return doc.RootElement.GetProperty("usage").GetProperty("prompt_tokens").GetInt32();
    }

    /// <summary>Raw relevance score per document, in input order.</summary>
    public float[] Rerank(string query, IReadOnlyList<string> documents)
    {
        using var doc = Post("/v1/rerank", "{\"query\":" + JsonString(query) + ",\"documents\":[" + string.Join(",", documents.Select(JsonString)) + "]}");
        var scores = new float[documents.Count];
        foreach (var r in doc.RootElement.GetProperty("results").EnumerateArray())
            scores[r.GetProperty("index").GetInt32()] = r.GetProperty("relevance_score").GetSingle();
        return scores;
    }

    public void Dispose()
    {
        _http.Dispose();
        try { if (!_process.HasExited) _process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
        _process.WaitForExit(10000);
        _process.Dispose();
    }
}
