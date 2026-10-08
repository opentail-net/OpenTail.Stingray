using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Engine.Verification;

/// <summary>What <c>llama-server</c> returned for a raw-token completion.</summary>
public sealed record CompletionResult(int[] Tokens, string Content, int TokensEvaluated, int TokensPredicted, double[]? Margins = null);

/// <summary>
/// The pure, testable parts of capturing a golden from the vendored <c>llama-server</c> / <c>llama-tokenize</c>: building the request,
/// reading the responses, hashing the model. Reflection-free (JsonDocument, hand-built JSON) so it is NativeAOT-safe.
/// </summary>
public static class GoldenCaptureParsing
{
    /// <summary>Request body for a greedy, cache-free completion of a raw token-id prompt.</summary>
    public static string BuildCompletionRequest(IReadOnlyList<int> promptTokens, int nPredict, int seed = 0)
    {
        var sb = new StringBuilder("{\"prompt\":[");
        for (int i = 0; i < promptTokens.Count; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(promptTokens[i].ToString(CultureInfo.InvariantCulture));
        }
        sb.Append("],\"n_predict\":").Append(nPredict.ToString(CultureInfo.InvariantCulture))
          .Append(",\"temperature\":0,\"top_k\":1,\"seed\":").Append(seed.ToString(CultureInfo.InvariantCulture))
          .Append(",\"repeat_penalty\":1.0,\"cache_prompt\":false,\"return_tokens\":true,\"n_probs\":2}");
        return sb.ToString();
    }

    public static CompletionResult ParseCompletion(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (!root.TryGetProperty("tokens", out var tokens) || tokens.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException("llama-server response has no 'tokens' array (was return_tokens honoured?).");
        var ids = tokens.EnumerateArray().Select(t => t.GetInt32()).ToArray();
        string content = root.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String ? c.GetString() ?? "" : "";
        int evaluated = root.TryGetProperty("tokens_evaluated", out var e) ? e.GetInt32() : -1;
        int predicted = root.TryGetProperty("tokens_predicted", out var p) ? p.GetInt32() : ids.Length;
        return new CompletionResult(ids, content, evaluated, predicted, ParseMargins(root, ids.Length));
    }

    /// <summary>top-1 minus top-2 log-probability per generated token from <c>completion_probabilities</c>; null when absent or incomplete.</summary>
    private static double[]? ParseMargins(JsonElement root, int count)
    {
        if (!root.TryGetProperty("completion_probabilities", out var probs) || probs.ValueKind != JsonValueKind.Array) return null;
        var margins = new List<double>(count);
        foreach (var p in probs.EnumerateArray())
        {
            if (!p.TryGetProperty("top_logprobs", out var top) || top.ValueKind != JsonValueKind.Array || top.GetArrayLength() < 2) return null;
            var e = top.EnumerateArray().GetEnumerator();
            e.MoveNext(); double first = e.Current.GetProperty("logprob").GetDouble();
            e.MoveNext(); double second = e.Current.GetProperty("logprob").GetDouble();
            margins.Add(Math.Round(first - second, 4));
        }
        return margins.Count == count ? margins.ToArray() : null;
    }

    private static readonly Regex s_idList = new(@"\[\s*(-?\d+(\s*,\s*-?\d+)*)?\s*\]", RegexOptions.Compiled);

    /// <summary>Reads the first <c>[1, 2, 3]</c> list from <c>llama-tokenize --ids</c> output (other lines are logging).</summary>
    public static int[] ParseTokenizeIds(string stdout)
    {
        var m = s_idList.Match(stdout);
        if (!m.Success) throw new InvalidDataException("llama-tokenize printed no token id list.");
        return m.Groups[1].Success && m.Groups[1].Length > 0
            ? m.Groups[1].Value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                .Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray()
            : [];
    }

    private static readonly Regex s_version = new(@"version:\s*(.+)", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>The build string from <c>llama-server --version</c> output, e.g. <c>10306 (6b5c2efb4)</c>.</summary>
    public static string? ParseVersion(string text)
    {
        var m = s_version.Match(text);
        return m.Success ? m.Groups[1].Value.Trim() : null;
    }

    /// <summary>Lowercase hex SHA-256 of a file, streamed (a 12 GB checkpoint is never held in memory).</summary>
    public static string Sha256Hex(string path, Action<double>? progress = null)
    {
        using var sha = SHA256.Create();
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        var buffer = new byte[1 << 20];
        long total = fs.Length, done = 0;
        int read;
        while ((read = fs.Read(buffer, 0, buffer.Length)) > 0)
        {
            sha.TransformBlock(buffer, 0, read, null, 0);
            done += read;
            progress?.Invoke(total == 0 ? 1 : (double)done / total);
        }
        sha.TransformFinalBlock([], 0, 0);
        return Convert.ToHexString(sha.Hash!).ToLowerInvariant();
    }
}
