using System.Text;
using OpenTail.Stingray.Core.Grammar;

namespace OpenTail.Stingray.Cli;

public sealed partial class RunCommand
{
    /// <summary>
    /// <c>--verbose-prompt</c>: the log-probabilities of the K most likely first tokens, so two engines' distributions can be
    /// compared numerically (a sampled token or an answer's wording hides differences that a top-K list shows).
    /// </summary>
    private static void PrintTopLogProbs(ReadOnlySpan<float> logits, GgufTokenizer tok, int k)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) sum += Math.Exp(logits[i] - max);
        double logZ = max + Math.Log(sum);

        var top = new List<(int Id, float Logit)>(k + 1);
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (top.Count == k && v <= top[^1].Logit) continue;
            int at = top.Count;
            while (at > 0 && v > top[at - 1].Logit) at--;
            top.Insert(at, (i, v));
            if (top.Count > k) top.RemoveAt(k);
        }
        foreach (var (id, logit) in top)
            Console.Error.WriteLine($"[DBG-TOPK] id={id} '{tok.Decode([id]).Replace("\n", "\n")}' logprob={logit - logZ:F4}");
    }

    /// <summary>Count non-overlapping occurrences of <paramref name="needle"/> in <paramref name="haystack"/>.</summary>
    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0, idx = 0;
        while ((idx = haystack.IndexOf(needle, idx, StringComparison.Ordinal)) >= 0)
        {
            count++;
            idx += needle.Length;
        }
        return count;
    }

    /// <summary>
    /// Formats the <paramref name="k"/> highest-logit token candidates as
    /// <c>idx(value) idx(value) …</c> (descending by value, ties broken by lower index) for the
    /// <c>--verbose-prompt</c> debug line. Issue #155: a single O(V·k) pass over the logits span
    /// keeping the k best — no <c>logits.ToArray()</c> copy and no full O(V·logV) vocab sort per
    /// decode token (a ~1 MB alloc + sort on Gemma 4's 262144-token vocab that badly skewed
    /// decode t/s under <c>--verbose-prompt</c>).
    /// </summary>
    internal static string FormatTopLogits(ReadOnlySpan<float> logits, int k)
    {
        k = Math.Min(k, logits.Length);
        if (k <= 0) return "";

        // count-tracked insertion (not a value sentinel) so a real -Infinity logit still occupies
        // a slot and is printed — the old LINQ path showed it, and --verbose-prompt is exactly the
        // tool you reach for when a model emits non-finite garbage (#155 review). Stackalloc only
        // for small k (matches Sampler.FindKthLargest); heap-fall back guards against a large k
        // passed by some other caller stack-overflowing (#155 review).
        Span<float> bestVal = k <= 256 ? stackalloc float[k] : new float[k];
        Span<int> bestIdx = k <= 256 ? stackalloc int[k] : new int[k];
        int count = 0;

        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            // Skip only when the set is full AND v doesn't outrank the current worst. Ranking
            // matches a stable OrderByDescending: higher value first, NaN sorts last, equal values
            // keep the earlier (lower) index — so a later equal value never displaces.
            if (count == k && !SortsBefore(v, bestVal[k - 1])) continue;

            int pos = count < k ? count : k - 1;
            while (pos > 0 && SortsBefore(v, bestVal[pos - 1]))
            {
                bestVal[pos] = bestVal[pos - 1];
                bestIdx[pos] = bestIdx[pos - 1];
                pos--;
            }
            bestVal[pos] = v;
            bestIdx[pos] = i;
            if (count < k) count++;
        }

        var sb = new System.Text.StringBuilder(k * 12);
        for (int j = 0; j < count; j++)
        {
            if (j > 0) sb.Append(' ');
            sb.Append(bestIdx[j]).Append('(').Append($"{bestVal[j]:F2}").Append(')');
        }
        return sb.ToString();

        // True when a ranks strictly above b in a descending sort (higher value first; NaN is
        // least), matching Comparer<float> as used by OrderByDescending.
        static bool SortsBefore(float a, float b)
        {
            if (float.IsNaN(a)) return false;
            if (float.IsNaN(b)) return true;
            return a > b;
        }
    }

    // ── Tool calling (--tools / --tool-grammar) ────────────────────────────────

    /// <summary>
    /// Loads OpenAI-format tool definitions from a JSON file: a bare array of
    /// <c>{type:"function", function:{name, description, parameters}}</c> objects, or a
    /// <c>{ "tools": [ … ] }</c> wrapper. Returns the template-facing object graph (passed to the
    /// chat template's <c>tools</c> variable) and the parsed <see cref="ToolSchema"/>s (used to build
    /// the argument-grammar constraint). Eagerly materialised so the backing <see cref="JsonDocument"/>
    /// can be disposed here. Throws <see cref="FormatException"/> / <see cref="JsonException"/> on a
    /// malformed file.
    /// </summary>
    private static (IReadOnlyList<object?> Tools, List<ToolSchema> Schemas) LoadTools(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        JsonElement arr = root.ValueKind == JsonValueKind.Object && root.TryGetProperty("tools", out var t) ? t : root;
        if (arr.ValueKind != JsonValueKind.Array)
            throw new FormatException("expected a JSON array of tool definitions (or a { \"tools\": [ … ] } wrapper).");

        var tools = new List<object?>();
        var schemas = new List<ToolSchema>();
        foreach (var el in arr.EnumerateArray())
        {
            tools.Add(JsonToObject(el));   // detached object graph for the template
            if (el.TryGetProperty("function", out var fn) && fn.ValueKind == JsonValueKind.Object
                && fn.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String
                && n.GetString() is { Length: > 0 } name)
            {
                JsonElement? parameters = fn.TryGetProperty("parameters", out var p) ? p : null;
                schemas.Add(ToolSchema.FromOpenAiFunction(name, parameters));
            }
        }
        if (schemas.Count == 0)
            throw new FormatException("no tool with a function.name was found.");
        return (tools, schemas);
    }

    /// <summary>Recursively converts a <see cref="JsonElement"/> into a detached object graph
    /// (<see cref="Dictionary{TKey,TValue}"/> / <see cref="List{T}"/> / scalar) the Jinja engine
    /// consumes — reflection-free, so it survives NativeAOT trimming.</summary>
    private static object? JsonToObject(JsonElement el) => el.ValueKind switch
    {
        JsonValueKind.Object => el.EnumerateObject().ToDictionary(p => p.Name, p => JsonToObject(p.Value), StringComparer.Ordinal),
        JsonValueKind.Array  => el.EnumerateArray().Select(JsonToObject).ToList(),
        JsonValueKind.String => el.GetString(),
        JsonValueKind.Number => el.TryGetInt64(out long l) ? l : el.TryGetDouble(out double d) ? d : (object?)el.GetRawText(),
        JsonValueKind.True   => true,
        JsonValueKind.False  => false,
        _                    => null,
    };

    /// <summary>Parses the captured raw output with the model's tool-call adapter and prints the
    /// structured calls (name + JSON arguments). No-op when the model emitted no tool call.</summary>
    private static void PrintToolCalls(GgufTokenizer tok, List<int> tokens)
    {
        if (tokens.Count == 0) return;
        var (_, calls) = ToolCallAdapterRegistry.Get(s_arch).Parse(tok.Decode(tokens));
        if (calls.Count == 0) return;
        AnsiConsole.MarkupLine($"\n[green]Parsed {calls.Count} tool call(s):[/]");
        foreach (var c in calls)
            AnsiConsole.MarkupLine($"  [bold]{Markup.Escape(c.Name)}[/]({Markup.Escape(RenderArgs(c.Arguments))})");
    }

    /// <summary>Renders a parsed tool call's argument object as compact JSON via a manual
    /// <see cref="Utf8JsonWriter"/> walk (no reflection-based serialization → AOT-safe).</summary>
    private static string RenderArgs(IReadOnlyDictionary<string, object?> args)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();
        using (var w = new Utf8JsonWriter(buffer))
            WriteJsonValue(w, args);
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static void WriteJsonValue(Utf8JsonWriter w, object? value)
    {
        switch (value)
        {
            case null:                                       w.WriteNullValue(); break;
            case bool b:                                     w.WriteBooleanValue(b); break;
            case string s:                                   w.WriteStringValue(s); break;
            case long l:                                     w.WriteNumberValue(l); break;
            case int i:                                      w.WriteNumberValue(i); break;
            case double d:                                   w.WriteNumberValue(d); break;
            case IReadOnlyDictionary<string, object?> map:
                w.WriteStartObject();
                foreach (var kv in map) { w.WritePropertyName(kv.Key); WriteJsonValue(w, kv.Value); }
                w.WriteEndObject();
                break;
            case System.Collections.IEnumerable seq:
                w.WriteStartArray();
                foreach (var item in seq) WriteJsonValue(w, item);
                w.WriteEndArray();
                break;
            default:                                         w.WriteStringValue(value.ToString() ?? ""); break;
        }
    }
}
