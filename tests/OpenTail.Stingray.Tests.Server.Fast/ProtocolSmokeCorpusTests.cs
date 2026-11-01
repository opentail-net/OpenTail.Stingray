using System.Text.Json;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 5 E2: a small request/response corpus (<c>Corpus/protocol-smoke.json</c>) replayed against the OpenAI, Anthropic, Responses,
/// speech, image and rerank routes with the fake engine. Each case pins the status; each refusal also pins the error shape clients read
/// (flat <c>type</c> and <c>message</c> plus the nested <c>error</c> object) and that the message names the field. The parameter audits of
/// 2026-10-31 and 2026-11-01 found their defects by hand; a validation that is removed or an error that loses its body fails this test.
/// </summary>
public sealed class ProtocolSmokeCorpusTests
{
    private static string CorpusPath => Path.Combine(AppContext.BaseDirectory, "Corpus", "protocol-smoke.json");

    [Fact]
    public async Task EveryCorpusCase_GetsTheRecordedStatus_AndRefusalsKeepTheirErrorShape()
    {
        using var doc = JsonDocument.Parse(await File.ReadAllTextAsync(CorpusPath));
        var cases = doc.RootElement.EnumerateArray().ToList();
        Assert.True(cases.Count >= 15, $"the corpus has only {cases.Count} cases");

        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"))));
        using var client = factory.CreateClient();

        var failures = new List<string>();
        foreach (var c in cases)
        {
            string name = c.GetProperty("name").GetString()!;
            string payload = c.TryGetProperty("rawBody", out var raw) ? raw.GetString()! : c.GetProperty("body").GetRawText();
            using var request = new StringContent(payload, System.Text.Encoding.UTF8, "application/json");
            using var response = await client.PostAsync(c.GetProperty("route").GetString(), request);
            string text = await response.Content.ReadAsStringAsync();

            int expected = c.GetProperty("status").GetInt32();
            if ((int)response.StatusCode != expected) { failures.Add($"{name}: status {(int)response.StatusCode}, expected {expected}. Body: {Trim(text)}"); continue; }
            if (expected < 400) continue;

            if (string.IsNullOrWhiteSpace(text)) { failures.Add($"{name}: the {expected} has no body"); continue; }
            try
            {
                using var body = JsonDocument.Parse(text);
                string? type = body.RootElement.TryGetProperty("type", out var t) ? t.GetString() : null;
                string? message = body.RootElement.TryGetProperty("message", out var m) ? m.GetString() : null;
                string? nested = body.RootElement.TryGetProperty("error", out var e) && e.TryGetProperty("message", out var em) ? em.GetString() : null;
                string expectedType = c.GetProperty("errorType").GetString()!;
                if (type != expectedType) failures.Add($"{name}: error type '{type}', expected '{expectedType}'");
                if (string.IsNullOrWhiteSpace(message)) failures.Add($"{name}: no flat message");
                if (string.IsNullOrWhiteSpace(nested)) failures.Add($"{name}: no nested error.message");
                if (c.TryGetProperty("messageContains", out var needle) && message is not null
                    && !message.Contains(needle.GetString()!, StringComparison.Ordinal))
                    failures.Add($"{name}: message '{message}' does not name '{needle.GetString()}'");
            }
            catch (JsonException) { failures.Add($"{name}: the {expected} body is not JSON: {Trim(text)}"); }
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private static string Trim(string s) => s.Length <= 160 ? s : s[..160] + "...";
}
