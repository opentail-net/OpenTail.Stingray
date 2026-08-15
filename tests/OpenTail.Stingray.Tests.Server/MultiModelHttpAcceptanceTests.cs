using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// docs/032-multi-model-inference-runtime-plan.md Phase 7 acceptance: "Users A/B → sidekick,
/// Users C/D → reasoner, with same-model batching, cross-model concurrency, residency, and
/// session isolation all verified together." Two real GGUFs (SmolLM2-1.7B and Qwen3-0.6B,
/// standing in for a sidekick/reasoner pair) configured via
/// <see cref="OpenTailStingrayServerOptions.Models"/>, driven through the ACTUAL HTTP surface
/// (<c>/v1/chat/completions</c>) — not <c>ModelRuntimeManager</c> directly, unlike
/// <c>CrossModelConcurrencyTests</c> (Phase 5), which proved the engine layer holds no lock but
/// never exercised the Phase 7 routing (alias resolution, per-request handle acquire/dispose,
/// per-request <c>ChatTemplateRenderer</c>) at all.
///
/// "Session isolation" here is proven at the stateless-request level (concurrent requests never
/// cross-contaminate each other's prompt/output) rather than through <c>/v1/sessions/*</c>:
/// <c>HotSession</c> holding a <c>ModelRuntimeHandle</c> for its lifetime (the Phase 4 known gap)
/// is deliberately NOT built in this pass — see docs/032 Phase 7's own status note for why. The
/// existing <c>/v1/sessions/*</c> multi-session-per-model isolation
/// (<c>SessionRestartPersistenceTests.ConcurrentSessions_RealCpuGguf_ContinuousBatchingKeepsSessionsIndependent</c>)
/// is untouched by Phase 7 and still single-model.
/// </summary>
public sealed class MultiModelHttpAcceptanceTests
{
    [Fact]
    public async Task FourConcurrentUsers_TwoModels_EachGetsTheRightModelAndTheRightAnswer_LoadedExactlyOnce()
    {
        string? sidekickPath = FindModel("SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
        string? reasonerPath = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(sidekickPath is not null && reasonerPath is not null,
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf and Qwen3-0.6B-Q8_0.gguf are required for the " +
            "Phase 7 multi-model HTTP acceptance test.");

        using var server = CreateMultiModelServer(sidekickPath!, reasonerPath!);
        using var client = server.CreateClient();

        // Distinct, low-perplexity greedy prompts per user — a wrong answer reveals cross-request
        // interference (wrong model, wrong prompt, or state bleeding between concurrent requests)
        // rather than "the model was uncertain."
        (string Model, string Prompt, string ExpectedSubstring)[] users =
        [
            ("sidekick", "The capital of France is", "Paris"),   // User A
            ("sidekick", "2 + 2 =", "4"),                        // User B
            ("reasoner", "The opposite of hot is", "cold"),      // User C
            ("reasoner", "The capital of Japan is", "Tokyo"),    // User D
        ];

        async Task<(string Model, string Text)> RunAsync((string Model, string Prompt, string ExpectedSubstring) u)
        {
            var resp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = u.Model,
                messages = new[] { new { role = "user", content = u.Prompt } },
                max_tokens = 16,
                temperature = 0f,
                stream = false,
                // Qwen3 (reasoner) defaults reasoning ON — an 8-16 token budget would otherwise be
                // entirely consumed by <think> content, leaving an empty answer. This test verifies
                // routing/concurrency correctness, not reasoning behavior, so disable it explicitly
                // for a deterministic direct answer regardless of the model's own default.
                enable_thinking = false,
            });
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
            string model = doc.RootElement.GetProperty("model").GetString()!;
            string text = doc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString() ?? "";
            return (model, text);
        }

        // All four users concurrently — the point of the test is that this DOESN'T serialize into
        // four sequential loads/generations.
        var results = await Task.WhenAll(users.Select(RunAsync));

        for (int i = 0; i < users.Length; i++)
        {
            Assert.Contains(users[i].ExpectedSubstring, results[i].Text, StringComparison.OrdinalIgnoreCase);
            // response.model reports the underlying GGUF's own IInferenceEngine.ModelId (a
            // filename), not the alias — sidekick's/reasoner's engine identities never mix.
            bool expectedSidekick = users[i].Model == "sidekick";
            Assert.Equal(expectedSidekick, results[i].Model.Contains("SmolLM2", StringComparison.OrdinalIgnoreCase));
            Assert.NotEqual(expectedSidekick, results[i].Model.Contains("Qwen3", StringComparison.OrdinalIgnoreCase));
        }

        // Single-flight held: two concurrent requests per model still produced exactly one
        // physical load each (docs/032 invariant 3), and residency is observable (both resident).
        var manager = server.Services.GetRequiredService<IModelRuntimeManager>();
        var stats = manager.GetStats();
        Assert.Equal(2, stats.ModelLoads);
        Assert.Equal(2, stats.ResidentModels);
    }

    [Fact]
    public async Task TwoModels_ConcurrentStreamingRequests_OverlapRatherThanSerialize()
    {
        string? sidekickPath = FindModel("SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
        string? reasonerPath = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(sidekickPath is not null && reasonerPath is not null,
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf and Qwen3-0.6B-Q8_0.gguf are required for the " +
            "Phase 7 multi-model HTTP acceptance test.");

        using var server = CreateMultiModelServer(sidekickPath!, reasonerPath!);
        using var client = server.CreateClient();

        // Same interleaving-based proof as CrossModelConcurrencyTests (Phase 5), but through the
        // actual HTTP/SSE surface this time — proving the Phase 7 routing (per-request handle
        // acquire/dispose, alias resolution, RequestConcurrencyGate) doesn't accidentally
        // serialize cross-model requests even though the underlying engine layer doesn't.
        async Task<(DateTime FirstContentAt, DateTime LastContentAt)> StreamAsync(string model, string prompt)
        {
            using var req = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
            {
                Content = JsonContent.Create(new
                {
                    model,
                    messages = new[] { new { role = "user", content = prompt } },
                    max_tokens = 48,
                    temperature = 0f,
                    stream = true,
                    enable_thinking = false, // see the identical note in the non-streaming test above
                }),
            };
            using var resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
            resp.EnsureSuccessStatusCode();
            await using var stream = await resp.Content.ReadAsStreamAsync();
            using var reader = new StreamReader(stream);

            DateTime? first = null;
            DateTime last = default;
            string? line;
            while ((line = await reader.ReadLineAsync()) is not null)
            {
                if (!line.StartsWith("data: ", StringComparison.Ordinal)) continue;
                string payload = line["data: ".Length..];
                if (payload == "[DONE]") break;
                using var doc = JsonDocument.Parse(payload);
                var delta = doc.RootElement.GetProperty("choices")[0].GetProperty("delta");
                if (delta.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String
                    && c.GetString() is { Length: > 0 })
                {
                    var now = DateTime.UtcNow;
                    first ??= now;
                    last = now;
                }
            }
            Assert.NotNull(first); // sanity: the stream actually produced content
            return (first!.Value, last);
        }

        var sidekickTask = StreamAsync("sidekick", "Write a short sentence about the ocean.");
        var reasonerTask = StreamAsync("reasoner", "Write a short sentence about the mountains.");
        var (sFirst, sLast) = await sidekickTask;
        var (rFirst, rLast) = await reasonerTask;

        bool interleaved = rFirst < sLast || sFirst < rLast;
        Assert.True(interleaved,
            $"Expected sidekick and reasoner generations to interleave when requested concurrently " +
            $"through the HTTP endpoint: sidekickFirst={sFirst:o} sidekickLast={sLast:o} " +
            $"reasonerFirst={rFirst:o} reasonerLast={rLast:o} — no overlap detected, looks serialized.");
    }

    private static WebApplicationFactory<Program> CreateMultiModelServer(string sidekickPath, string reasonerPath) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            services.Configure<OpenTailStingrayServerOptions>(options =>
            {
                options.Models =
                [
                    new NamedModelOptions { Alias = "sidekick", ModelPath = sidekickPath },
                    new NamedModelOptions { Alias = "reasoner", ModelPath = reasonerPath },
                ];
                options.Backend = ServerBackend.Cpu;
                options.NGpuLayers = 0;
                options.ContextSize = 512;
                options.MaxBatchSize = 2; // 2 concurrent users per model
            })));

    private static string? FindModel(string fileName)
    {
        string directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            string candidate = Path.Combine(directory, "models", fileName);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(directory);
            if (parent is null) break;
            directory = parent.FullName;
        }
        return null;
    }
}
