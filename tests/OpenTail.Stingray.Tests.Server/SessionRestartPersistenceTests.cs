using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// The real-model tests in the Server test surface: loads an actual GGUF checkpoint to verify
/// session behavior that a fake engine can't meaningfully exercise (restart persistence, genuine
/// concurrent continuous batching). Split out from OpenTail.Stingray.Tests.Server.Fast (see that
/// project's ServerTests.cs, which covers everything else via FakeInferenceEngine) because these
/// are the only tests in the whole surface that touch a real model — keeping them isolated lets
/// the other ~360 fake-engine tests run in parallel instead of paying real CPU-inference cost as
/// a serialization tax on every run.
/// </summary>
public sealed class SessionRestartPersistenceTests
{
    [Fact]
    public async Task SessionLifecycle_RealCpuGguf_RestoresAcrossServerRestart()
    {
        string? modelPath = FindSessionReferenceModel();
        // Skip, never silently pass. This is the acceptance test behind a capability the server
        // reports as available:true, so a green run that never executed it would read as "restart
        // continuation is verified" when nothing ran — and CI has no models/ directory. Assert.Skip
        // makes the absence visible in the summary, matching the Sessions-layer counterpart
        // ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay.
        Assert.SkipUnless(modelPath is not null,
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for the server session restart acceptance test.");

        string storage = Path.Combine(Path.GetTempPath(), $"opentail_server_session_{Guid.NewGuid():N}");
        try
        {
            string id;
            long revision;
            using (var first = CreatePersistentSessionServer(modelPath, storage))
            using (var client = first.CreateClient())
            {
                var create = await client.PostAsync("/v1/sessions", content: null);
                Assert.Equal(HttpStatusCode.Created, create.StatusCode);
                using var created = JsonDocument.Parse(await create.Content.ReadAsStringAsync());
                id = created.RootElement.GetProperty("id").GetString()!;

                var turn = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", new
                {
                    append_prompt = "The capital of France is",
                    expected_revision = 0,
                    max_tokens = 1,
                    temperature = 0f,
                });
                Assert.Equal(HttpStatusCode.OK, turn.StatusCode);
                using var result = JsonDocument.Parse(await turn.Content.ReadAsStringAsync());
                Assert.Equal("completed", result.RootElement.GetProperty("state").GetString());
                revision = result.RootElement.GetProperty("session").GetProperty("committed_revision").GetInt64();
                Assert.True(revision > 0);
            }

            // A fresh host has no hot state. GET must load the persisted manifest + KV packs,
            // not merely retrieve the first factory's in-memory session.
            using (var second = CreatePersistentSessionServer(modelPath, storage))
            using (var client = second.CreateClient())
            {
                var restored = await client.GetAsync($"/v1/sessions/{id}");
                Assert.Equal(HttpStatusCode.OK, restored.StatusCode);
                using var snapshot = JsonDocument.Parse(await restored.Content.ReadAsStringAsync());
                Assert.Equal(revision, snapshot.RootElement.GetProperty("committed_revision").GetInt64());

                var continuation = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", new
                {
                    append_prompt = " Paris.",
                    expected_revision = revision,
                    max_tokens = 1,
                    temperature = 0f,
                });
                Assert.Equal(HttpStatusCode.OK, continuation.StatusCode);
            }
        }
        finally
        {
            if (Directory.Exists(storage)) Directory.Delete(storage, recursive: true);
        }
    }

    [Fact]
    public async Task ConcurrentSessions_RealCpuGguf_ContinuousBatchingKeepsSessionsIndependent()
    {
        // docs/032-multi-model-inference-runtime-plan.md Phase 4 acceptance: "N sessions on one
        // runtime behave exactly as today's same-model concurrency." The engine here is
        // constructed via ModelRuntimeManager.AcquireAsync (ServiceCollectionExtensions), not the
        // old direct InferenceEngineLoader.Load call — this proves that change didn't alter the
        // object continuous batching actually runs, by driving real concurrent generation through
        // it and checking for cross-session KV/state corruption, not just checking it doesn't throw.
        string? modelPath = FindSessionReferenceModel();
        Assert.SkipUnless(modelPath is not null,
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for the concurrent-sessions acceptance test.");

        using var server = CreateConcurrentSessionServer(modelPath);
        using var client = server.CreateClient();

        // Distinct, low-perplexity prompts whose greedy (temperature=0) continuation is
        // predictable enough that a wrong answer reveals cross-session interference rather than
        // just "the model was uncertain."
        (string Prompt, string ExpectedSubstring)[] scenarios =
        [
            ("The capital of France is", "Paris"),
            ("2 + 2 =", "4"),
            ("The opposite of hot is", "cold"),
        ];

        async Task<string> RunSessionAsync((string Prompt, string ExpectedSubstring) scenario)
        {
            var create = await client.PostAsync("/v1/sessions", content: null);
            Assert.Equal(HttpStatusCode.Created, create.StatusCode);
            using var created = JsonDocument.Parse(await create.Content.ReadAsStringAsync());
            string id = created.RootElement.GetProperty("id").GetString()!;

            var turn = await client.PostAsJsonAsync($"/v1/sessions/{id}/turns", new
            {
                append_prompt = scenario.Prompt,
                expected_revision = 0,
                max_tokens = 6,
                temperature = 0f,
            });
            Assert.Equal(HttpStatusCode.OK, turn.StatusCode);
            using var result = JsonDocument.Parse(await turn.Content.ReadAsStringAsync());
            Assert.Equal("completed", result.RootElement.GetProperty("state").GetString());
            return result.RootElement.GetProperty("text").GetString()!;
        }

        // Genuinely concurrent — all three requests in flight together against the one shared
        // continuous-batching engine, not run one at a time.
        var results = await Task.WhenAll(scenarios.Select(RunSessionAsync));

        for (int i = 0; i < scenarios.Length; i++)
            Assert.Contains(scenarios[i].ExpectedSubstring, results[i], StringComparison.OrdinalIgnoreCase);
    }

    private static WebApplicationFactory<Program> CreateConcurrentSessionServer(string modelPath) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            services.Configure<OpenTailStingrayServerOptions>(options =>
            {
                options.ModelPath = modelPath;
                options.Backend = ServerBackend.Cpu;
                options.NGpuLayers = 0;
                options.ContextSize = 512;
                options.MaxBatchSize = 4;
                options.EnableSessions = true;
            })));

    private static WebApplicationFactory<Program> CreatePersistentSessionServer(string modelPath, string storage) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            services.Configure<OpenTailStingrayServerOptions>(options =>
            {
                options.ModelPath = modelPath;
                options.Backend = ServerBackend.Cpu;
                options.NGpuLayers = 0;
                options.ContextSize = 512;
                options.MaxBatchSize = 1;
                options.EnableSessions = true;
                options.SessionStorageDirectory = storage;
            })));

    private static string? FindSessionReferenceModel()
    {
        string directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            string candidate = Path.Combine(directory, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(directory);
            if (parent is null) break;
            directory = parent.FullName;
        }
        return null;
    }
}
