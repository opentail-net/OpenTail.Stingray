
namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// docs/032-multi-model-inference-runtime-plan.md Phase 7 follow-up acceptance: the
/// "session isolation... verified together" half of the original Phase 7 acceptance line that
/// <see cref="MultiModelHttpAcceptanceTests"/>'s own doc comment explicitly deferred (it proved
/// isolation only at the stateless-request level, not through <c>/v1/sessions/*</c>, because a
/// <c>HotSession</c> didn't hold a <c>ModelRuntimeHandle</c> yet). Two real GGUFs standing in for
/// a sidekick/reasoner pair, driven through the actual <c>/v1/sessions/*</c> HTTP surface.
/// </summary>
public sealed class MultiModelSessionHttpAcceptanceTests
{
    [Fact]
    public async Task TwoSessions_OnDifferentModels_EachGetsCorrectAnswer_LoadedExactlyOnce()
    {
        string? sidekickPath = FindModel("SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
        string? reasonerPath = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(sidekickPath is not null && reasonerPath is not null,
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf and Qwen3-0.6B-Q8_0.gguf are required for the " +
            "Phase 7 multi-model session HTTP acceptance test.");

        using var server = CreateMultiModelSessionServer(sidekickPath!, reasonerPath!);
        using var client = server.CreateClient();

        var sidekickSession = await CreateSessionAsync(client, "sidekick");
        var reasonerSession = await CreateSessionAsync(client, "reasoner");

        // Distinct, low-perplexity greedy prompts per session — a wrong answer reveals
        // cross-session interference (wrong model, wrong session state) rather than model
        // uncertainty, same technique MultiModelHttpAcceptanceTests already uses.
        string sidekickText = await RunTurnAsync(client, sidekickSession, "The capital of France is");
        string reasonerText = await RunTurnAsync(client, reasonerSession, "The opposite of hot is");

        Assert.Contains("Paris", sidekickText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("cold", reasonerText, StringComparison.OrdinalIgnoreCase);

        // Single-flight held, residency observable — same invariant MultiModelHttpAcceptanceTests
        // checks for the stateless endpoints, now proven for session-originated loads too.
        var manager = server.Services.GetRequiredService<IModelRuntimeManager>();
        var stats = manager.GetStats();
        Assert.Equal(2, stats.ModelLoads);
        Assert.Equal(2, stats.ResidentModels);

        // Deleting one session must not disturb the other — the Phase 4 gap fix means each
        // session's model-residency claim is independent, held via its own bound
        // ModelRuntimeHandle rather than any shared/global relay.
        Assert.Equal(HttpStatusCode.NoContent, (await client.DeleteAsync($"/v1/sessions/{sidekickSession}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync($"/v1/sessions/{sidekickSession}")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync($"/v1/sessions/{reasonerSession}")).StatusCode);

        Assert.Equal(HttpStatusCode.NoContent, (await client.DeleteAsync($"/v1/sessions/{reasonerSession}")).StatusCode);
    }

    private static async Task<Guid> CreateSessionAsync(HttpClient client, string model)
    {
        var resp = await client.PostAsJsonAsync("/v1/sessions", new { model });
        Assert.Equal(HttpStatusCode.Created, resp.StatusCode);
        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        return Guid.Parse(doc.RootElement.GetProperty("id").GetString()!);
    }

    private static async Task<string> RunTurnAsync(HttpClient client, Guid sessionId, string prompt, int maxTokens = 16)
    {
        var resp = await client.PostAsJsonAsync($"/v1/sessions/{sessionId}/turns", new
        {
            append_prompt = prompt,
            expected_revision = 0,
            max_tokens = maxTokens,
            temperature = 0f,
        });
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        return doc.RootElement.GetProperty("text").GetString() ?? "";
    }

    private static WebApplicationFactory<Program> CreateMultiModelSessionServer(string sidekickPath, string reasonerPath) =>
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
                options.EnableSessions = true; // CPU-dense lane both models already run on
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
