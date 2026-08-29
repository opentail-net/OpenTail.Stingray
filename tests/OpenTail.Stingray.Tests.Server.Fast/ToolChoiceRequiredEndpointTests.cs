
namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// OpenAI <c>tool_choice:"required"</c> (forced tool call).
///
/// <para>These go further than wiring assertions. <see cref="FakeInferenceEngine"/> ignores
/// <c>sp.Constraint</c> — it never calls <c>Accept</c> — so the constraint that reached the sampler
/// arrives at <see cref="FakeInferenceEngine.LastSamplingParams"/> still armed, and the tests drive
/// its <c>Filter</c> directly. That asserts the property the feature actually promises (only the
/// tool-call open marker is samplable) rather than the type of the object holding it, which matters
/// because the constraint may legitimately arrive alone or AND-composed with the argument
/// grammar.</para>
/// </summary>
public sealed class ToolChoiceRequiredEndpointTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private const int ToolCallMarkerToken = 1;

    /// <summary>
    /// Minimal tokenizer exposing <c>&lt;tool_call&gt;</c> as a special token — the Qwen family's
    /// open marker, which is what makes forcing possible at all.
    /// </summary>
    private sealed class MarkerTokenizer : ITokenizer
    {
        public int VocabSize => 8;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;
        public ImmutableArray<int> EogTokenIds => [0];
        public IReadOnlyDictionary<string, int> SpecialTokens { get; } =
            new Dictionary<string, int>(StringComparer.Ordinal) { ["<tool_call>"] = ToolCallMarkerToken };

        public byte[] DecodeBytes(int token) => [];
        public IReadOnlyList<int> Encode(string text) => [];
        public string Decode(IEnumerable<int> tokens) => "";
    }

    /// <summary>
    /// Same shape, but with NO tool-call marker in the vocabulary — stands in for a model whose open
    /// marker isn't a single token, which is the case the server must refuse rather than serve
    /// unforced.
    /// </summary>
    private sealed class NoMarkerTokenizer : ITokenizer
    {
        public int VocabSize => 8;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;
        public ImmutableArray<int> EogTokenIds => [0];
        public IReadOnlyDictionary<string, int> SpecialTokens { get; } =
            new Dictionary<string, int>(StringComparer.Ordinal);

        public byte[] DecodeBytes(int token) => [];
        public IReadOnlyList<int> Encode(string text) => [];
        public string Decode(IEnumerable<int> tokens) => "";
    }

    private HttpClient CreateClient(
        FakeInferenceEngine fake,
        ITokenizer? tokenizer = null,
        bool toolGrammar = false)
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                s.Configure<OpenTailStingrayServerOptions>(o =>
                {
                    o.Architecture = "qwen2";
                    o.ToolGrammar = toolGrammar;
                });
                if (tokenizer is not null)
                {
                    // The DI-constructed renderer never receives a vocabulary in these tests (no GGUF
                    // is loaded), so pin one that has been Configure()d with a grammar vocabulary.
                    var renderer = new ChatTemplateRenderer("qwen2");
                    renderer.Configure("qwen2", null, null, new GrammarVocabulary(tokenizer));
                    s.AddSingleton(renderer);
                }
                s.AddSingleton<IInferenceEngine>(fake);
            }));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    private static object BuildRequest(object toolChoice, bool withParameters = false) => new
    {
        model = "test-model",
        messages = new[] { new { role = "user", content = "Weather?" } },
        max_tokens = 16,
        stream = false,
        tool_choice = toolChoice,
        tools = new[] { new
        {
            type = "function",
            function = new
            {
                name = "get_weather",
                description = "Get weather",
                parameters = withParameters
                    ? (object)new { type = "object", properties = new { city = new { type = "string" } } }
                    : new { type = "object" },
            }
        } }
    };

    /// <summary>
    /// Drives the constraint the endpoint attached to the sampler and asserts it permits the open
    /// marker and nothing else — the actual guarantee behind "required".
    /// </summary>
    private static void AssertForcesOnlyTheMarker(ITokenConstraint? constraint)
    {
        Assert.NotNull(constraint);
        Assert.True(constraint!.IsConstraining,
            "the forced constraint must still be armed: nothing has consumed the open marker yet.");

        Span<float> logits = stackalloc float[8];
        logits.Fill(1f);
        var filtered = constraint.Filter(logits);

        for (int i = 0; i < filtered.Length; i++)
        {
            if (i == ToolCallMarkerToken)
                Assert.True(float.IsFinite(filtered[i]), "the open marker must remain samplable.");
            else
                Assert.True(float.IsNegativeInfinity(filtered[i]),
                    $"token {i} must be masked so the turn cannot begin as prose.");
        }

        // Once the marker is emitted the constraint must step aside, or the rest of the call body
        // would be masked to a single token and no arguments could ever be generated.
        constraint.Accept(ToolCallMarkerToken);
        Assert.False(constraint.IsConstraining);
    }

    [Fact]
    public async Task Required_ForcesTheOpenMarker_WhenNoArgumentGrammarIsActive()
    {
        var fake = new FakeInferenceEngine("test-model", [(GenerateChunkKind.Text, "hello")]);
        var client = CreateClient(fake, new MarkerTokenizer());

        var response = await client.PostAsJsonAsync("/v1/chat/completions", BuildRequest("required"));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        // ToolGrammar is off here, so this is the forced constraint standing alone: forcing must not
        // depend on the server-wide argument-grammar switch, or a default-configured server would
        // reject a standard OpenAI request.
        Assert.NotNull(fake.LastSamplingParams);
        Assert.IsType<ForcedToolCallConstraint>(fake.LastSamplingParams!.Constraint);
        AssertForcesOnlyTheMarker(fake.LastSamplingParams.Constraint);
    }

    [Fact]
    public async Task Required_ComposesWithTheArgumentGrammar_AndStillForcesTheMarkerFirst()
    {
        var fake = new FakeInferenceEngine("test-model", [(GenerateChunkKind.Text, "hello")]);
        var client = CreateClient(fake, new MarkerTokenizer(), toolGrammar: true);

        var response = await client.PostAsJsonAsync(
            "/v1/chat/completions", BuildRequest("required", withParameters: true));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        // Both constraints are live, so composition is what reaches the sampler...
        Assert.IsType<AndTokenConstraint>(fake.LastSamplingParams!.Constraint);
        // ...and composing must not dilute the forcing: the argument grammar is inert until the
        // marker appears, so the intersection is still exactly the marker.
        AssertForcesOnlyTheMarker(fake.LastSamplingParams.Constraint);
    }

    [Fact]
    public async Task Required_IsRejected_WhenTheModelsOpenMarkerIsNotASingleToken()
    {
        var fake = new FakeInferenceEngine("test-model", [(GenerateChunkKind.Text, "hello")]);
        var client = CreateClient(fake, new NoMarkerTokenizer());

        var response = await client.PostAsJsonAsync("/v1/chat/completions", BuildRequest("required"));

        // Refusing is the point: generating unforced would return prose to a client that asked for a
        // guaranteed call, with no way for it to tell the two apart.
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();
        Assert.Contains("required", body, StringComparison.Ordinal);
        Assert.Contains("auto", body, StringComparison.Ordinal);
        Assert.Null(fake.LastSamplingParams);
    }

    [Fact]
    public async Task Required_IsRejected_WhenNoVocabularyIsAvailable()
    {
        // No pinned renderer => no grammar vocabulary => nothing to resolve a marker against.
        var fake = new FakeInferenceEngine("test-model", [(GenerateChunkKind.Text, "hello")]);
        var client = CreateClient(fake);

        var response = await client.PostAsJsonAsync("/v1/chat/completions", BuildRequest("required"));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Null(fake.LastSamplingParams);
    }

    [Fact]
    public async Task Required_WithoutTools_IsRejected()
    {
        var fake = new FakeInferenceEngine("test-model", [(GenerateChunkKind.Text, "hello")]);
        var client = CreateClient(fake, new MarkerTokenizer());

        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Weather?" } },
            max_tokens = 16,
            tool_choice = "required",
        });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("tools", await response.Content.ReadAsStringAsync(), StringComparison.Ordinal);
        Assert.Null(fake.LastSamplingParams);
    }

    /// <summary>
    /// Contradictory request: whole-turn JSON and a forced tool call both claim the first token, so
    /// their masks intersect to nothing. The sampler treats a fully masked vocabulary as "sample
    /// freely" rather than failing, so without this refusal the response would satisfy neither
    /// constraint and look like a model quality problem.
    /// </summary>
    [Fact]
    public async Task Required_CombinedWithSchemaConstrainedResponseFormat_IsRejected()
    {
        var fake = new FakeInferenceEngine("test-model", [(GenerateChunkKind.Text, "hello")]);
        var client = CreateClient(fake, new MarkerTokenizer());

        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Weather?" } },
            max_tokens = 16,
            tool_choice = "required",
            response_format = new
            {
                type = "json_schema",
                json_schema = new { schema = new { type = "object", properties = new { a = new { type = "string" } } } },
            },
            tools = new[] { new
            {
                type = "function",
                function = new { name = "get_weather", description = "w", parameters = new { type = "object" } }
            } }
        });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("response_format", await response.Content.ReadAsStringAsync(), StringComparison.Ordinal);
        Assert.Null(fake.LastSamplingParams);
    }

    /// <summary>
    /// The neighbouring tool_choice values must be unaffected: "auto" leaves generation free, and
    /// forcing must not leak into requests that didn't ask for it.
    /// </summary>
    [Theory]
    [InlineData("auto")]
    [InlineData("none")]
    public async Task OtherToolChoiceValues_DoNotForceACall(string toolChoice)
    {
        var fake = new FakeInferenceEngine("test-model", [(GenerateChunkKind.Text, "hello")]);
        var client = CreateClient(fake, new MarkerTokenizer());

        var response = await client.PostAsJsonAsync("/v1/chat/completions", BuildRequest(toolChoice));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.IsNotType<ForcedToolCallConstraint>(fake.LastSamplingParams!.Constraint);
    }
}
