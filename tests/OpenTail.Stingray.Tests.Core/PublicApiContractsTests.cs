#nullable enable

using OpenTail.Stingray;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Executors;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public class PublicApiContractsTests
{
    [Fact]
    public void ModelParams_HasExpectedDefaults()
    {
        var p = new ModelParams("test/model.gguf");

        Assert.Equal("test/model.gguf", p.ModelPath);
        Assert.Equal("auto", p.Backend);
        Assert.Equal(-1, p.GpuLayerCount);
        Assert.Equal(0, p.MainGpu);
        Assert.Null(p.TensorSplit);
        Assert.True(p.UseMemoryMap);
        Assert.False(p.UseMemoryLock);
        Assert.False(p.AllowUnverifiedArch);
        Assert.Null(p.DraftModelPath);
        Assert.False(p.DraftLookup);
        Assert.Null(p.DSparkModelPath);
    }

    [Fact]
    public void ContextParams_HasExpectedDefaults()
    {
        var p = new ContextParams();

        Assert.Equal(0u, p.ContextSize);
        Assert.Equal(512u, p.BatchSize);
        Assert.Equal(0, p.ThreadCount);
        Assert.True(p.FlashAttention);
        Assert.Null(p.TurboQuantMode);
        Assert.Null(p.TurboQuantHeadDim);
        Assert.Null(p.RopeFrequencyBase);
        Assert.Null(p.RopeFrequencyScale);
        Assert.False(p.Embeddings);
    }

    [Fact]
    public void InferenceParams_HasExpectedDefaults()
    {
        var p = new InferenceParams();

        Assert.Equal(512, p.MaxTokens);
        Assert.Equal(0.7f, p.Temperature);
        Assert.Equal(40, p.TopK);
        Assert.Equal(0.9f, p.TopP);
        Assert.Equal(0.0f, p.MinP);
        Assert.Equal(1.0f, p.RepetitionPenalty);
        Assert.Equal(64, p.RepeatLastTokensCount);
        Assert.Equal(0.0f, p.PresencePenalty);
        Assert.Equal(0.0f, p.FrequencyPenalty);
        Assert.Null(p.LogitBias);
        Assert.Null(p.StopSequences);
        Assert.Null(p.StopTokens);
        Assert.Null(p.AdditionalStopTokens);
        Assert.Null(p.AllowedChoices);
        Assert.Null(p.EnableThinking);
        Assert.Equal(0, p.ThinkingBudget);
        Assert.Equal(SpecType.Auto, p.SpecType);
        Assert.Equal(0, p.SpecDraftNMax);
        Assert.Null(p.Constraint);
        Assert.Null(p.Seed);
        Assert.Null(p.CanonicalHistoryPrefix);
    }

    [Fact]
    public void InferenceParams_ToSamplingParams_MapsCorrectly()
    {
        var p = new InferenceParams
        {
            MaxTokens = 1024,
            Seed = 42,
            Temperature = 0.0f,
            TopK = 50,
            TopP = 0.95f,
            MinP = 0.08f,
            RepetitionPenalty = 1.15f,
            RepeatLastTokensCount = 128,
            PresencePenalty = 0.2f,
            FrequencyPenalty = 0.3f,
            StopTokens = [1, 2, 3],
            AdditionalStopTokens = [4, 5],
            AllowedChoices = ["YES", "NO"],
            EnableThinking = false,
            ThinkingBudget = 256,
            SpecType = SpecType.Mtp,
            SpecDraftNMax = 2
        };

        var sp = p.ToSamplingParams();

        Assert.Equal(1024, sp.MaxNewTokens);
        Assert.Equal(42, sp.Seed);
        Assert.Equal(0.0f, sp.Temperature);
        Assert.Equal(50, sp.TopK);
        Assert.Equal(0.95f, sp.TopP);
        Assert.Equal(0.08f, sp.MinP);
        Assert.Equal(1.15f, sp.RepetitionPenalty);
        Assert.Equal(128, sp.RepeatLastN);
        Assert.Equal(0.2f, sp.PresencePenalty);
        Assert.Equal(0.3f, sp.FrequencyPenalty);
        Assert.Equal(new[] { 1, 2, 3 }, sp.StopTokenIds);
        Assert.Equal(new[] { 4, 5 }, sp.AdditionalStopTokenIds);
        Assert.Equal(new[] { "YES", "NO" }, sp.AllowedChoices);
        Assert.True(sp.ThinkingDisabled);
        Assert.Equal(256, sp.MaxThinkingTokens);
        Assert.Equal(SpecType.Mtp, sp.SpecType);
        Assert.Equal(2, sp.SpecDraftNMax);
    }

    [Fact]
    public void Model_Load_Throws_WhenFileNotFound()
    {
        Assert.Throws<FileNotFoundException>(() => Model.Load("non_existent_model_file.gguf"));
    }

    [Fact]
    public void Model_Load_Throws_WhenParametersNull()
    {
        Assert.Throws<ArgumentNullException>(() => Model.Load((ModelParams)null!));
    }

    [Fact]
    public void Model_Load_And_MultipleContexts_Lifecycle()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        using var model = Model.Load(modelPath);
        Assert.False(string.IsNullOrEmpty(model.Architecture));
        Assert.True(model.ContextLength > 0);
        Assert.True(model.EmbeddingLength > 0);
        Assert.Equal(0, model.ActiveContextCount);

        var ctx1 = model.CreateContext(new ContextParams { ContextSize = 128 });
        Assert.Equal(1, model.ActiveContextCount);
        Assert.Equal(128, ctx1.ContextSize);
        Assert.NotNull(ctx1.Tokenizer);

        var ctx2 = model.CreateContext(new ContextParams { ContextSize = 256 });
        Assert.Equal(2, model.ActiveContextCount);
        Assert.Equal(256, ctx2.ContextSize);

        ctx1.Dispose();
        Assert.Equal(1, model.ActiveContextCount);

        ctx2.Dispose();
        Assert.Equal(0, model.ActiveContextCount);
    }

    [Fact]
    public void Model_Dispose_DisposesActiveChildContexts()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        var model = Model.Load(modelPath);
        var ctx = model.CreateContext();
        Assert.Equal(1, model.ActiveContextCount);

        model.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ctx.Reset());
        Assert.Throws<ObjectDisposedException>(() => model.CreateContext());
    }

    [Fact]
    public async Task InteractiveExecutor_DualStreaming_And_PrefixHistory()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        using var model = Model.Load(modelPath);
        using var ctx = (ModelContext)model.CreateContext();
        var fakeEngine = new FakeInferenceEngine();
        ctx.SetEngine(fakeEngine);

        var executor = new InteractiveExecutor(ctx);

        // Turn 1: simple string stream filters out thinking and metadata
        var strings = new List<string>();
        await foreach (var piece in executor.InferAsync("Turn 1 prompt"))
        {
            strings.Add(piece);
        }
        Assert.Equal(["Hello", " world!"], strings);
        Assert.Single(fakeEngine.RecordedCalls);
        Assert.Equal("Turn 1 prompt", fakeEngine.RecordedCalls[0].Prompt);
        Assert.Null(fakeEngine.RecordedCalls[0].Prefix);

        // Turn 2: should carry Turn 1 prompt as canonicalHistoryPrefix
        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in executor.InferChunksAsync("Turn 2 prompt"))
        {
            chunks.Add(chunk);
        }
        Assert.Equal(2, fakeEngine.RecordedCalls.Count);
        Assert.Equal("Turn 2 prompt", fakeEngine.RecordedCalls[1].Prompt);
        Assert.Equal("Turn 1 promptHello world!", fakeEngine.RecordedCalls[1].Prefix);

        // Rich typed stream includes thinking and usage
        Assert.Contains(chunks, c => c.Kind == GenerateChunkKind.Thinking && c.Text.Contains("Thinking"));
        Assert.Contains(chunks, c => c.Kind == GenerateChunkKind.Usage);
        Assert.Contains(chunks, c => c.Kind == GenerateChunkKind.Stop);
    }

    [Fact]
    public async Task StatelessExecutor_ResetsContext_And_DoesNotPassPrefix()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        using var model = Model.Load(modelPath);
        using var ctx = (ModelContext)model.CreateContext();
        var fakeEngine = new FakeInferenceEngine();
        ctx.SetEngine(fakeEngine);

        var executor = new StatelessExecutor(ctx);

        await foreach (var _ in executor.InferAsync("Stateless 1")) { }
        await foreach (var _ in executor.InferAsync("Stateless 2")) { }

        Assert.Equal(2, fakeEngine.RecordedCalls.Count);
        Assert.Null(fakeEngine.RecordedCalls[0].Prefix);
        Assert.Null(fakeEngine.RecordedCalls[1].Prefix);
    }

    [Fact]
    public async Task BatchedExecutor_DelegatesToEngine()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        using var model = Model.Load(modelPath);
        using var ctx = (ModelContext)model.CreateContext();
        var fakeEngine = new FakeInferenceEngine();
        ctx.SetEngine(fakeEngine);

        var executor = new BatchedExecutor(ctx);
        var strings = new List<string>();
        await foreach (var s in executor.InferAsync("Batched request"))
        {
            strings.Add(s);
        }

        Assert.Equal(["Hello", " world!"], strings);
        Assert.Single(fakeEngine.RecordedCalls);
        Assert.Equal("Batched request", fakeEngine.RecordedCalls[0].Prompt);
    }

    [Fact]
    public void ChatHistory_CollectionOperations_WorkCorrectly()
    {
        var history = new ChatHistory();
        Assert.Empty(history);

        history.AddSystemMessage("You are a helpful assistant.");
        history.AddUserMessage("Hello!");
        history.AddAssistantMessage("Hi there!");
        history.AddToolMessage("Tool result");

        Assert.Equal(4, history.Count);
        Assert.Equal(AuthorRole.System, history[0].Role);
        Assert.Equal("You are a helpful assistant.", history[0].Content);
        Assert.Equal(AuthorRole.User, history[1].Role);
        Assert.Equal("Hello!", history[1].Content);
        Assert.Equal(AuthorRole.Assistant, history[2].Role);
        Assert.Equal("Hi there!", history[2].Content);
        Assert.Equal(AuthorRole.Tool, history[3].Role);
        Assert.Equal("Tool result", history[3].Content);

        // Verification of IEnumerable enumeration
        var roles = history.Select(m => m.Role).ToList();
        Assert.Equal([AuthorRole.System, AuthorRole.User, AuthorRole.Assistant, AuthorRole.Tool], roles);

        // Verification of Clear
        history.Clear();
        Assert.Empty(history);
    }

    [Fact]
    public void ChatMessage_StaticFactories_WorkCorrectly()
    {
        var sys = ChatMessage.System("sys");
        var usr = ChatMessage.User("usr");
        var ast = ChatMessage.Assistant("ast");
        var tool = ChatMessage.Tool("tool");

        Assert.Equal(AuthorRole.System, sys.Role);
        Assert.Equal("sys", sys.Content);
        Assert.Equal(AuthorRole.User, usr.Role);
        Assert.Equal("usr", usr.Content);
        Assert.Equal(AuthorRole.Assistant, ast.Role);
        Assert.Equal("ast", ast.Content);
        Assert.Equal(AuthorRole.Tool, tool.Role);
        Assert.Equal("tool", tool.Content);
    }

    [Fact]
    public async Task ChatSession_DualStreaming_And_MultiTurnHistoryTracking()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        using var model = Model.Load(modelPath);
        using var ctx = (ModelContext)model.CreateContext();
        var fakeEngine = new FakeInferenceEngine();
        ctx.SetEngine(fakeEngine);

        var executor = new InteractiveExecutor(ctx);
        var session = new ChatSession(executor);
        session.AddSystemMessage("You are a test assistant.");

        // Turn 1: Simple string streaming via ChatAsync
        var turn1Tokens = new List<string>();
        await foreach (var token in session.ChatAsync("What is 2+2?"))
        {
            turn1Tokens.Add(token);
        }

        // Filters thinking tokens, returns only user-facing text
        Assert.Equal(["Hello", " world!"], turn1Tokens);
        Assert.Equal(3, session.History.Count);
        Assert.Equal(AuthorRole.System, session.History[0].Role);
        Assert.Equal(AuthorRole.User, session.History[1].Role);
        Assert.Equal("What is 2+2?", session.History[1].Content);
        Assert.Equal(AuthorRole.Assistant, session.History[2].Role);
        Assert.Equal("Hello world!", session.History[2].Content);

        // Turn 2: Rich typed chunk streaming via ChatChunksAsync
        var turn2Chunks = new List<GenerateChunk>();
        await foreach (var chunk in session.ChatChunksAsync("And what is 3+3?"))
        {
            turn2Chunks.Add(chunk);
        }

        // Rich stream preserves thinking, usage, and stop chunks
        Assert.Contains(turn2Chunks, c => c.Kind == GenerateChunkKind.Thinking);
        Assert.Contains(turn2Chunks, c => c.Kind == GenerateChunkKind.Usage);
        Assert.Contains(turn2Chunks, c => c.Kind == GenerateChunkKind.Stop);

        // History now has 5 messages: System, User1, Assistant1, User2, Assistant2
        Assert.Equal(5, session.History.Count);
        Assert.Equal(AuthorRole.User, session.History[3].Role);
        Assert.Equal("And what is 3+3?", session.History[3].Content);
        Assert.Equal(AuthorRole.Assistant, session.History[4].Role);
        Assert.Equal("Hello world!", session.History[4].Content);
    }

    [Fact]
    public void ChatSession_DefaultChatMLFormat_FormatsCorrectly()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        using var model = Model.Load(modelPath);
        using var ctx = (ModelContext)model.CreateContext();
        var executor = new StatelessExecutor(ctx);

        var session = new ChatSession(executor);
        session.AddSystemMessage("Be concise.");
        session.AddUserMessage("Hello");

        string formatted = session.FormatPrompt(session.History);
        Assert.Contains("<|im_start|>system\nBe concise.<|im_end|>", formatted);
        Assert.Contains("<|im_start|>user\nHello<|im_end|>", formatted);
        Assert.EndsWith("<|im_start|>assistant\n", formatted);
    }

    [Fact]
    public void ChatSession_CustomPromptFormatter_OverridesDefault()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        using var model = Model.Load(modelPath);
        using var ctx = (ModelContext)model.CreateContext();
        var executor = new StatelessExecutor(ctx);

        var session = new ChatSession(executor)
        {
            PromptFormatter = history => string.Join(" --- ", history.Select(m => $"{m.Role}:{m.Content}"))
        };
        session.AddUserMessage("Question");

        string formatted = session.FormatPrompt(session.History);
        Assert.Equal("User:Question", formatted);
    }

    [Fact]
    public async Task LlamaSharpPatternParity_EndToEndApplicationFlow()
    {
        // Demonstrates that a developer coming from LLamaSharp writes code with identical structure:
        // 1. ModelParams & Model.Load
        // 2. ContextParams & model.CreateContext
        // 3. InteractiveExecutor
        // 4. ChatSession
        // 5. Dual-stream chat: string streaming and typed chunk streaming

        var modelPath = Path.Combine(RepoRoot, "models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        var modelParams = new ModelParams(modelPath)
        {
            GpuLayerCount = 0
        };

        using var model = Model.Load(modelParams);
        var contextParams = new ContextParams
        {
            ContextSize = 2048,
            BatchSize = 512
        };

        using var context = (ModelContext)model.CreateContext(contextParams);
        var fakeEngine = new FakeInferenceEngine();
        context.SetEngine(fakeEngine);

        var executor = new InteractiveExecutor(context);
        var session = new ChatSession(executor);
        session.AddSystemMessage("You are an expert assistant.");

        // First turn: simple text streaming
        var responseTokens = new List<string>();
        await foreach (var piece in session.ChatAsync("What is prefix caching?"))
        {
            responseTokens.Add(piece);
        }

        Assert.Equal("Hello world!", string.Concat(responseTokens));
        Assert.Equal(3, session.History.Count);

        // Second turn: rich chunk streaming preserving thinking and usage
        var chunks = new List<GenerateChunk>();
        var inferenceParams = new InferenceParams
        {
            MaxTokens = 100,
            Temperature = 0.7f,
            TopP = 0.9f
        };

        await foreach (var chunk in session.ChatChunksAsync("Can you elaborate?", inferenceParams))
        {
            chunks.Add(chunk);
        }

        Assert.Equal(5, session.History.Count);
        Assert.Contains(chunks, c => c.Kind == GenerateChunkKind.Thinking);
        Assert.Contains(chunks, c => c.Kind == GenerateChunkKind.Text);
        Assert.Contains(chunks, c => c.Kind == GenerateChunkKind.Usage);
    }

    private static readonly string RepoRoot = FindRepoRoot();

    private static string FindRepoRoot()
    {
        string? root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "CLAUDE.md")))
            root = Path.GetDirectoryName(root);
        return root ?? Directory.GetCurrentDirectory();
    }

    // Real-weights check (silently no-ops without the checkpoint; a genuine run takes seconds).
    // Expected text is the CPU greedy baseline in docs/2-coverage/2026-10-05-forward-pass-selection-matrix.md.
    // Real-weights check (silently no-ops without the checkpoint; a genuine run takes seconds).
    // Expected text is the CPU greedy baseline in docs/2-coverage/2026-10-05-forward-pass-selection-matrix.md.
    [Fact]
    public async Task ChatSession_RealSmolLM2_GreedyMatchesCliBaseline_AndStatefulTurns()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (!File.Exists(modelPath)) return;

        var modelParams = new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 };
        using var model = Model.Load(modelParams);
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512 });
        var session = new ChatSession(new InteractiveExecutor(ctx));
        var inf = new InferenceParams { MaxTokens = 24, Temperature = 0f };

        var sb = new System.Text.StringBuilder();
        await foreach (var piece in session.ChatAsync("Write a short story: Once upon a time", inf))
        {
            sb.Append(piece);
        }

        Assert.StartsWith("Once upon a time, there lived a young girl named Lily.", sb.ToString());
        Assert.Equal(2, session.History.Count);
        Assert.Equal(sb.ToString(), session.History[1].Content);

        var turn2 = new System.Text.StringBuilder();
        await foreach (var piece in session.ChatAsync("Continue.", new InferenceParams { MaxTokens = 8, Temperature = 0f }))
        {
            turn2.Append(piece);
        }
        Assert.False(string.IsNullOrWhiteSpace(turn2.ToString()));
        Assert.Equal(4, session.History.Count);

        // Verification of review item 4: prefix tokens were actually reused on turn 2
        Assert.True(ctx.Engine.PrefillTokensReused > 0,
            $"Expected PrefillTokensReused > 0 on turn 2, but was {ctx.Engine.PrefillTokensReused}");
    }

    [Fact]
    public void ModelContext_Reset_ClearsEngineState()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (!File.Exists(modelPath)) return;

        var modelParams = new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 };
        using var model = Model.Load(modelParams);
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512 });

        // Force engine initialization
        _ = ctx.Engine;

        // Reset should execute cleanly without throwing
        ctx.Reset();
    }

    [Fact]
    public void BatchedExecutor_ConstructsRealContinuousBatchingEngine()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (!File.Exists(modelPath)) return;

        var modelParams = new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 };
        using var model = Model.Load(modelParams);
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512, BatchSize = 4 });

        var batchedExecutor = new BatchedExecutor(ctx);
        Assert.IsType<ContinuousBatchingEngine>(ctx.Engine);
    }

    [Fact]
    public async Task InferenceParams_Seed_ProducesDeterministicOutput()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (!File.Exists(modelPath)) return;

        var modelParams = new ModelParams(modelPath) { Backend = "cpu", GpuLayerCount = 0 };
        using var model = Model.Load(modelParams);
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512 });

        var executor = new StatelessExecutor(ctx);
        var inf1 = new InferenceParams { MaxTokens = 12, Temperature = 0.8f, Seed = 12345 };
        var inf2 = new InferenceParams { MaxTokens = 12, Temperature = 0.8f, Seed = 12345 };

        var sb1 = new System.Text.StringBuilder();
        await foreach (var piece in executor.InferAsync("The universe is", inf1))
        {
            sb1.Append(piece);
        }

        var sb2 = new System.Text.StringBuilder();
        await foreach (var piece in executor.InferAsync("The universe is", inf2))
        {
            sb2.Append(piece);
        }

        Assert.Equal(sb1.ToString(), sb2.ToString());
    }

    [Fact]
    public void ModelParams_ExplicitVulkanBackend_ConfiguresGpuForwardPass()
    {
        var modelPath = Path.Combine(RepoRoot, "models", "_models", "SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (!File.Exists(modelPath)) return;

        var modelParams = new ModelParams(modelPath) { Backend = "vulkan", GpuLayerCount = -1 };
        using var model = Model.Load(modelParams);
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 512 });

        var engine = ctx.Engine;
        Assert.NotNull(engine);
    }

    [Fact]
    public void ChatSession_JinjaRaiseException_ThrowsChatTemplateException_DoesNotFallbackToChatML()
    {
        var tokSource = new TokenizerSource
        {
            Tokens = ["<unk>", "<s>", "</s>"],
            ChatTemplate = "{{ raise_exception('Rejected by template') }}"
        };
        var tok = GgufTokenizer.FromSource(tokSource);
        var ctx = new ContextWithTokenizer(tok);
        var session = new ChatSession(new StubExecutor(ctx));

        session.AddUserMessage("Hello");
        var ex = Assert.Throws<ChatTemplateException>(() => session.FormatPrompt(session.History));
        Assert.Contains("Rejected by template", ex.Message);
    }

    private sealed class ContextWithTokenizer(ITokenizer tokenizer) : IModelContext
    {
        public IModel Model => null!;
        public ITokenizer Tokenizer => tokenizer;
        public int ContextSize => 512;
        public IInferenceEngine Engine => null!;
        public void Reset() { }
        public void Dispose() { }
    }

    private sealed class StubExecutor(IModelContext context) : IExecutor
    {
        public IModelContext Context => context;
        public IAsyncEnumerable<string> InferAsync(string prompt, IInferenceParams? inferenceParams = null, CancellationToken cancellationToken = default) => throw new NotImplementedException();
        public IAsyncEnumerable<GenerateChunk> InferChunksAsync(string prompt, IInferenceParams? inferenceParams = null, CancellationToken cancellationToken = default) => throw new NotImplementedException();
    }

    private sealed class FakeInferenceEngine : IInferenceEngine
    {
        public string ModelId => "fake-model";
        public int QueueDepth => 0;
        public int ActiveRequests => 0;
        public bool PrefixCacheEnabled => true;
        public long PrefillTokensReused => 0;

        public List<(string Prompt, string? Prefix)> RecordedCalls { get; } = [];

        public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
            string prompt,
            SamplingParams sp,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default,
            string? canonicalHistoryPrefix = null)
        {
            RecordedCalls.Add((prompt, canonicalHistoryPrefix));
            yield return new GenerateChunk(GenerateChunkKind.Usage, "", 5);
            yield return new GenerateChunk(GenerateChunkKind.Thinking, "Thinking about answer...");
            yield return new GenerateChunk(GenerateChunkKind.Text, "Hello");
            yield return new GenerateChunk(GenerateChunkKind.Text, " world!");
            yield return new GenerateChunk(GenerateChunkKind.Stop, "");
        }
    }
}
