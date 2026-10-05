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
    }

    [Fact]
    public void InferenceParams_ToSamplingParams_MapsCorrectly()
    {
        var p = new InferenceParams
        {
            MaxTokens = 1024,
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
        var modelPath = Path.Combine("models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
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
        var modelPath = Path.Combine("models", "_models", "all-MiniLM-L6-v2-Q8_0.gguf");
        if (!File.Exists(modelPath)) return;

        var model = Model.Load(modelPath);
        var ctx = model.CreateContext();
        Assert.Equal(1, model.ActiveContextCount);

        model.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ctx.Reset());
        Assert.Throws<ObjectDisposedException>(() => model.CreateContext());
    }
}
