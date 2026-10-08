
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Liquid LFM2 (lfm2: gated short-conv layers + GQA attention with QK-norm, NeoX RoPE) against llama-server
/// (vendored tools/llama.cpp, LiquidAI/LFM2-1.2B-GGUF Q8_0, raw completion with BOS, temperature 0, ignore_eos,
/// n_probs 2; captured 2026-09-27). LFM2's top-2 are often close, so the continuation is teacher-forced and our
/// argmax must equal llama.cpp's token wherever its top-1 margin exceeds 1.5 nats (TeacherForcedParity). Admission
/// evidence is second-half perplexity (ModelCompatibility.cs). Also pins that the short-conv state resets.
/// </summary>
public sealed class Lfm2ParityTests : HeavyTestBase
{
    private const string ModelFile = "LFM2-1.2B-Q8_0.gguf";

    private const string EinsteinPrompt =
        "In 1905, Albert Einstein published four papers that changed physics. The first explained the photoelectric effect, and";

    [Fact]
    public void Lfm2_ShortPrompt_ConfidentTokensMatchLlamaServer()
    {
        // ' Paris. However, the largest city in terms of population is actually Marseille, but'
        TeacherForcedParity.Assert(ModelFile, "lfm2", TeacherForcedParity.Tokenize(ModelFile, "The capital of France is"),
            [5242, 523, 2876, 521, 779, 6584, 3771, 797, 4450, 803, 3443, 856, 4703, 42161, 521, 1203],
            [3.349, 0.021, 0.306, 8.597, 0.672, 0.092, 2.53, 0.117, 0.163, 11.463, 3.932, 0.89, 0.327, 0.701, 0.692, 0.846],
            minChecked: 4);
    }

    [Fact]
    public void Lfm2_LongerPrompt_ConfidentTokensMatchLlamaServer()
    {
        // ' the others introduced the theory of special relativity. Which of these directly challenged classical physics'
        TeacherForcedParity.Assert(ModelFile, "lfm2", TeacherForcedParity.Tokenize(ModelFile, EinsteinPrompt),
            [779, 3393, 7673, 779, 5654, 803, 3429, 55458, 523, 20463, 803, 1518, 5828, 27731, 13813, 14625],
            [3.892, 0.585, 1.859, 0.559, 1.321, 9.394, 0.952, 5.808, 1.792, 1.772, 2.219, 0.73, 0.157, 2.575, 2.003, 0.267],
            minChecked: 8);
    }

    [Fact]
    public void Lfm2_ResetCache_ClearsShortConvState()
    {
        var path = TeacherForcedParity.FindModel(ModelFile);
        Assert.SkipWhen(path is null, $"{ModelFile} is required.");
        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.NotNull(hp.IsShortConvLayer);
        var tokens = TeacherForcedParity.Tokenize(ModelFile, EinsteinPrompt);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024);
        var first = fwd.Prefill(tokens).ToArray();
        fwd.ResetCache();
        var second = fwd.Prefill(tokens).ToArray();
        Assert.Equal(first, second);
    }
}
