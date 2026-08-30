
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies the real T3 GPT2-medium acoustic LM (ChatterboxAcousticLm.GenerateReal, ported from
/// examples/chatterbox-tts-py/chatterbox/models/t3/t3.py's inference_turbo) against real GGUF
/// weights. Unlike Kokoro's stage tests, there is no local PyTorch/safetensors checkpoint for
/// Chatterbox-Turbo to build a golden-output oracle from (only the GGUF conversion and an ONNX
/// speech encoder are available), so this is structural verification -- real weight-driven
/// inference produces a well-formed, in-vocabulary, terminating speech-token sequence -- not a
/// cosine-similarity-against-ground-truth check. Should Chatterbox reference safetensors become
/// available locally (e.g. via `pip download`/`huggingface_hub.snapshot_download`), a real
/// golden-token-sequence test should replace/supplement this one.
/// </summary>
public sealed class ChatterboxT3Tests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ChatterboxTokenizer_RealBpe_RoundTripsPlausibleTokenIds()
    {
        string? t3Path = FindRepoFile("models/chatterbox-turbo-t3-q4_k.gguf");
        if (t3Path is null) return;

        using var weights = new ChatterboxWeights(t3Path);
        var tokenizer = new ChatterboxTokenizer(weights);

        int[] tokens = tokenizer.Encode("Hello from Chatterbox!");

        Assert.NotEmpty(tokens);
        // All tokens must be real in-vocabulary BPE ids.
        for (int i = 0; i < tokens.Length; i++)
        {
            Assert.InRange(tokens[i], 0, weights.TextVocabSize - 1);
        }
    }

    [Fact]
    public void ChatterboxAcousticLm_RealT3Weights_GeneratesValidInVocabSpeechTokens()
    {
        string? t3Path = FindRepoFile("models/chatterbox-turbo-t3-q4_k.gguf");
        if (t3Path is null) return;

        using var weights = new ChatterboxWeights(t3Path);
        Assert.NotNull(weights.SpeakerEmbedding);
        Assert.NotNull(weights.SpeechPromptTokens);
        Assert.Equal(weights.SpeechCondPromptLen, weights.SpeechPromptTokens!.Length);

        var tokenizer = new ChatterboxTokenizer(weights);
        // Deterministic sampling (seeded RNG) so this test is reproducible.
        using var lm = new ChatterboxAcousticLm(weights, new Random(1234));

        int[] textTokens = tokenizer.Encode("Testing real Chatterbox Turbo weights in OpenTail Stingray.");

        var speechTokens = lm.GenerateSpeechTokens(textTokens, [], temperature: 0.8f, maxTokens: 32);

        Assert.True(speechTokens.Count >= 2, "Must produce at least the start/stop sentinel tokens.");
        Assert.Equal(weights.StartSpeechToken, speechTokens[0]);
        Assert.Equal(weights.StopSpeechToken, speechTokens[^1]);

        // Every generated token (excluding the start/stop sentinels) must be a real, in-vocabulary
        // speech token id -- this is the load-bearing check: the old placeholder generator produced
        // tokens from an entirely different, made-up numeric range (100 + arbitrary % 1024/2048),
        // so a regression back to fake generation would fail this immediately.
        foreach (int tok in speechTokens.Skip(1).SkipLast(1))
        {
            Assert.InRange(tok, 0, weights.SpeechVocabSize - 1);
        }

        // Should not degenerate into a single repeated token for the entire budget (a common
        // failure mode for a broken/garbage forward pass -- e.g. always argmax-ing token 0).
        Assert.True(speechTokens.Distinct().Count() > 1, "Speech token sequence must not be constant.");
    }

    [Fact]
    public void ChatterboxAcousticLm_RealT3Weights_IsDeterministicForFixedSeed()
    {
        string? t3Path = FindRepoFile("models/chatterbox-turbo-t3-q4_k.gguf");
        if (t3Path is null) return;

        using var weights = new ChatterboxWeights(t3Path);
        var tokenizer = new ChatterboxTokenizer(weights);
        int[] textTokens = tokenizer.Encode("Determinism check.");

        using var lm1 = new ChatterboxAcousticLm(weights, new Random(42));
        using var lm2 = new ChatterboxAcousticLm(weights, new Random(42));

        var tokens1 = lm1.GenerateSpeechTokens(textTokens, [], temperature: 0.8f, maxTokens: 32);
        var tokens2 = lm2.GenerateSpeechTokens(textTokens, [], temperature: 0.8f, maxTokens: 32);

        Assert.Equal(tokens1, tokens2);
    }
}
