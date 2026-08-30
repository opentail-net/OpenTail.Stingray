using OpenTail.Stingray.Audio.Xtts;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end EXACT token-match verification for <see cref="XttsGptSampler"/>'s
/// greedy path, against `scratch-llamacpp-ref/xtts_greedy_generate_golden.py`'s real
/// `model.gpt.generate(..., do_sample=False)` output. Greedy removes all RNG concerns, so this
/// is directly, exactly comparable token-for-token (not just cosine-similar) -- the strongest
/// possible check for the whole autoregressive loop + embeddings + trunk + conditioning chain.
/// Uses the real tokenizer's own output ids (this port's own BPE tokenizer is not built yet).</summary>
public sealed class XttsGptSamplerTests : HeavyTestBase
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

    private static float[] ReadCsv(string path) =>
        Array.ConvertAll(File.ReadAllText(path).Trim().Split(','), float.Parse);

    [Fact]
    public void Generate_Greedy_RealWeights_ExactlyMatchesGoldenOracle()
    {
        string? weightsPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/xtts-v2/model.safetensors not found");
        string? melPath = FindRepoFile("scratch-llamacpp-ref/xtts_greedy_golden_mel.txt");
        string? textIdsPath = FindRepoFile("scratch-llamacpp-ref/xtts_greedy_golden_text_ids.txt");
        string? tokensPath = FindRepoFile("scratch-llamacpp-ref/xtts_greedy_golden_tokens.txt");
        Assert.SkipUnless(melPath != null && textIdsPath != null && tokensPath != null,
            "golden greedy-generate files not found (re-run scratch-llamacpp-ref/xtts_greedy_generate_golden.py)");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var condWeights = new XttsConditioningWeights(loader);
        var trunkWeights = new XttsGptWeights(loader);
        var embWeights = new XttsGptEmbeddings(loader);

        float[] mel = ReadCsv(melPath!);
        int melT = mel.Length / XttsConditioningWeights.MelDim;
        float[] condLatents = XttsConditioningEncoder.Encode(condWeights, mel, melT);

        int[] textIds = Array.ConvertAll(File.ReadAllText(textIdsPath!).Trim().Split(','), int.Parse);
        float[] prefix = XttsGptGenerator.BuildPrefix(embWeights, condLatents, XttsConditioningWeights.PerceiverNumLatents, textIds, out int prefixLen);

        // Greedy: temperature=0 routes Sampler.Sample to its deterministic argmax path --
        // RNG is never consulted, so the passed Random is irrelevant (Sampler.Sample's own
        // temperature<=0 branch ignores it). Repetition penalty was also disabled in the golden
        // run (repetition_penalty=1.0) for a clean greedy trace.
        var greedyParams = new SamplingParams { Temperature = 0f, RepetitionPenalty = 1.0f };
        var generated = XttsGptSampler.Generate(trunkWeights, embWeights, prefix, prefixLen, new Random(0), greedyParams, maxTokens: 8);

        int[] golden = Array.ConvertAll(File.ReadAllText(tokensPath!).Trim().Split(','), int.Parse);

        Assert.Equal(golden.Length, generated.Count);
        for (int i = 0; i < golden.Length; i++)
            Assert.Equal(golden[i], generated[i]);
    }
}
