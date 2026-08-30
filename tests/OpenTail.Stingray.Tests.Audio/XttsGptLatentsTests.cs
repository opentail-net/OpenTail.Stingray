using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsGptLatents"/> (the real vocoder-
/// input hidden-state extraction), against `scratch-llamacpp-ref/xtts_gpt_latents_golden.py`'s
/// real `GPT.forward(..., return_latent=True)` output. See <see cref="XttsGptLatents"/>'s own
/// doc comment for why this port deliberately doesn't replicate the reference's padding/trim
/// arithmetic and instead relies on causal-attention invariance.</summary>
public sealed class XttsGptLatentsTests : HeavyTestBase
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

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    [Fact]
    public void ComputeLatents_RealWeights_MatchesGoldenOracle()
    {
        string? weightsPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/xtts-v2/model.safetensors not found");
        string? melPath = FindRepoFile("scratch-llamacpp-ref/xtts_gpt_latents_golden_mel.txt");
        string? textIdsPath = FindRepoFile("scratch-llamacpp-ref/xtts_gpt_latents_golden_text_ids.txt");
        string? codesPath = FindRepoFile("scratch-llamacpp-ref/xtts_gpt_latents_golden_codes.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/xtts_gpt_latents_golden_output.txt");
        Assert.SkipUnless(melPath != null && textIdsPath != null && codesPath != null && outPath != null,
            "golden gpt_latents files not found (re-run scratch-llamacpp-ref/xtts_gpt_latents_golden.py)");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var condWeights = new XttsConditioningWeights(loader);
        var trunkWeights = new XttsGptWeights(loader);
        var embWeights = new XttsGptEmbeddings(loader);

        float[] mel = ReadCsv(melPath!);
        int melT = mel.Length / XttsConditioningWeights.MelDim;
        float[] condLatents = XttsConditioningEncoder.Encode(condWeights, mel, melT);

        int[] textIds = Array.ConvertAll(File.ReadAllText(textIdsPath!).Trim().Split(','), int.Parse);
        float[] prefix = XttsGptGenerator.BuildPrefix(embWeights, condLatents, XttsConditioningWeights.PerceiverNumLatents, textIds, out int prefixLen);

        // Golden's own gpt_codes included a trailing stop_audio_token (1025) as its LAST entry --
        // strip it here since this port's ComputeLatents expects real generated codes only
        // (matching what XttsGptSampler.Generate returns), prepending start_audio_token itself.
        int[] rawCodes = Array.ConvertAll(File.ReadAllText(codesPath!).Trim().Split(','), int.Parse);
        int[] generatedCodes = rawCodes[..^1];

        float[] latents = XttsGptLatents.ComputeLatents(trunkWeights, embWeights, prefix, prefixLen, generatedCodes);

        // Golden is real PyTorch [1, T, dim] token-major -- transpose to channel-first to match.
        float[] goldenTokenMajor = ReadCsv(outPath!);
        int dim = XttsGptWeights.ModelDim;
        int goldenLenRaw = goldenTokenMajor.Length / dim;
        var golden = new float[goldenTokenMajor.Length];
        for (int ti = 0; ti < goldenLenRaw; ti++)
            for (int d = 0; d < dim; d++)
                golden[d * goldenLenRaw + ti] = goldenTokenMajor[ti * dim + d];

        // Compare over the overlapping prefix length only (see class doc comment on why the
        // reference's own padding/trim length need not match exactly) -- channel-first layout on
        // both sides, so slice per-channel.
        int myLen = latents.Length / dim;
        int goldenLen = golden.Length / dim;
        int commonLen = Math.Min(myLen, goldenLen);
        Assert.True(commonLen > 0, $"no overlapping positions: mine={myLen} golden={goldenLen}");

        var mineSlice = new float[dim * commonLen];
        var goldenSlice = new float[dim * commonLen];
        for (int d = 0; d < dim; d++)
        {
            Array.Copy(latents, d * myLen, mineSlice, d * commonLen, commonLen);
            Array.Copy(golden, d * goldenLen, goldenSlice, d * commonLen, commonLen);
        }

        foreach (float v in mineSlice)
        {
            Assert.False(float.IsNaN(v), "gpt_latents must not contain NaN");
            Assert.False(float.IsInfinity(v), "gpt_latents must not contain Infinity");
        }

        double cosine = Cosine(mineSlice, goldenSlice);
        Assert.True(cosine > 0.99, $"gpt_latents cosine {cosine} too low vs golden (mine len={myLen}, golden len={goldenLen}, compared common prefix={commonLen})");
    }
}
