using OpenTail.Stingray.Audio.Xtts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for <see cref="XttsGptGenerator"/>'s first decode
/// step (real conditioning + real text embeddings + real GPT2 trunk + real mel_head), against
/// `scratch-llamacpp-ref/xtts_first_step_golden.py`'s real reference output. Ties together every
/// XTTS-v2 piece shipped so far into one real forward pass.</summary>
public sealed class XttsGptGeneratorTests : HeavyTestBase
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
    public void NextMelLogits_RealWeights_MatchesGoldenOracle_FirstStep()
    {
        string? weightsPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(weightsPath != null, "models/xtts-v2/model.safetensors not found");
        string? melPath = FindRepoFile("scratch-llamacpp-ref/xtts_first_step_golden_mel.txt");
        string? textIdsPath = FindRepoFile("scratch-llamacpp-ref/xtts_first_step_golden_text_ids.txt");
        string? logitsPath = FindRepoFile("scratch-llamacpp-ref/xtts_first_step_golden_mel_logits.txt");
        Assert.SkipUnless(melPath != null && textIdsPath != null && logitsPath != null,
            "golden first-step files not found (re-run scratch-llamacpp-ref/xtts_first_step_golden.py)");

        using var loader = SafetensorsLoader.Open(weightsPath!);
        var condWeights = new XttsConditioningWeights(loader);
        var trunkWeights = new XttsGptWeights(loader);
        var embWeights = new XttsGptEmbeddings(loader);

        float[] mel = ReadCsv(melPath!);
        int melT = mel.Length / XttsConditioningWeights.MelDim;
        float[] condLatents = XttsConditioningEncoder.Encode(condWeights, mel, melT);

        int[] textIds = Array.ConvertAll(File.ReadAllText(textIdsPath!).Trim().Split(','), int.Parse);

        float[] prefix = XttsGptGenerator.BuildPrefix(embWeights, condLatents, XttsConditioningWeights.PerceiverNumLatents, textIds, out int prefixLen);

        int[] melTokensSoFar = [XttsGptEmbeddings.AudioStartToken];
        float[] logits = XttsGptGenerator.NextMelLogits(trunkWeights, embWeights, prefix, prefixLen, melTokensSoFar);

        float[] golden = ReadCsv(logitsPath!);
        Assert.Equal(golden.Length, logits.Length);
        foreach (float v in logits)
        {
            Assert.False(float.IsNaN(v), "mel logits must not contain NaN");
            Assert.False(float.IsInfinity(v), "mel logits must not contain Infinity");
        }

        double cosine = Cosine(logits, golden);
        Assert.True(cosine > 0.99, $"first-step mel logits cosine {cosine} too low vs golden");

        // Real reference's top-5 highest-logit tokens should also come out on top here --
        // stronger check than cosine alone for a classification-style output.
        int argmax = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[argmax]) argmax = i;
        int goldenArgmax = 0;
        for (int i = 1; i < golden.Length; i++) if (golden[i] > golden[goldenArgmax]) goldenArgmax = i;
        Assert.Equal(goldenArgmax, argmax);
    }
}
