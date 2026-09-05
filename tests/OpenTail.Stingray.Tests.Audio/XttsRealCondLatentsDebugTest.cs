
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: isolates whether XTTS-v2's "never stops, always hits the
/// 605-token cap" bug is in our conditioning encoder or in the GPT trunk/generation loop, by
/// feeding the REAL PyTorch reference's own dumped conditioning latents + text ids (see
/// scratch-llamacpp-ref/xtts_real_generation_length_check.py) directly into our C# generation
/// loop, bypassing our own conditioning computation entirely. If our port still fails to stop
/// given the REAL, known-correct conditioning, the bug is in the GPT trunk/generation loop, not
/// conditioning.</summary>
public sealed class XttsRealCondLatentsDebugTest : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadFloatCsv(string path) =>
        Array.ConvertAll(File.ReadAllText(path).Trim().Split(','), float.Parse);

    private static int[] ReadIntCsv(string path) =>
        Array.ConvertAll(File.ReadAllText(path).Trim().Split(','), int.Parse);

    [Fact]
    public void RealCondLatents_RealTextIds_DoesOurPortStop()
    {
        string? checkpointPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(checkpointPath != null, "XTTS checkpoint not found");
        string? condPath = FindRepoFile("scratch-llamacpp-ref/xtts_real_cond_latents.txt");
        string? shapePath = FindRepoFile("scratch-llamacpp-ref/xtts_real_cond_latents_shape.txt");
        string? textIdsPath = FindRepoFile("scratch-llamacpp-ref/xtts_real_text_ids.txt");
        string? realCodesPath = FindRepoFile("scratch-llamacpp-ref/xtts_real_generated_codes.txt");
        Assert.SkipUnless(condPath != null && shapePath != null && textIdsPath != null,
            "real cond-latent dump not found (run scratch-llamacpp-ref/xtts_real_generation_length_check.py first)");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(checkpointPath!);
        var trunkWeights = new OpenTail.Stingray.Audio.Xtts.XttsGptWeights(loader);
        var embWeights = new OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings(loader);

        int[] shape = ReadIntCsv(shapePath!); // [1, T, dim] token-major (PyTorch's own transpose(1,2) output)
        int numCondLatents = shape[1];
        int dim = shape[2];
        float[] condTokenMajor = ReadFloatCsv(condPath!);
        Assert.Equal(numCondLatents * dim, condTokenMajor.Length);

        // BuildPrefix expects channel-first [dim, numCondLatents] -- transpose from the dumped
        // token-major [numCondLatents, dim] layout.
        var condChannelFirst = new float[dim * numCondLatents];
        for (int t = 0; t < numCondLatents; t++)
            for (int d = 0; d < dim; d++)
                condChannelFirst[d * numCondLatents + t] = condTokenMajor[t * dim + d];

        int[] textIds = ReadIntCsv(textIdsPath!);
        int[]? realGreedyCodes = realCodesPath != null ? ReadIntCsv(realCodesPath!) : null;

        float[] prefix = OpenTail.Stingray.Audio.Xtts.XttsGptGenerator.BuildPrefix(embWeights, condChannelFirst, numCondLatents, textIds, out int prefixLen);

        var cache = new OpenTail.Stingray.Audio.Xtts.XttsGptCache(trunkWeights);
        cache.Reset();
        for (int i = 0; i < prefixLen; i++)
            OpenTail.Stingray.Audio.Xtts.XttsGptTrunk.Step(trunkWeights, cache, prefix.AsSpan(i * OpenTail.Stingray.Audio.Xtts.XttsGptWeights.ModelDim, OpenTail.Stingray.Audio.Xtts.XttsGptWeights.ModelDim));

        int startAudioToken = OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings.AudioStartToken;
        var startEmb = embWeights.EmbedSingleMel(startAudioToken, 0);
        float[] lastHidden = OpenTail.Stingray.Audio.Xtts.XttsGptTrunk.Step(trunkWeights, cache, startEmb).ToArray();

        var generated = new List<int>();
        int maxSteps = 150;
        bool stopped = false;
        for (int step = 0; step < maxSteps; step++)
        {
            float[] logits = embWeights.MelLogits(lastHidden);
            int next = ArgMax(logits);
            if (next == OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings.AudioStopToken)
            {
                stopped = true;
                Console.Error.WriteLine($"[XttsRealCond] OUR PORT stopped naturally at step={step} (real reference greedy stopped at step={realGreedyCodes?.Length - 1})");
                break;
            }
            generated.Add(next);
            if (step < 10 || step % 10 == 0)
                Console.Error.WriteLine($"[XttsRealCond] step={step} argmax={next} stopLogit={logits[OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings.AudioStopToken]:F3} topLogit={logits[next]:F3}");
            var nextEmb = embWeights.EmbedSingleMel(next, step + 1);
            lastHidden = OpenTail.Stingray.Audio.Xtts.XttsGptTrunk.Step(trunkWeights, cache, nextEmb).ToArray();
        }

        if (!stopped)
            Console.Error.WriteLine($"[XttsRealCond] OUR PORT did NOT stop within {maxSteps} steps, given the REAL reference's own conditioning latents + text ids.");

        if (realGreedyCodes != null)
            Console.Error.WriteLine($"[XttsRealCond] real reference greedy generated {realGreedyCodes.Length} tokens (last={realGreedyCodes[^1]}), ours generated {generated.Count} before {(stopped ? "stopping" : "cap")}.");
    }

    private static int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
}
