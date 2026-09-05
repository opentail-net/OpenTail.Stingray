
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: XTTS-v2 generation always hits the 605-token hard safety cap
/// (never naturally samples stop_audio_token=1025) in real end-to-end runs, producing ~28s of
/// audio for a 6-word sentence -- confirmed via direct trace, with the vocoder's own upsampling
/// math independently checked correct (expected/actual sample counts match to within 0.02%). This
/// test isolates whether the KV-cached incremental path (<see cref="XttsGptTrunk.Step"/>, what
/// production generation actually uses) drifts from a full, uncached recompute
/// (<see cref="XttsGptGenerator.NextMelLogits"/>, the only path this codebase's existing golden
/// test actually verifies -- and only at step 0) as the token history grows.</summary>
public sealed class XttsCacheDriftDebugTest : HeavyTestBase
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

    [Fact]
    public void CachedStep_MatchesFullRecompute_AcrossGrowingHistory()
    {
        string? checkpointPath = FindRepoFile("models/xtts-v2/model.safetensors");
        Assert.SkipUnless(checkpointPath != null, "XTTS checkpoint not found");
        string? refWav = FindRepoFile("docs/audio-samples/fishspeech-lunch-REFERENCE.wav");
        Assert.SkipUnless(refWav != null, "reference audio not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(checkpointPath!);
        var condWeights = new OpenTail.Stingray.Audio.Xtts.XttsConditioningWeights(loader);
        var trunkWeights = new OpenTail.Stingray.Audio.Xtts.XttsGptWeights(loader);
        var embWeights = new OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings(loader);
        string? checkpointDir = Path.GetDirectoryName(checkpointPath!);
        string? vocabPath = FindRepoFile(Path.Combine(checkpointDir!, "vocab.json"));
        Assert.SkipUnless(vocabPath != null, "XTTS vocab.json not found");
        var tokenizer = new OpenTail.Stingray.Audio.Xtts.XttsBpeTokenizer(vocabPath!);

        // Build a real prefix the same way XttsPipeline.Generate does.
        var (samples, sr, _) = OpenTail.Stingray.Audio.WavReader.ReadWav(refWav!);
        float[] pcm22050 = sr == OpenTail.Stingray.Audio.Xtts.XttsMelExtractor.SampleRate
            ? samples : OpenTail.Stingray.Audio.AudioResampler.Resample(samples, sr, OpenTail.Stingray.Audio.Xtts.XttsMelExtractor.SampleRate);
        int maxSamples = Math.Min(pcm22050.Length, OpenTail.Stingray.Audio.Xtts.XttsMelExtractor.SampleRate * 6);
        var clipped = pcm22050.AsSpan(0, maxSamples);
        var extractor = OpenTail.Stingray.Audio.Xtts.XttsMelExtractor.ForConditioningCloning();
        using var melStatsLoader = OpenTail.Stingray.Core.SafetensorsLoader.Open(Path.Combine(checkpointDir!, "mel_stats.safetensors"));
        float[] melStats = melStatsLoader.ReadF32("mel_stats");
        float[] mel = extractor.ExtractMel(clipped, melStats);
        int melT = mel.Length / OpenTail.Stingray.Audio.Xtts.XttsMelExtractor.NumMels;
        float[] condLatents = OpenTail.Stingray.Audio.Xtts.XttsConditioningEncoder.Encode(condWeights, mel, melT);
        int numCondLatents = OpenTail.Stingray.Audio.Xtts.XttsConditioningWeights.PerceiverNumLatents;

        string text = "hello, i will make some lunch, darling!";
        string taggedText = $"[en]{text.Replace(" ", "[SPACE]")}";
        int[] textIds = tokenizer.Encode(taggedText).ToArray();

        float[] prefix = OpenTail.Stingray.Audio.Xtts.XttsGptGenerator.BuildPrefix(embWeights, condLatents, numCondLatents, textIds, out int prefixLen);

        // Drive the CACHED path manually so we retain raw (pre-final-norm) hidden states.
        var cache = new OpenTail.Stingray.Audio.Xtts.XttsGptCache(trunkWeights);
        cache.Reset();
        int dim = OpenTail.Stingray.Audio.Xtts.XttsGptWeights.ModelDim;
        for (int i = 0; i < prefixLen; i++)
            OpenTail.Stingray.Audio.Xtts.XttsGptTrunk.Step(trunkWeights, cache, prefix.AsSpan(i * dim, dim));

        int startAudioToken = OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings.AudioStartToken;
        var startEmb = embWeights.EmbedSingleMel(startAudioToken, 0);
        float[] lastHidden = OpenTail.Stingray.Audio.Xtts.XttsGptTrunk.Step(trunkWeights, cache, startEmb).ToArray();

        var melTokensSoFar = new List<int> { startAudioToken };
        int[] checkpoints = [1, 5, 10, 20, 40];
        int nextCheckpointIdx = 0;

        for (int step = 0; step < 45; step++)
        {
            float[] cachedLogits = embWeights.MelLogits(lastHidden);

            if (nextCheckpointIdx < checkpoints.Length && step == checkpoints[nextCheckpointIdx])
            {
                float[] fullLogits = OpenTail.Stingray.Audio.Xtts.XttsGptGenerator.NextMelLogits(
                    trunkWeights, embWeights, prefix, prefixLen, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(melTokensSoFar));

                int cachedArgmax = ArgMax(cachedLogits);
                int fullArgmax = ArgMax(fullLogits);
                double cos = Cosine(cachedLogits, fullLogits);
                float cachedStopLogit = cachedLogits[OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings.AudioStopToken];
                float fullStopLogit = fullLogits[OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings.AudioStopToken];

                Console.Error.WriteLine($"[XttsCacheDrift] step={step} cachedArgmax={cachedArgmax} fullArgmax={fullArgmax} " +
                    $"cosineSim={cos:F6} cachedStopLogit={cachedStopLogit:F3} fullStopLogit={fullStopLogit:F3} " +
                    $"cachedTop3Logit={Top3(cachedLogits)} fullTop3Logit={Top3(fullLogits)}");

                nextCheckpointIdx++;
            }

            // Advance using the FULL RECOMPUTE's greedy choice, so both paths see the exact same
            // history (isolates cache drift from ordinary sampling-noise divergence).
            float[] logitsForAdvance = OpenTail.Stingray.Audio.Xtts.XttsGptGenerator.NextMelLogits(
                trunkWeights, embWeights, prefix, prefixLen, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(melTokensSoFar));
            int next = ArgMax(logitsForAdvance);
            if (next == OpenTail.Stingray.Audio.Xtts.XttsGptEmbeddings.AudioStopToken)
            {
                Console.Error.WriteLine($"[XttsCacheDrift] full-recompute greedy path naturally stopped at step={step}");
                break;
            }
            melTokensSoFar.Add(next);
            var nextEmb = embWeights.EmbedSingleMel(next, step + 1);
            lastHidden = OpenTail.Stingray.Audio.Xtts.XttsGptTrunk.Step(trunkWeights, cache, nextEmb).ToArray();
        }
    }

    private static int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    private static string Top3(float[] a)
    {
        var idx = new int[a.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        Array.Sort(idx, (x, y) => a[y].CompareTo(a[x]));
        return $"[{idx[0]}:{a[idx[0]]:F2}, {idx[1]}:{a[idx[1]]:F2}, {idx[2]}:{a[idx[2]]:F2}]";
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }
}
