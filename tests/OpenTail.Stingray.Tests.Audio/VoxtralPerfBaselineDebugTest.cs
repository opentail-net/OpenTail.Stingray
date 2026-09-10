
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for Voxtral-Mini-4B-Realtime against the real
/// audio.cpp reference (examples/audio.cpp/assets/resources/b.wav, 14.1s), for
/// PerformanceLeague.md backfill. Not part of the permanent suite, delete after use.
/// </summary>
public sealed class VoxtralPerfBaselineDebugTest : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Bench_Transcribe_BWav_RealWeights()
    {
        string? audioModelPath = FindRepoFile("models/_models/voxtral-mini-realtime/model.safetensors");
        Assert.SkipUnless(audioModelPath != null, "voxtral model.safetensors not found");
        string? tekkenPath = FindRepoFile("models/_models/voxtral-mini-realtime/tekken.json");
        Assert.SkipUnless(tekkenPath != null, "voxtral tekken.json not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/b.wav");
        Assert.SkipUnless(audioPath != null, "reference b.wav not found");

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);
        double audioSec = samples.Length / 16000.0;

        const int hopLength = 160;
        const int audioLengthPerTok = 8;
        const int unit = audioLengthPerTok * hopLength;
        const int defaultNumDelayTokens = 6;
        const int kOfflineStreamingBufferTokens = 10;
        long leftPad = 32L * unit;
        long rightBase = (unit - (samples.Length % unit)) % unit;
        long rightExtra = (long)(defaultNumDelayTokens + 1 + kOfflineStreamingBufferTokens) * unit;
        var padded = new float[leftPad + samples.Length + rightBase + rightExtra];
        Array.Copy(samples, 0, padded, leftPad, samples.Length);

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(audioModelPath!);
        var audioWeights = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights(loader);
        var textWeights = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights(loader);

        // Warmup (not timed).
        {
            var melExtractorW = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralMelExtractor();
            var melW = melExtractorW.ExtractMel(padded);
            int melFramesW = melW.Length / OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralMelExtractor.NumMels;
            _ = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoder.Forward(audioWeights, melW, melFramesW);
        }

        const int runs = 3;
        double[] elapsedSec = new double[runs];
        int lastTokenCount = 0;
        string lastText = "";

        for (int run = 0; run < runs; run++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            var melExtractor = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralMelExtractor();
            var mel = melExtractor.ExtractMel(padded);
            int melFrames = mel.Length / OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralMelExtractor.NumMels;
            var audioEmbeddings = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoder.Forward(audioWeights, mel, melFrames);

            const int promptPadTokens = 32 + defaultNumDelayTokens;
            var tokenIds = new int[1 + promptPadTokens];
            tokenIds[0] = 1;
            for (int i = 1; i < tokenIds.Length; i++) tokenIds[i] = 32;

            int capacity = tokenIds.Length;
            int copyTokens = Math.Min(audioEmbeddings.Length, capacity);
            var prefillAudio = new float[capacity][];
            for (int t = 0; t < capacity; t++)
                prefillAudio[t] = t < copyTokens ? audioEmbeddings[t] : new float[OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights.HiddenSize];

            var (prefillLogits, cache) = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoder.PrefillWithCache(textWeights, tokenIds, prefillAudio, numDelayTokens: defaultNumDelayTokens);

            int nextToken = ArgMax(prefillLogits[^1]);
            var generated = new List<int> { nextToken };

            int position = tokenIds.Length;
            int nextAudioRow = copyTokens;
            const int maxNewTokens = 200;
            const int eosTokenId = 2;

            for (int step = 0; step < maxNewTokens; step++)
            {
                float[]? audioRow = nextAudioRow < audioEmbeddings.Length ? audioEmbeddings[nextAudioRow] : null;
                if (audioRow != null) nextAudioRow++;

                var stepLogits = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoder.Step(textWeights, cache, nextToken, audioRow, position, defaultNumDelayTokens);
                nextToken = ArgMax(stepLogits);
                position++;
                if (nextToken == eosTokenId) break;
                generated.Add(nextToken);
            }

            sw.Stop();
            elapsedSec[run] = sw.Elapsed.TotalSeconds;
            lastTokenCount = generated.Count;
            lastText = DecodeTekken(tekkenPath!, generated);
        }

        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSec;
        string msg = $"[Voxtral-Perf] audio={audioSec:F2}s tokens={lastTokenCount} text=\"{lastText}\"\n" +
                     $"[Voxtral-Perf] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} x_realtime={1.0 / rtf:F3}";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static string DecodeTekken(string tekkenPath, List<int> tokenIds)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(tekkenPath));
        var root = doc.RootElement;
        var specials = root.GetProperty("special_tokens");
        var vocab = root.GetProperty("vocab");

        var specialByRank = new Dictionary<int, string>();
        foreach (var s in specials.EnumerateArray())
            specialByRank[s.GetProperty("rank").GetInt32()] = s.GetProperty("token_str").GetString() ?? "";

        var vocabByRank = new Dictionary<int, byte[]>();
        foreach (var v in vocab.EnumerateArray())
            vocabByRank[v.GetProperty("rank").GetInt32()] = Convert.FromBase64String(v.GetProperty("token_bytes").GetString() ?? "");

        int numSpecial = 1000;
        var bytes = new List<byte>();
        var sb = new System.Text.StringBuilder();
        foreach (var id in tokenIds)
        {
            if (id < numSpecial)
            {
                if (bytes.Count > 0) { sb.Append(System.Text.Encoding.UTF8.GetString(bytes.ToArray())); bytes.Clear(); }
                sb.Append(specialByRank.TryGetValue(id, out var s) ? s : $"<UNK:{id}>");
            }
            else if (vocabByRank.TryGetValue(id - numSpecial, out var b))
            {
                bytes.AddRange(b);
            }
        }
        if (bytes.Count > 0) sb.Append(System.Text.Encoding.UTF8.GetString(bytes.ToArray()));
        return sb.ToString();
    }
}
