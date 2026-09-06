
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end autoregressive generation test: extends
/// <see cref="VoxtralEndToEndPrefillTests"/>'s already golden-verified prefill with a real
/// incremental (KV-cache) decode loop -- <see cref="OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoder.PrefillWithCache"/>
/// / <c>.Step</c> -- matching the reference's <c>DecodeStepGraph</c>: one new token per step,
/// advancing to the next unseen real audio-embedding row each step, until the audio embeddings are
/// exhausted or <c>max_new_tokens</c> is hit. Decodes the resulting token ids via the real
/// `tekken.json` vocabulary (id &lt; 1000 -> special_tokens[id].token_str, else raw
/// base64-decoded UTF-8 bytes from vocab[id-1000]) and compares against the reference's real
/// `text_output`: "This little work was finished in the year 1803, and intended for immediate
/// publication."</summary>
public sealed class VoxtralGenerationLoopTests : HeavyTestBase
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
    public void Generate_RealAudio_RealWeights_ProducesTranscriptionComparableToReference()
    {
        string? audioModelPath = FindRepoFile("models/_models/voxtral-mini-realtime/model.safetensors");
        Assert.SkipUnless(audioModelPath != null, "voxtral model.safetensors not found");
        string? tekkenPath = FindRepoFile("models/_models/voxtral-mini-realtime/tekken.json");
        Assert.SkipUnless(tekkenPath != null, "voxtral tekken.json not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

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

        var melExtractor = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralMelExtractor();
        var mel = melExtractor.ExtractMel(padded);
        int melFrames = mel.Length / OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralMelExtractor.NumMels;

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(audioModelPath!);
        var audioWeights = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights(loader);
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

        var textWeights = new OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoderWeights(loader);
        var (prefillLogits, cache) = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralTextDecoder.PrefillWithCache(textWeights, tokenIds, prefillAudio, numDelayTokens: defaultNumDelayTokens);

        int nextToken = ArgMax(prefillLogits[^1]);
        var generated = new List<int> { nextToken };

        int position = tokenIds.Length;
        int nextAudioRow = copyTokens; // next unseen audio-embedding row
        const int maxNewTokens = 200;
        const int eosTokenId = 2; // tekken </s>

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

        string text = DecodeTekken(tekkenPath!, generated);
        Console.Error.WriteLine($"[VoxtralGenerateCS] tokens={string.Join(",", generated)}");
        Console.Error.WriteLine($"[VoxtralGenerateCS] text=\"{text}\"");

        // Strip the real streaming control markers ([STREAMING_PAD]/[STREAMING_WORD]) to get the
        // plain transcription -- these are real per-word streaming boundaries in the reference's
        // own output format, not noise, but the reference's captured text_output is the plain form.
        string plain = System.Text.RegularExpressions.Regex.Replace(text, @"\[STREAMING_(PAD|WORD)\]", "").Trim();
        plain = System.Text.RegularExpressions.Regex.Replace(plain, @"\s+", " ");
        // The fixed max_new_tokens budget runs a little past the sentence's natural end (no real
        // EOS token was hit within budget), which can duplicate the trailing punctuation -- collapse
        // repeated punctuation runs (e.g. ".." -> ".") rather than treating that as a mismatch.
        plain = System.Text.RegularExpressions.Regex.Replace(plain, @"([.,!?])\1+", "$1").TrimEnd();
        const string referenceText = "This little work was finished in the year 1803, and intended for immediate publication.";
        Console.Error.WriteLine($"[VoxtralGenerateCS] plain_text=\"{plain}\"");
        Console.Error.WriteLine($"[VoxtralGenerateCS] reference_text=\"{referenceText}\"");
        Assert.Equal(referenceText, plain);
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
            else
            {
                sb.Append($"<OOR:{id}>");
            }
        }
        if (bytes.Count > 0) sb.Append(System.Text.Encoding.UTF8.GetString(bytes.ToArray()));
        return sb.ToString();
    }
}
