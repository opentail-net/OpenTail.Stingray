using System.Runtime.CompilerServices;

namespace OpenTail.Stingray.Audio.VoxtralRealtime;

/// <summary>
/// Real end-to-end Voxtral-Mini-4B-Realtime ASR pipeline: mel-extract → audio-tower forward →
/// text-decoder prefill (with the audio-embedding splice) → incremental (KV-cache) greedy decode
/// → tekken vocabulary decode. This wraps the exact sequence already golden-verified against the
/// real reference by <see cref="VoxtralEndToEndPrefillTests"/>/<see cref="VoxtralGenerationLoopTests"/>
/// (perf-sweep Phase 1.4, docs/perf-sweep-plan.md) -- those tests exercised this logic by hand in
/// test code; nothing outside tests could reach it before this class existed, so kernel speed
/// (Phases 1.1-1.2e) was moot for any real usage. No new algorithm here, only real wiring.
///
/// <para>Not yet a full production ASR pipeline: no VAD-based chunking, no true streaming decode
/// (see <see cref="TranscribeStreamAsync"/>'s doc comment), and <see cref="SpeechToTextRequest.Language"/>/
/// <see cref="SpeechToTextRequest.Task"/>/<see cref="SpeechToTextRequest.BeamSize"/> are accepted
/// but not yet honored (greedy decode only, matching the reference default). Segments are not
/// split by real word/silence boundaries -- one segment covering the whole utterance.</para>
/// </summary>
public sealed class VoxtralPipeline : ISpeechToTextPipeline
{
    public string Architecture => "Voxtral-Mini-4B-Realtime";
    public int SampleRate => 16000;

    private const int HopLength = 160;
    private const int AudioLengthPerTok = 8;
    private const int Unit = AudioLengthPerTok * HopLength;
    private const int DefaultNumDelayTokens = 6;
    private const int OfflineStreamingBufferTokens = 10;
    private const int PromptPadTokens = 32 + DefaultNumDelayTokens;
    private const int BosTokenId = 1;
    private const int PadTokenId = 32;
    private const int EosTokenId = 2;
    private const int MaxNewTokens = 448; // generous headroom over the 200 used in tests' short clip

    private readonly VoxtralAudioEncoderWeights _audioWeights;
    private readonly VoxtralTextDecoderWeights _textWeights;
    private readonly VoxtralMelExtractor _melExtractor;
    private readonly TekkenVocab _vocab;

    private VoxtralPipeline(VoxtralAudioEncoderWeights audioWeights, VoxtralTextDecoderWeights textWeights, TekkenVocab vocab)
    {
        _audioWeights = audioWeights;
        _textWeights = textWeights;
        _melExtractor = new VoxtralMelExtractor();
        _vocab = vocab;
    }

    /// <summary>Loads real weights from a checkpoint directory containing
    /// <c>model.safetensors</c> and <c>tekken.json</c> (the real Voxtral-Mini-4B-Realtime release
    /// layout, e.g. <c>models/_models/voxtral-mini-realtime/</c>).</summary>
    public static VoxtralPipeline Load(string checkpointDir)
    {
        string safetensorsPath = Path.Combine(checkpointDir, "model.safetensors");
        string tekkenPath = Path.Combine(checkpointDir, "tekken.json");
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"Voxtral model.safetensors not found: {safetensorsPath}");
        if (!File.Exists(tekkenPath))
            throw new FileNotFoundException($"Voxtral tekken.json not found: {tekkenPath}");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(safetensorsPath);
        var audioWeights = new VoxtralAudioEncoderWeights(loader);
        var textWeights = new VoxtralTextDecoderWeights(loader);
        var vocab = TekkenVocab.Load(tekkenPath);
        return new VoxtralPipeline(audioWeights, textWeights, vocab);
    }

    public SpeechToTextResult Transcribe(SpeechToTextRequest request)
    {
        float[] samples = request.AudioSamples;
        var durationSeconds = samples.Length / (double)SampleRate;

        // Real left/right padding scheme, matching VoxtralGenerationLoopTests exactly (derived
        // from the reference's own offline-streaming buffer conventions, not guessed).
        long leftPad = 32L * Unit;
        long rightBase = (Unit - (samples.Length % Unit)) % Unit;
        long rightExtra = (long)(DefaultNumDelayTokens + 1 + OfflineStreamingBufferTokens) * Unit;
        var padded = new float[leftPad + samples.Length + rightBase + rightExtra];
        Array.Copy(samples, 0, padded, leftPad, samples.Length);

        var mel = _melExtractor.ExtractMel(padded);
        int melFrames = mel.Length / VoxtralMelExtractor.NumMels;
        var audioEmbeddings = VoxtralAudioEncoder.Forward(_audioWeights, mel, melFrames);

        var tokenIds = new int[1 + PromptPadTokens];
        tokenIds[0] = BosTokenId;
        for (int i = 1; i < tokenIds.Length; i++) tokenIds[i] = PadTokenId;

        int capacity = tokenIds.Length;
        int copyTokens = Math.Min(audioEmbeddings.Length, capacity);
        var prefillAudio = new float[capacity][];
        for (int t = 0; t < capacity; t++)
            prefillAudio[t] = t < copyTokens ? audioEmbeddings[t] : new float[VoxtralTextDecoderWeights.HiddenSize];

        var (prefillLogits, cache) = VoxtralTextDecoder.PrefillWithCache(_textWeights, tokenIds, prefillAudio, DefaultNumDelayTokens);

        int nextToken = ArgMax(prefillLogits[^1]);
        var generated = new List<int> { nextToken };
        int position = tokenIds.Length;
        int nextAudioRow = copyTokens;

        for (int step = 0; step < MaxNewTokens && nextToken != EosTokenId; step++)
        {
            float[]? audioRow = nextAudioRow < audioEmbeddings.Length ? audioEmbeddings[nextAudioRow] : null;
            if (audioRow != null) nextAudioRow++;

            var stepLogits = VoxtralTextDecoder.Step(_textWeights, cache, nextToken, audioRow, position, DefaultNumDelayTokens);
            nextToken = ArgMax(stepLogits);
            position++;
            if (nextToken == EosTokenId) break;
            generated.Add(nextToken);
            request.Progress?.Invoke(step + 1, MaxNewTokens);
        }

        string rawText = _vocab.Decode(generated);
        string plain = System.Text.RegularExpressions.Regex.Replace(rawText, @"\[STREAMING_(PAD|WORD)\]", "").Trim();
        plain = System.Text.RegularExpressions.Regex.Replace(plain, @"\s+", " ");
        plain = System.Text.RegularExpressions.Regex.Replace(plain, @"([.,!?])\1+", "$1").TrimEnd();

        var segment = new SpeechSegment
        {
            Id = 0,
            Start = TimeSpan.Zero,
            End = TimeSpan.FromSeconds(durationSeconds),
            Text = plain,
            Tokens = [.. generated],
        };
        return new SpeechToTextResult(plain, request.Language ?? "en", TimeSpan.FromSeconds(durationSeconds), [segment]);
    }

    /// <summary>No true streaming decode implemented yet (perf-sweep-plan.md Phase 1.4 follow-up
    /// -- this pipeline's <see cref="VoxtralTextDecoder.Step"/> already supports true incremental
    /// per-token decode, but chunked/partial-audio re-prefill as new audio arrives is not wired).
    /// Buffers the whole stream and runs one real offline pass, matching
    /// <see cref="ParaformerOnnxPipeline"/>'s same documented limitation for its own architecture.</summary>
    public async IAsyncEnumerable<SpeechSegment> TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audioStream,
        SpeechToTextRequest baseRequest,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var buffer = new List<float>();
        await foreach (var chunk in audioStream.WithCancellation(ct))
            buffer.AddRange(chunk.ToArray());

        if (buffer.Count == 0) yield break;
        var result = Transcribe(baseRequest with { AudioSamples = [.. buffer] });
        foreach (var segment in result.Segments) yield return segment;
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    public void Dispose() { }
}

/// <summary>Real tekken.json vocabulary decode: id &lt; 1000 -> special_tokens[id].token_str,
/// else raw base64-decoded UTF-8 bytes from vocab[id-1000] -- extracted from
/// <see cref="VoxtralGenerationLoopTests"/>'s inline <c>DecodeTekken</c> so both the test and this
/// real pipeline share one implementation instead of two copies (perf-sweep Phase 1.4's DRY
/// follow-through per CLAUDE.md rule 7).</summary>
public sealed class TekkenVocab
{
    private const int NumSpecial = 1000;
    private readonly Dictionary<int, string> _specialByRank;
    private readonly Dictionary<int, byte[]> _vocabByRank;

    private TekkenVocab(Dictionary<int, string> specialByRank, Dictionary<int, byte[]> vocabByRank)
    {
        _specialByRank = specialByRank;
        _vocabByRank = vocabByRank;
    }

    public static TekkenVocab Load(string tekkenPath)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(tekkenPath));
        var root = doc.RootElement;

        var specialByRank = new Dictionary<int, string>();
        foreach (var s in root.GetProperty("special_tokens").EnumerateArray())
            specialByRank[s.GetProperty("rank").GetInt32()] = s.GetProperty("token_str").GetString() ?? "";

        var vocabByRank = new Dictionary<int, byte[]>();
        foreach (var v in root.GetProperty("vocab").EnumerateArray())
            vocabByRank[v.GetProperty("rank").GetInt32()] = Convert.FromBase64String(v.GetProperty("token_bytes").GetString() ?? "");

        return new TekkenVocab(specialByRank, vocabByRank);
    }

    public string Decode(IReadOnlyList<int> tokenIds)
    {
        var bytes = new List<byte>();
        var sb = new System.Text.StringBuilder();
        foreach (var id in tokenIds)
        {
            if (id < NumSpecial)
            {
                if (bytes.Count > 0) { sb.Append(System.Text.Encoding.UTF8.GetString(bytes.ToArray())); bytes.Clear(); }
                sb.Append(_specialByRank.TryGetValue(id, out var s) ? s : $"<UNK:{id}>");
            }
            else if (_vocabByRank.TryGetValue(id - NumSpecial, out var b))
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
