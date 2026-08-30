using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.Xtts;

/// <summary>
/// End-to-end XTTS-v2 voice-cloning TTS pipeline, wiring every already golden-verified real piece
/// (see `docs/audio-review-progress.md`'s XTTS-v2 entries) into one `Xtts.inference`-equivalent
/// call chain: BPE tokenizer -&gt; conditioning-encoder mel (real `wav_to_mel_cloning`,
/// n_fft=2048/hop=256/win=1024) -&gt; <see cref="XttsConditioningEncoder"/> (perceiver-resampled
/// style embedding) -&gt; GPT prefix -&gt; autoregressive mel-token sampling
/// (<see cref="XttsGptSampler"/>) -&gt; <see cref="XttsGptLatents"/> -&gt; speaker-encoder mel (real
/// `wav_to_mel_cloning`-style 16kHz frontend) -&gt; <see cref="XttsResNetEncoder"/> (L2-normalized
/// 512-dim d-vector, real `l2_norm=True` at this call site) -&gt; <see cref="XttsHifiDecoder"/>
/// (interpolation + vocoder) -&gt; waveform.
///
/// <para><b>Known scope limitation</b> (documented, not silently dropped): the real
/// `VoiceBpeTokenizer.preprocess_text`'s large per-language `multilingual_cleaners`/number-
/// expansion text-normalization pass is NOT ported (see <see cref="XttsBpeTokenizer"/>'s class
/// doc) -- callers should pass reasonably normalized text (numbers spelled out, no unusual
/// symbols) for best results, matching real-pipeline text-cleaning expectations for now.
/// Conditioning also uses the real reference's SINGLE-CHUNK path (`gpt_cond_len=gpt_cond_chunk_len
/// =6s`, so the real reference itself does no chunk-averaging in the default case either) rather
/// than the multi-chunk-averaging path for longer `gpt_cond_len` settings.</para>
/// </summary>
public sealed class XttsPipeline : ITextToSpeechPipeline
{
    public string Architecture => "XTTS-v2";
    public int DefaultSampleRate => 24000;

    private const int MaxRefSeconds = 30;
    private const int GptCondSeconds = 6;
    private const float MinRefSeconds = 0.33f;

    private readonly XttsGptWeights _gptWeights;
    private readonly XttsGptEmbeddings _gptEmb;
    private readonly XttsGptCache _gptCache;
    private readonly XttsConditioningWeights _condWeights;
    private readonly XttsResNetWeights _resNetWeights;
    private readonly XttsVocoderWeights _vocoderWeights;
    private readonly XttsBpeTokenizer _tokenizer;
    private readonly float[] _melStats;

    private XttsPipeline(XttsGptWeights gptWeights, XttsGptEmbeddings gptEmb, XttsConditioningWeights condWeights, XttsResNetWeights resNetWeights, XttsVocoderWeights vocoderWeights, XttsBpeTokenizer tokenizer, float[] melStats)
    {
        _gptWeights = gptWeights;
        _gptEmb = gptEmb;
        _gptCache = new XttsGptCache(gptWeights);
        _condWeights = condWeights;
        _resNetWeights = resNetWeights;
        _vocoderWeights = vocoderWeights;
        _tokenizer = tokenizer;
        _melStats = melStats;
    }

    /// <summary>Loads a real XTTS-v2 checkpoint directory containing `vocab.json`, `model.safetensors`, and `mel_stats.safetensors` (converted from the real `coqui/XTTS-v2` `model.pth`/`mel_stats.pth`).</summary>
    public static XttsPipeline Load(string checkpointDir)
    {
        string vocabPath = Path.Combine(checkpointDir, "vocab.json");
        string weightsPath = Path.Combine(checkpointDir, "model.safetensors");
        string melStatsPath = Path.Combine(checkpointDir, "mel_stats.safetensors");
        if (!File.Exists(vocabPath)) throw new FileNotFoundException($"XTTS-v2 vocab.json not found: {vocabPath}");
        if (!File.Exists(weightsPath)) throw new FileNotFoundException($"XTTS-v2 model.safetensors not found: {weightsPath}");
        if (!File.Exists(melStatsPath)) throw new FileNotFoundException($"XTTS-v2 mel_stats.safetensors not found: {melStatsPath}");

        using var loader = SafetensorsLoader.Open(weightsPath);
        var gptWeights = new XttsGptWeights(loader);
        var gptEmb = new XttsGptEmbeddings(loader);
        var condWeights = new XttsConditioningWeights(loader);
        var resNetWeights = new XttsResNetWeights(loader, "hifigan_decoder.speaker_encoder");
        var vocoderWeights = new XttsVocoderWeights(loader, "hifigan_decoder.waveform_decoder");
        var tokenizer = new XttsBpeTokenizer(vocabPath);

        using var melStatsLoader = SafetensorsLoader.Open(melStatsPath);
        float[] melStats = melStatsLoader.ReadF32("mel_stats");

        return new XttsPipeline(gptWeights, gptEmb, condWeights, resNetWeights, vocoderWeights, tokenizer, melStats);
    }

    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, (float[] CondLatents, int NumCondLatents, float[] SpeakerEmb)> _refCache = new();

    private (float[] CondLatents, int NumCondLatents, float[] SpeakerEmb) GetOrComputeReference(string referenceAudioPath)
    {
        return _refCache.GetOrAdd(referenceAudioPath, path =>
        {
            var (refPcm22050, refPcm16000) = LoadReferenceAudio(path);
            float[] condLatents = ComputeConditioningLatents(refPcm22050, out int numCondLatents);
            float[] speakerEmbedding = ComputeSpeakerEmbedding(refPcm16000);
            return (condLatents, numCondLatents, speakerEmbedding);
        });
    }

    /// <summary>
    /// Real voice-cloned synthesis. `referenceAudioPath` is a real reference wav (any sample rate,
    /// resampled internally); `lang` is a real XTTS-v2 language tag (`en`, `fr`, `de`, ...).
    /// `text` must already be reasonably normalized (see class doc's scope limitation).
    /// </summary>
    public float[] Generate(string text, string referenceAudioPath, string lang = "en", int? seed = null)
    {
        var (condLatents, numCondLatents, speakerEmbedding) = GetOrComputeReference(referenceAudioPath);

        string taggedText = $"[{(lang == "zh" ? "zh-cn" : lang)}]{text.Replace(" ", "[SPACE]")}";
        var textIds = _tokenizer.Encode(taggedText).ToArray();

        var prefix = XttsGptGenerator.BuildPrefix(_gptEmb, condLatents, numCondLatents, textIds, out int prefixLen);

        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        var (generatedCodes, gptLatents) = XttsGptGenerator.Generate(_gptWeights, _gptEmb, _gptCache, prefix, prefixLen, rng);
        if (generatedCodes.Count == 0) return [];

        int latentsT = gptLatents.Length / XttsGptWeights.ModelDim;
        return XttsHifiDecoder.Forward(_vocoderWeights, gptLatents, latentsT, speakerEmbedding);
    }

    private float[] ComputeConditioningLatents(float[] refPcm22050, out int numCondLatents)
    {
        int maxSamples = Math.Min(refPcm22050.Length, XttsMelExtractor.SampleRate * GptCondSeconds);
        var clipped = refPcm22050.AsSpan(0, maxSamples);

        var extractor = XttsMelExtractor.ForConditioningCloning();
        float[] mel = extractor.ExtractMel(clipped, _melStats);
        int t = mel.Length / XttsMelExtractor.NumMels;

        float[] latents = XttsConditioningEncoder.Encode(_condWeights, mel, t);
        numCondLatents = XttsConditioningWeights.PerceiverNumLatents;
        return latents;
    }

    private float[] ComputeSpeakerEmbedding(float[] refPcm16000)
    {
        float[] preemph = XttsSpeakerMelExtractor.Preemphasis(refPcm16000);
        var melExtractor = new XttsSpeakerMelExtractor();
        float[] mel = melExtractor.ExtractMel(preemph);
        int t = mel.Length / XttsSpeakerMelExtractor.NumMels;

        float[] embedding = XttsResNetEncoder.Forward(_resNetWeights, mel, t);

        // Real call site: `hifigan_decoder.speaker_encoder.forward(audio_16k, l2_norm=True)`.
        double normSq = 0;
        foreach (float v in embedding) normSq += v * v;
        float invNorm = (float)(1.0 / Math.Sqrt(Math.Max(normSq, 1e-12)));
        for (int i = 0; i < embedding.Length; i++) embedding[i] *= invNorm;
        return embedding;
    }

    private static (float[] Pcm22050, float[] Pcm16000) LoadReferenceAudio(string path)
    {
        var (samples, sr, _) = WavReader.ReadWav(path);
        int maxLen22050 = XttsMelExtractor.SampleRate * MaxRefSeconds;

        float[] pcm22050 = sr == XttsMelExtractor.SampleRate ? samples : AudioResampler.Resample(samples, sr, XttsMelExtractor.SampleRate);
        if (pcm22050.Length > maxLen22050) pcm22050 = pcm22050[..maxLen22050];
        if (pcm22050.Length < (int)(XttsMelExtractor.SampleRate * MinRefSeconds))
            throw new InvalidOperationException($"Reference audio too short (minimum {MinRefSeconds:0.00}s required).");

        float[] pcm16000 = sr == XttsSpeakerMelExtractor.SampleRate ? samples : AudioResampler.Resample(samples, sr, XttsSpeakerMelExtractor.SampleRate);
        return (pcm22050, pcm16000);
    }

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
            return new AudioGenerationResult([], DefaultSampleRate);
        if (string.IsNullOrEmpty(request.ReferenceAudioPath) || !File.Exists(request.ReferenceAudioPath))
            throw new InvalidOperationException("XTTS-v2 requires a reference audio clip via AudioGenerationRequest.ReferenceAudioPath (voice cloning only, no built-in speaker bank).");

        float[] samples = Generate(request.Text, request.ReferenceAudioPath);
        var result = new AudioGenerationResult(samples, DefaultSampleRate);
        if (!string.IsNullOrEmpty(request.OutputPath))
            result.SaveWav(request.OutputPath);
        return result;
    }

    /// <summary>
    /// Real-time streaming voice-cloned synthesis. Yields decoded PCM chunks (24kHz mono) with low latency
    /// as mel tokens are generated by the GPT trunk.
    /// </summary>
    public async IAsyncEnumerable<float[]> GenerateStreamAsync(
        string text,
        string referenceAudioPath,
        string lang = "en",
        int chunkTokens = 6,
        int contextTokens = 8,
        int? seed = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        var (condLatents, numCondLatents, speakerEmbedding) = GetOrComputeReference(referenceAudioPath);

        string taggedText = $"[{(lang == "zh" ? "zh-cn" : lang)}]{text.Replace(" ", "[SPACE]")}";
        var textIds = _tokenizer.Encode(taggedText).ToArray();

        var prefix = XttsGptGenerator.BuildPrefix(_gptEmb, condLatents, numCondLatents, textIds, out int prefixLen);
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();

        var allLatents = new List<float[]>();
        var pendingLatents = new List<float[]>();

        foreach (var (token, latent) in XttsGptGenerator.GenerateLatentsStream(_gptWeights, _gptEmb, _gptCache, prefix, prefixLen, rng))
        {
            ct.ThrowIfCancellationRequested();
            allLatents.Add(latent);
            pendingLatents.Add(latent);

            if (pendingLatents.Count >= chunkTokens)
            {
                var chunkPcm = DecodeLatentsChunk(allLatents, pendingLatents.Count, contextTokens, speakerEmbedding);
                pendingLatents.Clear();
                if (chunkPcm.Length > 0)
                    yield return chunkPcm;
            }
        }

        if (pendingLatents.Count > 0)
        {
            var chunkPcm = DecodeLatentsChunk(allLatents, pendingLatents.Count, contextTokens, speakerEmbedding);
            if (chunkPcm.Length > 0)
                yield return chunkPcm;
        }
        await Task.CompletedTask;
    }

    private float[] DecodeLatentsChunk(List<float[]> allLatents, int newLatentsCount, int contextTokens, float[] speakerEmbedding)
    {
        int totalLatents = allLatents.Count;
        int startLatent = Math.Max(0, totalLatents - newLatentsCount - contextTokens);
        int windowLatents = totalLatents - startLatent;
        int priorOverlapLatents = windowLatents - newLatentsCount;

        int dim = XttsGptWeights.ModelDim;
        var channelFirstSlice = new float[dim * windowLatents];
        for (int li = 0; li < windowLatents; li++)
        {
            var h = allLatents[startLatent + li];
            for (int d = 0; d < dim; d++)
                channelFirstSlice[d * windowLatents + li] = h[d];
        }

        var decodedWindow = XttsHifiDecoder.Forward(_vocoderWeights, channelFirstSlice, windowLatents, speakerEmbedding);

        // Compute ratio of samples per window latent
        double samplesPerLatent = (double)decodedWindow.Length / windowLatents;
        int skipSamples = (int)Math.Round(priorOverlapLatents * samplesPerLatent);
        int takeSamples = (int)Math.Round(newLatentsCount * samplesPerLatent);

        if (skipSamples + takeSamples <= decodedWindow.Length)
        {
            var chunk = new float[takeSamples];
            Array.Copy(decodedWindow, skipSamples, chunk, 0, takeSamples);
            return chunk;
        }
        else if (skipSamples < decodedWindow.Length)
        {
            int available = decodedWindow.Length - skipSamples;
            var chunk = new float[available];
            Array.Copy(decodedWindow, skipSamples, chunk, 0, available);
            return chunk;
        }

        return decodedWindow;
    }

    public IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, CancellationToken ct = default)
    {
        if (string.IsNullOrEmpty(request.ReferenceAudioPath) || !File.Exists(request.ReferenceAudioPath))
            throw new InvalidOperationException("XTTS-v2 requires a reference audio clip via AudioGenerationRequest.ReferenceAudioPath.");
        return GenerateStreamAsync(request.Text, request.ReferenceAudioPath, lang: "en", chunkTokens: 6, contextTokens: 8, seed: null, ct: ct);
    }

    public void Dispose() { }
}
