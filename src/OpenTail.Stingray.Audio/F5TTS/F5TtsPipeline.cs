namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// End-to-end Flow-Matching Diffusion Transformer (DiT) Text-to-Speech pipeline with Voice Cloning.
/// </summary>
public sealed class F5TtsPipeline : ITextToSpeechPipeline
{
    public string Architecture => "F5-TTS";
    public int DefaultSampleRate => 24000;

    private readonly F5MelExtractor _melExtractor;
    private readonly F5TextEncoder _textEncoder;
    private readonly F5DiTModel _ditModel;
    private readonly F5VocosVocoder _vocoder;

    public F5TtsPipeline(
        F5MelExtractor? melExtractor = null,
        F5TextEncoder? textEncoder = null,
        F5DiTModel? ditModel = null,
        F5VocosVocoder? vocoder = null)
    {
        _melExtractor = melExtractor ?? new F5MelExtractor();
        _textEncoder = textEncoder ?? new F5TextEncoder();
        _ditModel = ditModel ?? new F5DiTModel();
        _vocoder = vocoder ?? new F5VocosVocoder();
    }

    /// <summary>
    /// Synthesizes text into 24kHz speech with optional reference audio voice cloning.
    /// </summary>
    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            return new AudioGenerationResult([], DefaultSampleRate);
        }

        // Estimate target duration (roughly 12-16 characters per second, ~93.75 frames per second at hop=256, 24kHz)
        int charCount = request.Text.Length;
        float baseSeconds = Math.Max(1.0f, (float)charCount / 14.0f) / Math.Max(0.2f, request.Speed);
        int numFrames = (int)(baseSeconds * (DefaultSampleRate / 256.0f));
        numFrames = Math.Clamp(numFrames, 32, 2048);

        // 1. Encode Text Features via 4-Stage ConvNeXtV2
        float[] textFeatures = _textEncoder.Encode(request.Text, numFrames);

        // 2. Reference Audio Conditioning (Voice Cloning)
        float[] condMel = new float[numFrames * F5MelExtractor.NumMels];
        if (!string.IsNullOrEmpty(request.ReferenceAudioPath) && File.Exists(request.ReferenceAudioPath))
        {
            // Load and extract reference mel
            float[] refPcm = LoadPcmFromWav(request.ReferenceAudioPath);
            float[] refMel = _melExtractor.ExtractMel(refPcm);
            int refFrames = refMel.Length / F5MelExtractor.NumMels;

            // Copy ref conditioning onto prefix of condMel
            int copyFrames = Math.Min(refFrames, numFrames / 2);
            Array.Copy(refMel, 0, condMel, 0, copyFrames * F5MelExtractor.NumMels);
        }

        // 3. Solve 22-layer Flow-Matching DiT Trajectory
        float[] denoisedMel = _ditModel.SolveFlowMatchingOde(
            condMel: condMel,
            textFeatures: textFeatures,
            numFrames: numFrames,
            odeSteps: 32,
            cfgStrength: 2.0f,
            swayCoef: -1.0f,
            seed: 42);

        // 4. Synthesize 24kHz PCM Audio via Vocos Neural Vocoder
        float[] samples = _vocoder.Synthesize(denoisedMel, numFrames);

        var result = new AudioGenerationResult(samples, DefaultSampleRate);

        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }

        return result;
    }

    private static float[] LoadPcmFromWav(string wavPath)
    {
        byte[] bytes = File.ReadAllBytes(wavPath);
        if (bytes.Length < 44) return [];

        int sampleCount = (bytes.Length - 44) / 2;
        var pcm = new float[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            short s16 = BitConverter.ToInt16(bytes, 44 + i * 2);
            pcm[i] = s16 / 32768.0f;
        }

        return pcm;
    }

    /// <summary>
    /// Synthesizes text in streaming fashion, yielding clause/sentence audio waveforms as they are generated.
    /// </summary>
    public async IAsyncEnumerable<float[]> GenerateStreamAsync(
        AudioGenerationRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.Text)) yield break;

        var sentences = System.Text.RegularExpressions.Regex.Split(request.Text, @"(?<=[.!?,
])\s+");
        foreach (var s in sentences)
        {
            var trimmed = s.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            ct.ThrowIfCancellationRequested();

            var req = request with { Text = trimmed, OutputPath = null };
            var res = Generate(req);
            if (res.Samples.Length > 0)
            {
                yield return res.Samples;
            }
            await Task.Yield();
        }
    }

    public void Dispose()
    {
        _ditModel.Dispose();
    }
}

public sealed record F5AudioGenerationRequest : AudioGenerationRequest
{
}
