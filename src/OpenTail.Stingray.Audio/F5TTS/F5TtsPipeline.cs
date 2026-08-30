
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// End-to-end Flow-Matching Diffusion Transformer (DiT) Text-to-Speech pipeline with Voice Cloning.
///
/// When constructed with a real `.safetensors` weights file (see <see cref="Load"/>), the text
/// encoding + DiT + flow-matching ODE stages are real and weight-driven (verified against the
/// real PyTorch reference, see tests/OpenTail.Stingray.Tests.Audio/F5DiTModelTests.cs). The
/// Vocos vocoder stage (mel -&gt; waveform) is ALSO real when a Vocos weights file is found
/// (`models/vocos-mel-24khz.safetensors`, converted from `charactr/vocos-mel-24khz`'s
/// `pytorch_model.bin` -- see <see cref="VocosWeights"/>'s class doc; verified against the real
/// PyTorch reference, see tests/OpenTail.Stingray.Tests.Audio/VocosVocoderTests.cs); falls back
/// to the original procedural placeholder (`F5VocosVocoder`) if that file isn't present.
/// See docs/audio-review-progress.md's F5-TTS section for the full status.
/// </summary>
public sealed class F5TtsPipeline : ITextToSpeechPipeline
{
    public string Architecture => "F5-TTS";
    public int DefaultSampleRate => 24000;

    private readonly F5MelExtractor _melExtractor;
    private readonly F5TextEncoder _textEncoder;
    private readonly F5VocosVocoder _vocoder;
    private readonly F5TtsWeights? _weights;
    private readonly F5Tokenizer? _tokenizer;
    private readonly VocosWeights? _vocosWeights;

    public F5TtsPipeline(
        F5MelExtractor? melExtractor = null,
        F5TextEncoder? textEncoder = null,
        F5VocosVocoder? vocoder = null,
        F5TtsWeights? weights = null,
        F5Tokenizer? tokenizer = null,
        VocosWeights? vocosWeights = null)
    {
        _weights = weights;
        _tokenizer = tokenizer;
        _vocosWeights = vocosWeights;
        _melExtractor = melExtractor ?? new F5MelExtractor();
        _textEncoder = textEncoder ?? new F5TextEncoder();
        _vocoder = vocoder ?? new F5VocosVocoder();
    }

    /// <summary>
    /// Loads a real F5-TTS pipeline directly from a safetensors model file. `vocabPath` defaults
    /// to `models/f5tts_vocab.txt` next to the weights file if not given (falls back to the fake
    /// text path if neither is found, same as the no-weights constructor). `vocosPath` defaults
    /// to `models/vocos-mel-24khz.safetensors` next to the weights file if not given (falls back
    /// to the fake vocoder if not found).
    /// </summary>
    public static F5TtsPipeline Load(string safetensorsPath, string? vocabPath = null, string? vocosPath = null)
    {
        if (string.IsNullOrWhiteSpace(safetensorsPath) || !File.Exists(safetensorsPath))
            throw new FileNotFoundException($"F5-TTS model file not found: {safetensorsPath}");

        var weights = new F5TtsWeights(safetensorsPath);
        var melExtractor = new F5MelExtractor();
        var textEncoder = new F5TextEncoder();
        var vocoder = new F5VocosVocoder();

        string modelDir = Path.GetDirectoryName(safetensorsPath) ?? ".";
        string resolvedVocabPath = vocabPath ?? Path.Combine(modelDir, "f5tts_vocab.txt");
        F5Tokenizer? tokenizer = File.Exists(resolvedVocabPath) ? new F5Tokenizer(resolvedVocabPath) : null;

        string resolvedVocosPath = vocosPath ?? Path.Combine(modelDir, "vocos-mel-24khz.safetensors");
        VocosWeights? vocosWeights = File.Exists(resolvedVocosPath) ? new VocosWeights(resolvedVocosPath) : null;

        return new F5TtsPipeline(melExtractor, textEncoder, vocoder, weights, tokenizer, vocosWeights);
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

        // 1. Reference Audio Conditioning (Voice Cloning)
        int refFrames = 0;
        float[]? refMel = null;
        string fullText = request.Text;
        string refText = "";

        if (!string.IsNullOrEmpty(request.ReferenceAudioPath) && File.Exists(request.ReferenceAudioPath))
        {
            float[] refPcm = LoadPcmFromWav(request.ReferenceAudioPath);
            refMel = _melExtractor.ExtractMel(refPcm);
            refFrames = refMel.Length / F5MelExtractor.NumMels;

            refText = request.ReferenceText ?? "";
            if (!string.IsNullOrEmpty(refText))
            {
                if (!refText.EndsWith(' ')) refText += " ";
                fullText = refText + request.Text;
            }
        }

        // 2. Target duration estimation. Real reference formula (utils_infer.py's
        // infer_batch_process): duration = ref_audio_len + int(ref_audio_len/ref_text_len *
        // gen_text_len / speed) -- scales the generated length by the REFERENCE clip's own real
        // speaking pace (frames per UTF-8 byte of its transcript), not a fixed chars-per-second
        // guess. A flat heuristic under/over-estimates whenever the reference speaker's pace
        // differs from the guessed rate, clipping the tail of the generated audio when it
        // under-estimates (confirmed: this was cutting off word endings like "-ing").
        int genTextBytes = System.Text.Encoding.UTF8.GetByteCount(request.Text);
        float speed = Math.Max(0.2f, request.Speed);
        int totalFrames;
        if (refFrames > 0 && !string.IsNullOrEmpty(refText))
        {
            int refTextBytes = System.Text.Encoding.UTF8.GetByteCount(refText);
            int genFrames = refTextBytes > 0 ? (int)(refFrames / (float)refTextBytes * genTextBytes / speed) : 0;
            totalFrames = refFrames + genFrames;
        }
        else
        {
            // No reference pace available (text-only synthesis): fall back to the flat heuristic.
            float genSeconds = Math.Max(1.0f, genTextBytes / 14.0f) / speed;
            int genFrames = (int)(genSeconds * (DefaultSampleRate / 256.0f));
            totalFrames = refFrames + genFrames;
        }
        totalFrames = Math.Clamp(totalFrames, 32, 2048);

        float[] condMel = new float[totalFrames * F5MelExtractor.NumMels];
        if (refMel is not null && refFrames > 0)
        {
            int copyFrames = Math.Min(refFrames, totalFrames);
            Array.Copy(refMel, 0, condMel, 0, copyFrames * F5MelExtractor.NumMels);
        }

        float[] generatedMel;
        if (_weights is not null && _tokenizer is not null)
        {
            int[] tokens = _tokenizer.Encode(fullText);
            generatedMel = F5FlowMatchingOde.Solve(_weights, condMel, tokens, totalFrames, steps: 32, cfgStrength: 2.0f, swaySamplingCoef: -1.0f);
        }
        else
        {
            // Fake/placeholder path (no real weights or vocab file) -- old procedural DiT stand-in.
            float[] textFeatures = _textEncoder.Encode(fullText, totalFrames);
            generatedMel = FakeSolveFlowMatchingOde(condMel, textFeatures, totalFrames);
        }

        // 3. Slice out the generated mel frames (omitting the reference audio prompt if present)
        float[] targetMel;
        int targetFrames;
        if (refFrames > 0 && totalFrames > refFrames)
        {
            targetFrames = totalFrames - refFrames;
            targetMel = new float[targetFrames * F5MelExtractor.NumMels];
            Array.Copy(generatedMel, refFrames * F5MelExtractor.NumMels, targetMel, 0, targetMel.Length);
        }
        else
        {
            targetFrames = totalFrames;
            targetMel = generatedMel;
        }

        // 4. Vocos Waveform Synthesis (Mel -> 24kHz Audio)
        float[] audio = _vocosWeights is not null
            ? VocosVocoder.Decode(_vocosWeights, targetMel, targetFrames)
            : _vocoder.Synthesize(targetMel, targetFrames);

        var result = new AudioGenerationResult(audio, DefaultSampleRate);

        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }

        return result;
    }

    /// <summary>
    /// Synthesizes text in streaming fashion, yielding clause/sentence audio waveforms as they are generated.
    /// </summary>
    public IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, CancellationToken ct = default)
        => TtsStreamingHelper.SplitAndGenerateAsync(request, Generate, ct);

    /// <summary>
    /// Fake/placeholder flow-matching ODE + DiT stand-in (no real weights) -- kept so parameterless
    /// callers (e.g. Fast tests) still get a valid, fast, non-real waveform. This is the original
    /// procedural implementation that used to live in F5DiTModel.cs before that class was rewritten
    /// to be the real weight-driven port (see docs/audio-review-progress.md's F5-TTS section).
    /// </summary>
    private static float[] FakeSolveFlowMatchingOde(float[] condMel, float[] textFeatures, int numFrames, int odeSteps = 8, float cfgStrength = 2.0f, int seed = 42)
    {
        const int inChannels = 100, hiddenDim = 1024;
        var rng = new Random(seed);
        var x = new float[numFrames * inChannels];
        for (int i = 0; i < x.Length; i++)
        {
            float u1 = 1.0f - (float)rng.NextDouble();
            float u2 = 1.0f - (float)rng.NextDouble();
            x[i] = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
        }

        var nullCond = new float[condMel.Length];
        var nullText = new float[textFeatures.Length];
        float dt = 1.0f / odeSteps;

        for (int step = 0; step < odeSteps; step++)
        {
            float t = (float)step / odeSteps;
            var vCond = FakeForwardVelocity(x, condMel, textFeatures, t, numFrames, inChannels, hiddenDim);
            var vUncond = FakeForwardVelocity(x, nullCond, nullText, t, numFrames, inChannels, hiddenDim);
            for (int i = 0; i < x.Length; i++)
            {
                float v = vUncond[i] + cfgStrength * (vCond[i] - vUncond[i]);
                x[i] += dt * v;
            }
        }
        return x;
    }

    private static float[] FakeForwardVelocity(float[] x, float[] cond, float[] text, float timestep, int numFrames, int inChannels, int hiddenDim)
    {
        var h = new float[numFrames * hiddenDim];
        for (int f = 0; f < numFrames; f++)
        {
            int offX = f * inChannels;
            int offH = f * hiddenDim;
            for (int d = 0; d < hiddenDim; d++)
            {
                int idxX = d % inChannels;
                int idxText = (d + 37) % F5TextEncoder.TextDim;
                float sum = (offX + idxX < x.Length) ? x[offX + idxX] * 0.5f : 0f;
                sum += (offX + idxX < cond.Length) ? cond[offX + idxX] * 0.3f : 0f;
                int offText = f * F5TextEncoder.TextDim;
                sum += (offText + idxText < text.Length) ? text[offText + idxText] * 0.4f : 0f;
                h[offH + d] = sum;
            }
        }

        float tPhase = timestep * MathF.PI;
        var velocity = new float[numFrames * inChannels];
        for (int f = 0; f < numFrames; f++)
        {
            int offH = f * hiddenDim;
            int offV = f * inChannels;
            for (int c = 0; c < inChannels; c++)
            {
                float val = 0f;
                for (int d = 0; d < 8; d++) val += h[offH + c * 8 + d] * 0.125f;
                velocity[offV + c] = val * MathF.Cos(tPhase);
            }
        }
        return velocity;
    }

    private static float[] LoadPcmFromWav(string wavPath)
    {
        try
        {
            var (samples, sr, _) = WavReader.ReadWav(wavPath);
            if (sr != 24000)
                samples = AudioResampler.Resample(samples, sr, 24000);
            return samples;
        }
        catch
        {
            return new float[24000]; // 1s fallback silence
        }
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _vocosWeights?.Dispose();
    }
}
