namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Native C# implementation of Piper (VITS) neural synthesis graph:
/// Prior Text Encoder + Duration Predictor + Normalizing Flow + HiFi-GAN MRF vocoder.
///
/// When constructed with a real .onnx weights file (see <see cref="PiperModel(string, int)"/>),
/// runs the real weight-driven pipeline (PiperTextEncoder -&gt; PiperDurationPredictor -&gt;
/// PiperLengthRegulator -&gt; PiperFlow -&gt; PiperHifiGanDecoder), each stage independently verified
/// against real onnxruntime golden output (see tests/OpenTail.Stingray.Tests.Audio/Piper*.cs).
/// Otherwise falls back to the original procedural placeholder synthesis (kept so parameterless/
/// no-model-file callers -- e.g. Fast tests -- still get a valid, fast, non-real waveform).
/// </summary>
public sealed class PiperModel : IDisposable
{
    public int HiddenDim { get; }
    public int NumHeads { get; } = 2;
    public int HeadDim => HiddenDim / NumHeads;
    public int NumLayers { get; } = 6;
    public int HopLength { get; } = 256;
    public int SampleRate { get; }

    private readonly PiperOnnxWeights? _weights;

    /// <summary>Fake/placeholder synthesis path (no real weights). Kept for callers without an ONNX model file.</summary>
    public PiperModel(int sampleRate = 22050, int hiddenDim = 192)
    {
        SampleRate = sampleRate;
        HiddenDim = hiddenDim;
        _weights = null;
    }

    /// <summary>Real weight-driven synthesis path, loading Piper's ONNX weights.</summary>
    public PiperModel(string onnxPath, int sampleRate = 22050)
    {
        SampleRate = sampleRate;
        _weights = new PiperOnnxWeights(onnxPath);
        HiddenDim = _weights.HiddenDim;
    }

    /// <summary>
    /// Executes the full VITS forward inference graph.
    /// </summary>
    public float[] Forward(
        ReadOnlySpan<int> tokens,
        int speakerId = 0,
        float noiseScale = 0.667f,
        float lengthScale = 1.0f,
        float noiseW = 0.8f)
    {
        int numTokens = tokens.Length;
        if (numTokens == 0) return [];

        return _weights is null
            ? ForwardFake(tokens, speakerId, noiseScale, lengthScale, noiseW)
            : ForwardReal(_weights, tokens, noiseScale, lengthScale, noiseW);
    }

    private static float[] ForwardReal(PiperOnnxWeights w, ReadOnlySpan<int> tokens, float noiseScale, float lengthScale, float noiseW)
    {
        int numTokens = tokens.Length;
        var tokenArray = tokens.ToArray();

        var (encoderHidden, mu, logs) = PiperTextEncoder.Forward(w, tokenArray);

        var rng = new GaussianRandom();
        float[] sdpNoise = rng.NextArray(2 * numTokens);
        float[] logw = PiperDurationPredictor.Predict(w, encoderHidden, numTokens, sdpNoise, noiseW);

        var durations = new int[numTokens];
        int totalFrames = 0;
        for (int i = 0; i < numTokens; i++)
        {
            int d = (int)MathF.Ceiling(MathF.Exp(logw[i]) * lengthScale);
            if (d < 1) d = 1;
            durations[i] = d;
            totalFrames += d;
        }

        float[] flowNoise = rng.NextArray(w.HiddenDim * totalFrames);
        var (zp, tFrames, _) = PiperLengthRegulator.ExpandWithDurations(mu, logs, w.HiddenDim, numTokens, durations, flowNoise, noiseScale);

        float[] flowOut = PiperFlow.Reverse(w, zp, tFrames);

        return PiperHifiGanDecoder.Forward(w, flowOut, tFrames);
    }

    private float[] ForwardFake(ReadOnlySpan<int> tokens, int speakerId, float noiseScale, float lengthScale, float noiseW)
    {
        int numTokens = tokens.Length;

        // 1. VITS Prior Text Encoder (Relative Positional Transformer)
        var mu = new float[numTokens * HiddenDim];
        var logVariance = new float[numTokens * HiddenDim];

        for (int i = 0; i < numTokens; i++)
        {
            int tid = tokens[i];
            float pos = i * 0.1f;

            for (int d = 0; d < HiddenDim; d++)
            {
                float freq = MathF.Exp(-d * (MathF.Log(10000.0f) / HiddenDim));
                float pe = (d % 2 == 0) ? MathF.Sin(pos * freq) : MathF.Cos(pos * freq);

                // Speaker embedding bias
                float spkEmbed = 0.05f * MathF.Sin(speakerId * 3.14f + d * 0.2f);
                float val = 0.08f * MathF.Sin(tid * 7.77f + d * 0.15f) + pe + spkEmbed;

                mu[i * HiddenDim + d] = val;
                logVariance[i * HiddenDim + d] = -0.5f + 0.1f * MathF.Cos(tid * 3.33f + d * 0.1f);
            }
        }

        // Apply Transformer Self-Attention Layers
        ApplyPriorTransformer(mu, numTokens);

        // 2. Duration Prediction & Length Regulator
        var durations = new int[numTokens];
        int totalFrames = 0;
        var rng = new Random(42 + speakerId);

        for (int i = 0; i < numTokens; i++)
        {
            int tid = tokens[i];
            // Base frame duration per token
            float baseDur = (tid == 0) ? 2.0f : 3.5f + (tid % 4) * 1.0f;
            float noise = (float)(rng.NextDouble() * 2.0 - 1.0) * noiseW * 0.2f;

            int dur = (int)MathF.Round((baseDur + noise) * lengthScale);
            dur = Math.Clamp(dur, 1, 15);

            durations[i] = dur;
            totalFrames += dur;
        }

        if (totalFrames == 0) totalFrames = 1;

        // 3. Length Expansion & Normalizing Flow Inversion
        var latents = new float[totalFrames * HiddenDim];
        int frameOffset = 0;

        for (int i = 0; i < numTokens; i++)
        {
            int dur = durations[i];
            int tokenOff = i * HiddenDim;

            for (int f = 0; f < dur; f++)
            {
                int outOff = (frameOffset + f) * HiddenDim;
                for (int d = 0; d < HiddenDim; d++)
                {
                    float mean = mu[tokenOff + d];
                    float std = MathF.Exp(logVariance[tokenOff + d]);
                    float eps = (float)(rng.NextDouble() * 2.0 - 1.0);

                    // Prior sample z_p
                    float z = mean + std * eps * noiseScale;
                    latents[outOff + d] = z;
                }
            }

            frameOffset += dur;
        }

        // Apply Invertible Normalizing Flow Coupling Layers
        ApplyNormalizingFlow(latents, totalFrames);

        // 4. HiFi-GAN MRF Waveform Synthesis (22050 Hz)
        int totalSamples = totalFrames * HopLength;
        var audio = new float[totalSamples];

        SynthesizeHifiGanWaveform(latents, totalFrames, audio);

        return audio;
    }

    private void ApplyPriorTransformer(float[] h, int seqLen)
    {
        float scale = 1.0f / MathF.Sqrt(HeadDim);
        var temp = new float[h.Length];

        for (int l = 0; l < 2; l++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                int offI = i * HiddenDim;
                for (int j = 0; j < seqLen; j++)
                {
                    int offJ = j * HiddenDim;
                    float dot = 0f;
                    for (int d = 0; d < 32; d++) dot += h[offI + d] * h[offJ + d];
                    dot *= scale;
                    float w = 1.0f / (1.0f + MathF.Exp(-dot));

                    for (int d = 0; d < HiddenDim; d++)
                    {
                        temp[offI + d] += w * h[offJ + d] * 0.1f;
                    }
                }
                for (int d = 0; d < HiddenDim; d++) temp[offI + d] += h[offI + d];
            }
            Array.Copy(temp, h, h.Length);
        }
    }

    private void ApplyNormalizingFlow(float[] latents, int numFrames)
    {
        // 4 affine coupling flow transformations
        for (int layer = 0; layer < 4; layer++)
        {
            int halfDim = HiddenDim / 2;
            for (int f = 0; f < numFrames; f++)
            {
                int off = f * HiddenDim;
                for (int d = 0; d < halfDim; d++)
                {
                    float x0 = latents[off + d];
                    float x1 = latents[off + halfDim + d];

                    float s = 0.1f * MathF.Tanh(x0);
                    float t = 0.05f * x0;

                    // Inverse affine transform: y = (x1 - t) * exp(-s)
                    latents[off + halfDim + d] = (x1 - t) * MathF.Exp(-s);
                }
            }
        }
    }

    private void SynthesizeHifiGanWaveform(float[] latents, int numFrames, float[] audio)
    {
        // 4-stage transposed conv upsampling (strides 8, 8, 2, 2 = 256x)
        for (int f = 0; f < numFrames; f++)
        {
            int latentOff = f * HiddenDim;
            int audioBase = f * HopLength;

            float pitch = 140.0f + 40.0f * MathF.Sin(f * 0.12f);
            float energy = 0.5f + 0.5f * MathF.Cos(latents[latentOff]);

            for (int n = 0; n < HopLength && (audioBase + n) < audio.Length; n++)
            {
                float t = (float)(audioBase + n) / SampleRate;

                // MRF multi-harmonic generator
                float s1 = MathF.Sin(2.0f * MathF.PI * pitch * t);
                float s2 = 0.45f * MathF.Sin(4.0f * MathF.PI * pitch * t);
                float s3 = 0.20f * MathF.Sin(6.0f * MathF.PI * pitch * t);

                float hann = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * n / HopLength));
                float sample = (s1 + s2 + s3) * energy * hann * 0.30f;

                audio[audioBase + n] += sample;
            }
        }

        // Peak normalization
        float maxPeak = 0f;
        for (int i = 0; i < audio.Length; i++)
        {
            float abs = MathF.Abs(audio[i]);
            if (abs > maxPeak) maxPeak = abs;
        }

        if (maxPeak > 0.001f)
        {
            float gain = 0.90f / maxPeak;
            for (int i = 0; i < audio.Length; i++) audio[i] *= gain;
        }
    }

    public void Dispose() { }
}

/// <summary>Box-Muller N(0,1) sampler -- the real Piper ONNX graph's `RandomNormalLike` draws are
/// per-run RNG (not reproducible weights), so production inference just needs valid Gaussian noise,
/// not a specific seed/sequence match (that isolation is covered separately by the golden-noise
/// tests, which feed an exact captured draw instead of using this sampler).</summary>
internal sealed class GaussianRandom
{
    private readonly Random _rng = new();

    public float[] NextArray(int count)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++) result[i] = Next();
        return result;
    }

    private float Next()
    {
        // Box-Muller transform.
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
