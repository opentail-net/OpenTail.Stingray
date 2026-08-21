using System;
using System.IO;

namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// Native C# implementation of the Kokoro-82M PLBERT + AdaIN-ResBlock + iSTFT neural synthesis graph.
/// Supports both pure simulated/procedural evaluation and real GGUF weights via <see cref="KokoroWeights"/>.
/// </summary>
public sealed class KokoroModel : IDisposable
{
    public int HiddenDim { get; } = 512;
    public int NumHeads { get; } = 8;
    public int HeadDim => HiddenDim / NumHeads;
    public int NumLayers { get; } = 3;
    public int HopLength { get; } = 256;
    public int SampleRate { get; } = 24000;

    private readonly KokoroWeights? _weights;

    public KokoroModel(int hiddenDim = 512, int numLayers = 3, KokoroWeights? weights = null)
    {
        HiddenDim = hiddenDim;
        NumLayers = numLayers;
        _weights = weights;
    }

    public static KokoroModel Load(string modelPath, string? voicePath = null)
    {
        var weights = new KokoroWeights(modelPath, voicePath);
        return new KokoroModel(512, 3, weights);
    }

    /// <summary>
    /// Executes the full Kokoro neural graph: PLBERT -> Duration Predictor -> Length Regulator -> AdaIN Decoder -> iSTFT.
    /// </summary>
    public float[] Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<float> styleVector, float speed = 1.0f)
    {
        int numTokens = tokens.Length;
        if (numTokens == 0) return [];

        // If styleVector is empty and we have a loaded voice vector in weights, use it
        if (styleVector.IsEmpty && _weights?.StyleVector is { } loadedStyle)
        {
            styleVector = loadedStyle;
        }

        if (_weights is not null)
            return ForwardReal(_weights, tokens, styleVector, speed);

        return ForwardFake(tokens, styleVector, speed);
    }

    /// <summary>
    /// Real forward pass: calls verified sub-modules against loaded GGUF weights.
    /// Implements model.py KModel.forward exactly.
    /// ref_s = styleVector [256]: s_dec = ref_s[:128], s_pred = ref_s[128:].
    /// </summary>
    private static float[] ForwardReal(KokoroWeights w, ReadOnlySpan<int> tokens, ReadOnlySpan<float> refS, float speed)
    {
        int t = tokens.Length; // number of phoneme tokens (including BOS/EOS added by caller)

        // Split style vector: s_dec = refS[:128], s_pred = refS[128:]
        int styleDim = w.StyleDim; // 128
        float[] sDec  = refS.Length >= styleDim     ? refS[..styleDim].ToArray()     : new float[styleDim];
        float[] sPred = refS.Length >= styleDim * 2 ? refS[styleDim..].ToArray()     : new float[styleDim];

        int[] inputIds = tokens.ToArray();

        // 1. BERT: token embedding + 12x shared ALBERT layer -> [T, 768]
        //    bert_proj -> transpose -> d_en [512, T]
        float[] dEn = KokoroBertEncoder.Forward(w, inputIds);        // [T, 768] row-major
        dEn = KokoroBertEncoder.ProjectToWorkingDim(w, dEn, t);      // [512, T] channel-first

        // 2. Text encoder: embedding -> 3x(Conv1d+LN+LeakyReLU) -> BiLSTM -> [512, T]
        float[] tEn = KokoroTextEncoder.Forward(w, inputIds);        // [512, T] channel-first

        // 3. Duration encoder + pred.lstm + duration_proj -> duration sums [T]
        float[] durEnc  = KokoroProsodyPredictor.EncodeDuration(w, dEn, sPred, t); // [T, 640]
        float[] durSums = KokoroProsodyPredictor.PredictDurations(w, durEnc, t);   // [T]

        // 4. Alignment / length regulator
        int[] predDur = KokoroAlignment.ToPredDur(durSums, speed);
        int[] f2tok   = KokoroAlignment.BuildFrameToTokenMap(predDur);
        int   totalF  = f2tok.Length;

        // en = d.T @ pred_aln_trg  (expand durEnc from token-rate to frame-rate)
        // durEnc is [T, 640] row-major; Expand needs channel-first [640, T]
        float[] dEncCF = TransposeToChannelFirst(durEnc, rows: t, cols: 640);
        float[] en     = KokoroAlignment.Expand(dEncCF, 640, f2tok); // [640, F]

        // asr = t_en @ pred_aln_trg
        float[] asr = KokoroAlignment.Expand(tEn, 512, f2tok);       // [512, F]

        // 5. F0Ntrain: predictor.shared BiLSTM -> F0/N AdainResBlk1d stacks -> [1, 2F]
        var (f0Curve, nCurve) = KokoroProsodyPredictor.F0Ntrain(w, en, totalF, sPred);

        // 6. Decoder: F0_conv/N_conv + encode/decode AdainResBlk1d + Generator -> waveform
        return KokoroDecoder.Forward(w.Decoder, w, asr, f0Curve, nCurve, sDec, t: totalF);
    }

    /// <summary>Transpose [rows, cols] row-major to [cols, rows] channel-first.</summary>
    private static float[] TransposeToChannelFirst(float[] input, int rows, int cols)
    {
        var output = new float[cols * rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                output[c * rows + r] = input[r * cols + c];
        return output;
    }

    private float[] ForwardFake(ReadOnlySpan<int> tokens, ReadOnlySpan<float> styleVector, float speed)
    {
        int numTokens = tokens.Length;
        // 1. PLBERT Phoneme Embedding & Transformer Encoder
        var encoded = new float[numTokens * HiddenDim];
        for (int i = 0; i < numTokens; i++)
        {
            int tid = tokens[i];
            float pos = i * 0.05f;

            for (int d = 0; d < HiddenDim; d++)
            {
                // Sinusoidal + token pseudo-embedding
                float freq = MathF.Exp(-d * (MathF.Log(10000.0f) / HiddenDim));
                float pe = (d % 2 == 0) ? MathF.Sin(pos * freq) : MathF.Cos(pos * freq);
                encoded[i * HiddenDim + d] = 0.1f * MathF.Sin(tid * 13.37f + d * 0.1f) + pe;
            }
        }

        // Apply Transformer / Self-Attention layers
        for (int l = 0; l < NumLayers; l++)
        {
            ApplySelfAttention(encoded, numTokens);
            ApplyFeedForward(encoded, numTokens);
        }

        // 2. Duration Predictor (LSTM / Linear) & Length Regulator
        var durations = new int[numTokens];
        int totalFrames = 0;

        for (int i = 0; i < numTokens; i++)
        {
            int tid = tokens[i];
            // Compute base phoneme duration modulated by style and speed
            float baseDuration = (tid == 0) ? 6.0f : 4.0f + (tid % 5) * 1.2f;
            float styleMod = (styleVector.Length > 0) ? 1.0f + 0.1f * styleVector[i % styleVector.Length] : 1.0f;
            int dur = (int)MathF.Round(baseDuration * styleMod / Math.Max(0.2f, speed));
            dur = Math.Clamp(dur, 1, 20);

            durations[i] = dur;
            totalFrames += dur;
        }

        if (totalFrames == 0) totalFrames = 1;

        // Upsample phoneme latents to acoustic frame representations [totalFrames, HiddenDim]
        var frameLatents = new float[totalFrames * HiddenDim];
        int frameIdx = 0;

        for (int i = 0; i < numTokens; i++)
        {
            int dur = durations[i];
            int tokenOff = i * HiddenDim;

            for (int f = 0; f < dur; f++)
            {
                int outOff = (frameIdx + f) * HiddenDim;
                Array.Copy(encoded, tokenOff, frameLatents, outOff, HiddenDim);
            }

            frameIdx += dur;
        }

        // 3. AdaIN-ResBlocks Decoder
        ApplyAdaInDecoder(frameLatents, totalFrames, styleVector);

        // 4. iSTFT / Waveform Vocoder Synthesis (24000 Hz)
        int totalSamples = totalFrames * HopLength;
        var audio = new float[totalSamples];

        SynthesizeWaveform(frameLatents, totalFrames, audio);

        return audio;
    }

    private void ApplySelfAttention(float[] h, int seqLen)
    {
        var next = new float[h.Length];
        float scale = 1.0f / MathF.Sqrt(HeadDim);

        for (int i = 0; i < seqLen; i++)
        {
            int offI = i * HiddenDim;
            for (int j = 0; j < seqLen; j++)
            {
                int offJ = j * HiddenDim;
                float dot = 0f;
                for (int d = 0; d < 32; d++) dot += h[offI + d] * h[offJ + d];
                dot *= scale;
                float weight = 1.0f / (1.0f + MathF.Exp(-dot));

                for (int d = 0; d < HiddenDim; d++)
                {
                    next[offI + d] += weight * h[offJ + d] * 0.1f;
                }
            }
            for (int d = 0; d < HiddenDim; d++) next[offI + d] += h[offI + d]; // Residual
        }

        Array.Copy(next, h, h.Length);
    }

    private void ApplyFeedForward(float[] h, int seqLen)
    {
        for (int i = 0; i < seqLen * HiddenDim; i++)
        {
            float v = h[i];
            // GeLU activation
            float gelu = 0.5f * v * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (v + 0.044715f * v * v * v)));
            h[i] = v + 0.1f * gelu; // Residual
        }
    }

    private void ApplyAdaInDecoder(float[] latents, int numFrames, ReadOnlySpan<float> style)
    {
        // 4 AdaIN residual blocks modulating frame channels
        for (int stage = 0; stage < 4; stage++)
        {
            float gamma = (style.Length > stage * 2) ? 1.0f + 0.2f * style[stage * 2] : 1.0f;
            float beta = (style.Length > stage * 2 + 1) ? 0.1f * style[stage * 2 + 1] : 0.0f;

            for (int f = 0; f < numFrames; f++)
            {
                int off = f * HiddenDim;
                // Compute frame mean and variance
                float mean = 0f;
                for (int d = 0; d < HiddenDim; d++) mean += latents[off + d];
                mean /= HiddenDim;

                float var = 0f;
                for (int d = 0; d < HiddenDim; d++)
                {
                    float diff = latents[off + d] - mean;
                    var += diff * diff;
                }
                float std = MathF.Sqrt(var / HiddenDim + 1e-5f);

                for (int d = 0; d < HiddenDim; d++)
                {
                    float norm = (latents[off + d] - mean) / std;
                    latents[off + d] = gamma * norm + beta;
                }
            }
        }
    }

    private void SynthesizeWaveform(float[] latents, int numFrames, float[] audio)
    {
        // Overlap-Add inverse STFT synthesis with Hann window
        for (int f = 0; f < numFrames; f++)
        {
            int latentOff = f * HiddenDim;
            int audioBase = f * HopLength;

            // Generate synthetic harmonic frame envelope
            float pitchFund = 150.0f + 50.0f * MathF.Sin(f * 0.15f); // Pitch trajectory
            float energy = 0.5f + 0.5f * MathF.Sin(latents[latentOff]);

            for (int n = 0; n < HopLength && (audioBase + n) < audio.Length; n++)
            {
                float t = (float)(audioBase + n) / SampleRate;
                // Harmonic synthesis: fundamental + 3 harmonics modulated by latent features
                float s = MathF.Sin(2.0f * MathF.PI * pitchFund * t)
                        + 0.5f * MathF.Sin(4.0f * MathF.PI * pitchFund * t)
                        + 0.25f * MathF.Sin(6.0f * MathF.PI * pitchFund * t);

                // Hann window
                float hann = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * n / HopLength));
                float sample = s * energy * hann * 0.35f;

                audio[audioBase + n] += sample;
            }
        }

        // Final peak normalization
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

    public void Dispose()
    {
        _weights?.Dispose();
    }
}
