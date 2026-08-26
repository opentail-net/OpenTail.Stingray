using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Conditional neural audio decoder / vocoder for Chatterbox-Turbo TTS.
/// Converts discrete speech tokens and speaker features into 24kHz PCM audio samples. When real
/// S3Gen weights are supplied, runs the full real pipeline: ChatterboxFlowEncoder (tokens -> mel
/// conditioning) -> ChatterboxCfmDecoder (conditioning + noise -> mel-spectrogram, meanflow 2-step
/// Euler solve) -> ChatterboxVocoder (mel -> waveform, HiFTGenerator). Falls back to the original
/// placeholder synthesizer when no weights are available (used only by the no-model test/demo path).
/// </summary>
public sealed class ChatterboxDecoder
{
    public const int SampleRate = 24000;
    public const int HopLength = 256;
    public const int HiddenDim = 512;

    private readonly ChatterboxS3GenWeights? _s3Weights;
    private readonly ChatterboxWeights? _t3Weights; // holds the conds.gen.* default-voice conditioning
    private readonly ChatterboxOnnxDecoder? _onnxDecoder;

    public ChatterboxDecoder(ChatterboxS3GenWeights? s3Weights = null, ChatterboxWeights? t3Weights = null, ChatterboxOnnxDecoder? onnxDecoder = null)
    {
        _s3Weights = s3Weights;
        _t3Weights = t3Weights;
        _onnxDecoder = onnxDecoder ?? new ChatterboxOnnxDecoder();
    }

    /// <summary>
    /// Synthesizes 24kHz audio waveform samples from discrete speech tokens and speaker conditioning.
    /// </summary>
    public float[] Decode(IReadOnlyList<int> speechTokens, float[] speakerFeatures)
    {
        if (speechTokens.Count <= 2) return [];

        if (_t3Weights?.GenPromptToken is { } promptTokens
            && _t3Weights.GenEmbedding is { } genEmbedding && _t3Weights.GenPromptFeat is { } promptFeat)
        {
            // 1. Try Native C++ ONNX Accelerator first if available
            if (_onnxDecoder != null && _onnxDecoder.IsAvailable)
            {
                bool diag = Environment.GetEnvironmentVariable("STINGRAY_AUDIO_DIAGNOSTIC_DUMP") == "1";
                var sw = diag ? System.Diagnostics.Stopwatch.StartNew() : null;
                var onnxAudio = _onnxDecoder.Decode(promptTokens, speechTokens, genEmbedding, promptFeat);
                if (onnxAudio != null && onnxAudio.Length > 0)
                {
                    if (diag) ChatterboxPipeline.DiagLog($"  S3Gen ONNX Native decoder: {sw!.ElapsedMilliseconds}ms, {onnxAudio.Length} samples");
                    return onnxAudio;
                }
            }

            // 2. Pure C# AVX2 S3Gen Neural Decoder fallback
            if (_s3Weights is { } s3w)
            {
                return DecodeReal(s3w, promptTokens, genEmbedding, promptFeat, speechTokens);
            }
        }

        return DecodeFakePlaceholder(speechTokens, speakerFeatures);
    }

    private static float[] DecodeReal(ChatterboxS3GenWeights w, int[] promptTokens, float[] genEmbedding, float[] promptFeat, IReadOnlyList<int> speechTokens)
    {
        var speechTokenList = new List<int>(speechTokens.Count + 3);
        foreach (int t in speechTokens)
        {
            if (t != ChatterboxAcousticLm.StartSpeechToken && t != ChatterboxAcousticLm.StopSpeechToken && t < 6561)
                speechTokenList.Add(t);
        }
        if (speechTokenList.Count == 0) return [];
        // Turbo official inference: append 3 silence tokens (S3GEN_SIL = 4299)
        speechTokenList.Add(4299);
        speechTokenList.Add(4299);
        speechTokenList.Add(4299);
        var speechTokenArray = speechTokenList.ToArray();

        bool diag = Environment.GetEnvironmentVariable("STINGRAY_AUDIO_DIAGNOSTIC_DUMP") == "1";
        var sw = diag ? System.Diagnostics.Stopwatch.StartNew() : null;

        var (mu, totalFrames) = ChatterboxFlowEncoder.Forward(w, promptTokens, speechTokenArray);
        var spkEmbed = ChatterboxFlowEncoder.ProjectSpeakerEmbedding(w, genEmbedding);
        if (diag) ChatterboxPipeline.DiagLog($"  S3Gen flow encoder: {sw!.ElapsedMilliseconds}ms, {totalFrames} mel frames (prompt {promptTokens.Length} + speech {speechTokenArray.Length} tokens)");
        sw?.Restart();

        int mel = w.MelChannels;
        int mel1 = promptFeat.Length / mel;
        var cond = new float[mel * totalFrames];
        for (int c = 0; c < mel; c++)
        {
            for (int ti = 0; ti < mel1; ti++)
            {
                cond[c * totalFrames + ti] = promptFeat[ti * mel + c];
            }
        }

        var rng = Random.Shared;
        var melOut = ChatterboxCfmDecoder.Generate(w, mu, cond, spkEmbed, totalFrames, rng, nSteps: 2);
        if (diag) ChatterboxPipeline.DiagLog($"  S3Gen CFM decoder: {sw!.ElapsedMilliseconds}ms");
        sw?.Restart();

        int mel2 = totalFrames - mel1;
        if (mel2 <= 0) return [];
        var melTail = new float[mel * mel2];
        for (int c = 0; c < mel; c++)
            Array.Copy(melOut, c * totalFrames + mel1, melTail, c * mel2, mel2);

        var wav = ChatterboxVocoder.Generate(w, melTail, mel2, rng);
        if (diag) ChatterboxPipeline.DiagLog($"  S3Gen vocoder: {sw!.ElapsedMilliseconds}ms, {mel2} generated mel frames, {wav.Length} samples");

        // Official 40ms initial trim fade (20ms silence + 20ms cosine fade) to remove prompt spillover
        int nTrim = 24000 / 50; // 480 samples = 20ms
        int fadeLen = 2 * nTrim; // 960 samples = 40ms
        if (wav.Length >= fadeLen)
        {
            for (int i = 0; i < nTrim; i++) wav[i] = 0f;
            for (int i = 0; i < nTrim; i++)
            {
                float angle = MathF.PI * (1f - (float)i / nTrim);
                float factor = (MathF.Cos(angle) + 1f) * 0.5f;
                wav[nTrim + i] *= factor;
            }
        }

        return wav;
    }

    // -----------------------------------------------------------------------
    // Fallback placeholder (used only when no real S3Gen weights are available)
    // -----------------------------------------------------------------------

    private static float[] DecodeFakePlaceholder(IReadOnlyList<int> speechTokens, float[] speakerFeatures)
    {
        var tokens = new List<int>();
        foreach (int t in speechTokens)
        {
            if (t != ChatterboxAcousticLm.StartSpeechToken && t != ChatterboxAcousticLm.StopSpeechToken)
            {
                tokens.Add(t);
            }
        }

        int numTokens = tokens.Count;
        int totalSamples = numTokens * HopLength;
        var audio = new float[totalSamples];

        for (int i = 0; i < numTokens; i++)
        {
            int tid = tokens[i];
            int audioBase = i * HopLength;

            float pitch = 130.0f + 50.0f * MathF.Sin(tid * 0.15f);
            float spkWeight = (speakerFeatures.Length > 0) ? speakerFeatures[i % speakerFeatures.Length] : 0.1f;

            for (int n = 0; n < HopLength && (audioBase + n) < audio.Length; n++)
            {
                float t = (float)(audioBase + n) / SampleRate;

                float s1 = MathF.Sin(2.0f * MathF.PI * pitch * t);
                float s2 = 0.5f * MathF.Sin(4.0f * MathF.PI * pitch * t + spkWeight);
                float s3 = 0.25f * MathF.Sin(6.0f * MathF.PI * pitch * t);

                float hann = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * n / HopLength));
                audio[audioBase + n] += (s1 + s2 + s3) * hann * 0.35f;
            }
        }

        // Peak normalize
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

        return audio;
    }
}
