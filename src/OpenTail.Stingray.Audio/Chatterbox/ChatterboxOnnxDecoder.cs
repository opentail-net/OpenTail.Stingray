using System;
using System.Collections.Generic;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Native C++ ONNX Runtime accelerator for Chatterbox-Turbo's conditional audio decoder.
/// Delegates execution to <see cref="OnnxModelSession"/> when conditional_decoder.onnx is present.
/// </summary>
public sealed class ChatterboxOnnxDecoder : IChatterboxDecoder, IDisposable
{
    private readonly OnnxModelSession? _session;
    private readonly ChatterboxWeights? _t3Weights;

    public bool IsAvailable => _session?.IsAvailable ?? false;

    public ChatterboxOnnxDecoder(string? modelPath = null, ChatterboxWeights? t3Weights = null)
    {
        _t3Weights = t3Weights;
        if (string.IsNullOrEmpty(modelPath))
        {
            string[] candidates = [
                "examples/Chatterbox-turbo-cpp/models/conditional_decoder.onnx",
                "models/conditional_decoder.onnx",
                "models/chatterbox-turbo-conditional_decoder.onnx"
            ];
            foreach (var candidate in candidates)
            {
                if (File.Exists(candidate))
                {
                    modelPath = candidate;
                    break;
                }
            }
        }

        _session = OnnxModelSession.TryLoad(modelPath);
    }

    /// <summary>
    /// Implements IChatterboxDecoder: executes the ONNX graph using default prompt tokens and speaker features.
    /// </summary>
    public float[] Decode(IReadOnlyList<int> speechTokens, float[] speakerFeatures)
    {
        if (_session == null || !_session.IsAvailable) return [];

        if (_t3Weights?.GenPromptToken is { } promptTokens
            && _t3Weights.GenEmbedding is { } genEmbedding && _t3Weights.GenPromptFeat is { } promptFeat)
        {
            bool diag = Environment.GetEnvironmentVariable("STINGRAY_AUDIO_DIAGNOSTIC_DUMP") == "1";
            var sw = diag ? System.Diagnostics.Stopwatch.StartNew() : null;
            var onnxAudio = Decode(promptTokens, speechTokens, genEmbedding, promptFeat);
            if (onnxAudio != null && onnxAudio.Length > 0)
            {
                if (diag) ChatterboxPipeline.DiagLog($"  S3Gen ONNX Native decoder: {sw!.ElapsedMilliseconds}ms, {onnxAudio.Length} samples");
                return onnxAudio;
            }
        }

        return [];
    }

    public float[]? Decode(int[] promptTokens, IReadOnlyList<int> speechTokens, float[] genEmbedding, float[] promptFeat)
    {
        if (_session == null || !_session.IsAvailable) return null;

        var fullTokens = new List<long>(promptTokens.Length + speechTokens.Count + 3);
        foreach (int t in promptTokens) fullTokens.Add(t);
        foreach (int t in speechTokens)
        {
            if (t != ChatterboxAcousticLm.StartSpeechToken && t != ChatterboxAcousticLm.StopSpeechToken && t < 6561)
                fullTokens.Add(t);
        }
        // Silence tokens (S3GEN_SIL = 4299)
        fullTokens.Add(4299);
        fullTokens.Add(4299);
        fullTokens.Add(4299);

        int mel = 80;
        int frames = promptFeat.Length / mel;

        var rawAudio = _session.RunToFloatArray(
            ("speech_tokens", fullTokens.ToArray(), [1, fullTokens.Count]),
            ("speaker_embeddings", genEmbedding, [1, genEmbedding.Length]),
            ("speaker_features", promptFeat, [1, frames, mel])
        );

        if (rawAudio == null || rawAudio.Length == 0) return null;

        float[] audio = new float[rawAudio.Length];
        for (int i = 0; i < rawAudio.Length; i++)
        {
            audio[i] = Math.Clamp(rawAudio[i], -1.0f, 1.0f);
        }

        return audio;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
