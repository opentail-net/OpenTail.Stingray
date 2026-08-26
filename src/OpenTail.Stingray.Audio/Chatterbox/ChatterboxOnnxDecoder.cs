using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Native C++ ONNX Runtime accelerator for Chatterbox-Turbo's conditional audio decoder.
/// When conditional_decoder.onnx is present, executes the entire Flow Encoder + UNet + Euler solve + Vocoder
/// in a single fused C++ graph execution, matching the C++ benchmark 1:1.
/// </summary>
public sealed class ChatterboxOnnxDecoder : IDisposable
{
    private readonly InferenceSession? _session;
    private readonly string? _modelPath;

    public bool IsAvailable => _session != null;

    public ChatterboxOnnxDecoder(string? modelPath = null)
    {
        if (string.IsNullOrEmpty(modelPath))
        {
            // Auto-detect conditional_decoder.onnx in standard locations
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

        if (!string.IsNullOrEmpty(modelPath) && File.Exists(modelPath))
        {
            try
            {
                var options = new SessionOptions();
                options.IntraOpNumThreads = Environment.ProcessorCount;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                _session = new InferenceSession(modelPath, options);
                _modelPath = modelPath;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ChatterboxOnnxDecoder] Could not load ONNX model '{modelPath}': {ex.Message}");
                _session = null;
            }
        }
    }

    public float[]? Decode(int[] promptTokens, IReadOnlyList<int> speechTokens, float[] genEmbedding, float[] promptFeat)
    {
        if (_session == null) return null;

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

        // 1. speech_tokens: [1, seq_len]
        var speechTokensTensor = new DenseTensor<long>(fullTokens.ToArray(), [1, fullTokens.Count]);

        // 2. speaker_embeddings: [1, 192]
        var spkEmbedTensor = new DenseTensor<float>(genEmbedding, [1, genEmbedding.Length]);

        // 3. speaker_features: [1, 500, 80]
        int mel = 80;
        int frames = promptFeat.Length / mel;
        var spkFeatTensor = new DenseTensor<float>(promptFeat, [1, frames, mel]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokensTensor),
            NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkEmbedTensor),
            NamedOnnxValue.CreateFromTensor("speaker_features", spkFeatTensor)
        };

        using var results = _session.Run(inputs);
        foreach (var r in results)
        {
            if (r.Value is DenseTensor<float> dt)
            {
                float[] raw = dt.Buffer.ToArray();
                return ApplyPostProcessTrim(raw);
            }
        }

        return null;
    }

    private static float[] ApplyPostProcessTrim(float[] wav)
    {
        // 40ms trim fade matching official inference
        int nTrim = 24000 / 50; // 480 samples = 20ms
        int fadeLen = 2 * nTrim; // 960 samples = 40ms
        if (wav.Length >= fadeLen)
        {
            for (int i = 0; i < nTrim; i++) wav[i] = 0f;
            for (int i = 0; i < nTrim; i++)
            {
                float progress = (float)i / nTrim;
                float w = 0.5f * (1f - MathF.Cos(progress * MathF.PI));
                wav[nTrim + i] *= w;
            }
        }
        return wav;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
