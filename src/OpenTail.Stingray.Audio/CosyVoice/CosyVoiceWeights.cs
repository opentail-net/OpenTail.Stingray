using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Container for CosyVoice 2.0 (0.5B) language model safetensors weights.
/// </summary>
public sealed class CosyVoiceWeights : IDisposable
{
    public SafetensorsLoader Loader { get; }

    // Verified directly against models/cosyvoice2_0.5b.safetensors's real tensor shapes and
    // models/cosyvoice2_config.json (a plain Qwen2ForCausalLM config, tie_word_embeddings=true)
    // -- see docs/audio-review-progress.md's CosyVoice section. The previous values here
    // (1024/16/8/2816) didn't match the actual checkpoint at all; CosyVoiceLlmConfig already
    // had the correct numbers, only this file was stale.
    public int NumLayers { get; } = 24;
    public int HiddenDim { get; } = 896;
    public int IntermediateDim { get; } = 4864;
    public int NumHeads { get; } = 14;
    public int NumKvHeads { get; } = 2;
    public int HeadDim => HiddenDim / NumHeads;
    public int VocabSize { get; } = 151936;

    public CosyVoiceWeights(string safetensorsPath)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"CosyVoice safetensors model file not found: {safetensorsPath}");

        Loader = SafetensorsLoader.Open(safetensorsPath);
    }

    public float[]? TryReadWeight(string name)
    {
        string[] candidates =
        {
            name,
            $"model.{name}",
            $"llm.{name}"
        };

        foreach (var cand in candidates)
        {
            if (Loader.Contains(cand))
            {
                return Loader.ReadF32(cand);
            }
        }
        return null;
    }

    public void Dispose()
    {
        Loader.Dispose();
    }
}
