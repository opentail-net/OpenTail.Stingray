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

    public int NumLayers { get; } = 24;
    public int HiddenDim { get; } = 1024;
    public int IntermediateDim { get; } = 2816;
    public int NumHeads { get; } = 16;
    public int NumKvHeads { get; } = 8;
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
