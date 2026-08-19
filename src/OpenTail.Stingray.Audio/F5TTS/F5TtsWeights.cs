using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// Container for F5-TTS Flow-Matching DiT safetensors weights.
/// </summary>
public sealed class F5TtsWeights : IDisposable
{
    public SafetensorsLoader Loader { get; }

    public int NumLayers { get; } = 22;
    public int HiddenDim { get; } = 1024;
    public int NumHeads { get; } = 16;
    public int HeadDim => HiddenDim / NumHeads;
    public int InChannels { get; } = 100;
    public int TextDim { get; } = 512;

    public F5TtsWeights(string safetensorsPath)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"F5-TTS safetensors model file not found: {safetensorsPath}");

        Loader = SafetensorsLoader.Open(safetensorsPath);
    }

    public float[]? TryReadWeight(string name)
    {
        string[] candidates =
        {
            name,
            $"ema_model.{name}",
            $"ema_model.transformer.{name}",
            $"transformer.{name}"
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
