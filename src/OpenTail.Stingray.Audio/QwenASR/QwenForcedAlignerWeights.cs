using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Container for Qwen3-ForcedAligner Safetensors weights (audio tower, projection layers, and alignment head).
/// </summary>
public sealed class QwenForcedAlignerWeights : IDisposable
{
    public SafetensorsLoader Loader { get; }
    public int AudioDim { get; } = 896;
    public int HiddenDim { get; } = 1024;
    public int NumAudioLayers { get; } = 18;

    public QwenForcedAlignerWeights(string safetensorsPath)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"Qwen3-ForcedAligner model file not found: {safetensorsPath}");

        Loader = SafetensorsLoader.Open(safetensorsPath);
    }

    public void Dispose()
    {
        Loader.Dispose();
    }
}
