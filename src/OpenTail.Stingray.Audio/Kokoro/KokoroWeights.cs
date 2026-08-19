using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// Container for Kokoro-82M PLBERT, Duration Predictor, and AdaIN synthesis weights parsed from GGUF.
/// </summary>
public sealed class KokoroWeights : IDisposable
{
    public GgufModel Model { get; }
    public GgufModel? VoiceModel { get; }
    public float[]? StyleVector { get; }

    public KokoroWeights(string modelPath, string? voicePath = null)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Kokoro model file not found: {modelPath}");

        Model = GgufModel.Open(modelPath);

        if (voicePath != null && File.Exists(voicePath))
        {
            VoiceModel = GgufModel.Open(voicePath);
            if (VoiceModel.Tensors.Count > 0)
            {
                var t = VoiceModel.Tensors[0];
                var data = VoiceModel.GetTensorData(t);
                StyleVector = new float[t.Dimensions[0]];
                for (int i = 0; i < StyleVector.Length && (i * 4 + 4) <= data.Length; i++)
                {
                    StyleVector[i] = BitConverter.ToSingle(data.Slice(i * 4, 4));
                }
            }
        }
    }

    public void Dispose()
    {
        VoiceModel?.Dispose();
        Model.Dispose();
    }
}
