using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Container for Chatterbox-Turbo T3 (Acoustic LM) and S3Gen (Vocoder/Flow Decoder) GGUF weights.
/// </summary>
public sealed class ChatterboxWeights : IDisposable
{
    public GgufModel T3Model { get; }
    public GgufModel? S3GenModel { get; }

    public int NumLayers { get; } = 24;
    public int HiddenDim { get; } = 1024;
    public int NumHeads { get; } = 16;
    public int TextVocabSize { get; } = 50276;
    public int SpeechVocabSize { get; } = 6563;

    public float[]? SpeakerEmbedding { get; }

    public ChatterboxWeights(string t3Path, string? s3GenPath = null)
    {
        if (!File.Exists(t3Path))
            throw new FileNotFoundException($"Chatterbox T3 model file not found: {t3Path}");

        T3Model = GgufModel.Open(t3Path);

        if (T3Model.Metadata.TryGetValue("chatterbox.t3.n_layers", out var nl) && nl is int nli)
            NumLayers = nli;
        if (T3Model.Metadata.TryGetValue("chatterbox.t3.hidden_size", out var hs) && hs is int hsi)
            HiddenDim = hsi;
        if (T3Model.Metadata.TryGetValue("chatterbox.t3.n_heads", out var nh) && nh is int nhi)
            NumHeads = nhi;
        if (T3Model.Metadata.TryGetValue("chatterbox.t3.text_vocab_size", out var tvs) && tvs is int tvsi)
            TextVocabSize = tvsi;
        if (T3Model.Metadata.TryGetValue("chatterbox.t3.speech_vocab_size", out var svs) && svs is int svsi)
            SpeechVocabSize = svsi;

        // Load reference speaker embedding if present in container
        if (T3Model.FindTensor("conds.t3.speaker_emb") is { } spkTensor)
        {
            var data = T3Model.GetTensorData(spkTensor);
            int count = (int)spkTensor.Dimensions[0];
            SpeakerEmbedding = new float[count];
            for (int i = 0; i < count && (i * 4 + 4) <= data.Length; i++)
            {
                SpeakerEmbedding[i] = BitConverter.ToSingle(data.Slice(i * 4, 4));
            }
        }

        if (s3GenPath != null && File.Exists(s3GenPath))
        {
            S3GenModel = GgufModel.Open(s3GenPath);
        }
    }

    public void Dispose()
    {
        S3GenModel?.Dispose();
        T3Model.Dispose();
    }
}
