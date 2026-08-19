using System;
using System.IO;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// Container for NVIDIA NeMo Parakeet FastConformer CTC ASR GGUF weights.
/// </summary>
public sealed class ParakeetWeights : IDisposable
{
    public GgufModel Model { get; }

    public int NumLayers { get; } = 24;
    public int HiddenDim { get; } = 1024;
    public int NumHeads { get; } = 8;
    public int FfDim { get; } = 4096;
    public int VocabSize { get; } = 1024;
    public int BlankTokenId { get; } = 1024;
    public int SubsampleFactor { get; } = 8;

    public float[]? CtcBias { get; }

    public ParakeetWeights(string ggufPath)
    {
        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Parakeet GGUF model file not found: {ggufPath}");

        Model = GgufModel.Open(ggufPath);

        if (Model.Metadata.TryGetValue("canary_ctc.n_layers", out var nl) && nl is int nli)
            NumLayers = nli;
        if (Model.Metadata.TryGetValue("canary_ctc.d_model", out var dm) && dm is int dmi)
            HiddenDim = dmi;
        if (Model.Metadata.TryGetValue("canary_ctc.n_heads", out var nh) && nh is int nhi)
            NumHeads = nhi;
        if (Model.Metadata.TryGetValue("canary_ctc.ff_dim", out var ff) && ff is int ffi)
            FfDim = ffi;
        if (Model.Metadata.TryGetValue("canary_ctc.vocab_size", out var vs) && vs is int vsi)
            VocabSize = vsi;
        if (Model.Metadata.TryGetValue("canary_ctc.blank_id", out var bi) && bi is int bii)
            BlankTokenId = bii;
        if (Model.Metadata.TryGetValue("canary_ctc.subsampling_factor", out var sf) && sf is int sfi)
            SubsampleFactor = sfi;

        if (Model.FindTensor("ctc.bias") is { } biasTensor)
        {
            var data = Model.GetTensorData(biasTensor);
            int count = (int)biasTensor.Dimensions[0];
            CtcBias = new float[count];
            for (int i = 0; i < count && (i * 4 + 4) <= data.Length; i++)
            {
                CtcBias[i] = BitConverter.ToSingle(data.Slice(i * 4, 4));
            }
        }
    }

    public void Dispose()
    {
        Model.Dispose();
    }
}
