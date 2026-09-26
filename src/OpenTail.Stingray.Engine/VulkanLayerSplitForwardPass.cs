using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// -g N on Vulkan for any architecture the full <see cref="GpuForwardPass"/> runs: layers [0, N) run
/// in a GpuForwardPass stopped early (<see cref="GpuForwardPass.LayerLimit"/>, only those layers'
/// weights and KV uploaded), the hidden state crosses to the CPU once per token, and a CPU
/// <see cref="ForwardPass"/> runs [N, L) and the output head (ForwardPass.ForwardFromHidden). Each
/// side owns the KV of its layers. Used for Gemma 4 and for the architectures
/// <see cref="HybridForwardPass"/> has no path for (<see cref="GpuForwardPass.PartialOffloadUnsupportedReason"/>).
/// Gemma 4: N may not exceed the first shared-KV source layer, so a KV-shared layer and the layer
/// whose cache it reads are always on the same side (the CudaHybridForwardPass rule).
/// </summary>
public sealed class VulkanLayerSplitForwardPass : IForwardPass
{
    private readonly GpuForwardPass _gpuPart;
    private readonly ForwardPass _cpuPart;
    private readonly CpuBackend _cpuBackend;
    private readonly int _split;
    private readonly float[] _hidden;

    public VulkanLayerSplitForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, int gpuLayers,
        int maxContextLength = 0)
    {
        int maxSplit = MaxGpuLayers(hp);
        if (gpuLayers <= 0 || gpuLayers > maxSplit)
            throw new NotSupportedException(
                $"-g {gpuLayers}: a layer split needs 1..{maxSplit} GPU layers (Gemma 4 keeps every shared-KV source layer on the CPU); use -g -1 for all layers.");
        _split = gpuLayers;
        _gpuPart = new GpuForwardPass(model, gpu, hp, maxContextLength, layerLimit: gpuLayers);
        _cpuBackend = new CpuBackend();
        _cpuPart = new ForwardPass(model, _cpuBackend, hp, maxContextLength: _gpuPart.MaxSeqLen);
        _hidden = new float[hp.EmbeddingDim];
    }

    /// <summary>Largest legal N: at least one CPU layer, and (Gemma 4) no CPU layer reading GPU KV.</summary>
    public static int MaxGpuLayers(ModelHyperparams hp)
    {
        int minSrc = hp.NumLayers;
        if (hp.KvSourceLayer is { } ksl)
            for (int i = 0; i < hp.NumLayers; i++)
                if (ksl[i] >= 0 && ksl[i] < minSrc) minSrc = ksl[i];
        return Math.Min(minSrc, hp.NumLayers - 1);
    }

    public int VocabSize => _cpuPart.VocabSize;
    public int MaxSeqLen => Math.Min(_gpuPart.MaxSeqLen, _cpuPart.MaxSeqLen);
    public int GpuLayers => _split;
    public bool SupportsPartialRewind => _gpuPart.SupportsPartialRewind && _cpuPart.SupportsPartialRewind;

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        _gpuPart.ForwardHidden(token, position, _hidden);
        return _cpuPart.ForwardFromHidden(_hidden, token, position, _split);
    }

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++) last = Forward(tokens[i], startPos + i);
        return last;
    }

    public void TruncateTo(int length)
    {
        _gpuPart.TruncateTo(length);
        _cpuPart.TruncateTo(length);
    }

    public void ResetCache()
    {
        _gpuPart.ResetCache();
        _cpuPart.ResetCache();
    }

    public void Dispose()
    {
        _cpuPart.Dispose();
        _cpuBackend.Dispose();
        _gpuPart.Dispose();
    }
}
