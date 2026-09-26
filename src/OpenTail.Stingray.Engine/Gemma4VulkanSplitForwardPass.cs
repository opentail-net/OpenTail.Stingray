using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Gemma 4 with only some layers on Vulkan (-g N): layers [0, N) run in a
/// <see cref="GpuForwardPass"/> (its verified Gemma 4 trunk, stopped early via
/// <see cref="GpuForwardPass.Gemma4LayerLimit"/>), the hidden state crosses to the CPU once per
/// token, and a CPU <see cref="ForwardPass"/> runs [N, L) and the output head. Each side owns the KV
/// of its layers and builds its own per-layer (PLE) inputs from the token. As in
/// CudaHybridForwardPass, N may not exceed the first shared-KV source layer, so a KV-shared layer and
/// the layer whose cache it reads are always on the same side.
/// </summary>
public sealed class Gemma4VulkanSplitForwardPass : IForwardPass
{
    private readonly GpuForwardPass _gpuPart;
    private readonly ForwardPass _cpuPart;
    private readonly CpuBackend _cpuBackend;
    private readonly int _split;
    private readonly float[] _hidden;

    public Gemma4VulkanSplitForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, int gpuLayers,
        int maxContextLength = 0)
    {
        if (hp.LayerHeadDim is null) throw new ArgumentException("Not a Gemma 4 model.", nameof(hp));
        int maxSplit = MaxGpuLayers(hp);
        if (gpuLayers <= 0 || gpuLayers > maxSplit)
            throw new NotSupportedException(
                $"Gemma 4 -g {gpuLayers}: a split must keep every shared-KV source layer on the CPU; use -g 1..{maxSplit} or -g -1.");
        _split = gpuLayers;
        _gpuPart = new GpuForwardPass(model, gpu, hp, maxContextLength, gemma4LayerLimit: gpuLayers);
        _cpuBackend = new CpuBackend();
        _cpuPart = new ForwardPass(model, _cpuBackend, hp, maxContextLength: _gpuPart.MaxSeqLen);
        _hidden = new float[hp.EmbeddingDim];
    }

    /// <summary>The largest N for which layers [0, N) own their KV and no CPU layer reads GPU KV.</summary>
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
        _gpuPart.ForwardGemma4Hidden(token, position, _hidden);
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
