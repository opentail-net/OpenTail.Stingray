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
    private readonly EngineSettings _settings;
    private readonly GpuForwardPass _gpuPart;
    private readonly ForwardPass _cpuPart;
    private readonly CpuBackend _cpuBackend;
    private readonly int _split;
    private readonly ModelHyperparams _hp;
    private readonly float[] _hidden;

    public VulkanLayerSplitForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, int gpuLayers,
        int maxContextLength = 0,
        EngineSettings? settings = null)
    {
        _settings = settings ?? EngineSettings.FromEnvironment();
        InitCpuPrefillHandoff();
        int maxSplit = MaxGpuLayers(hp);
        if (gpuLayers <= 0 || gpuLayers > maxSplit)
            throw new NotSupportedException(
                $"-g {gpuLayers}: a layer split needs 1..{maxSplit} GPU layers (Gemma 4 keeps every shared-KV source layer on the CPU); use -g -1 for all layers.");
        _split = gpuLayers;
        _model = model;
        _hp = hp;
        _gpuPart = new GpuForwardPass(model, gpu, hp, maxContextLength, layerLimit: gpuLayers, settings: _settings);
        _cpuBackend = new CpuBackend();
        _cpuPart = new ForwardPass(model, _cpuBackend, hp, maxContextLength: _gpuPart.MaxSeqLen, settings: _settings);
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

    private readonly GgufModel _model;
    private bool _cpuPrefillBroken;

    /// <summary>Master switch (<c>STINGRAY_HYBRID_CPU_PREFILL=0</c> turns it off); settable so tests can compare both paths.</summary>
    internal bool CpuPrefillEnabled { get; set; }

    internal int CpuPrefillMinTokens { get; set; }

    /// <summary>Largest temporary F32 K/V cache the CPU prefill may hold for the WHOLE model, GPU-resident layers included (their rows are read back into it
    /// for a later turn and written into it by the CPU pass). <c>STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB</c>, default 4 GiB, the same budget as the other handoffs.</summary>
    internal long CpuPrefillKvBudgetBytes { get; set; }

    private void InitCpuPrefillHandoff()
    {
        var p = _settings.Prefill;
        CpuPrefillEnabled = p.HybridCpuPrefill != "0";
        CpuPrefillMinTokens = p.HybridCpuPrefillMinTokens;
        CpuPrefillKvBudgetBytes = p.HybridCpuPrefillKvBudgetMb * 1024L * 1024L;
    }

    /// <summary>True when the last <see cref="Prefill"/> ran on the CPU and handed its KV to the GPU layers.</summary>
    internal bool LastPrefillUsedCpuHandoff { get; private set; }

    /// <summary>Why the last <see cref="Prefill"/> did not use the handoff, or null when it did.</summary>
    internal string? LastCpuPrefillRefusal { get; private set; }

    private string? CpuPrefillRefusal(int n, int startPos)
    {
        if (!CpuPrefillEnabled) return "disabled (STINGRAY_HYBRID_CPU_PREFILL=0)";
        if (_cpuPrefillBroken) return "the CPU pass refused batched prefill";
        if (startPos < 0 || startPos > _gpuPart.KvLength) return $"startPos {startPos} is beyond the {_gpuPart.KvLength} K/V rows the GPU layers hold";
        if (n < CpuPrefillMinTokens) return $"prompt shorter than {CpuPrefillMinTokens} tokens";
        if (startPos + n > MaxSeqLen) return "prompt longer than the context";
        if (_hp.LayerHeadDim is not null) return "per-layer head dimensions";
        long kvBytes = KvHandoff.TemporaryKvBytes(startPos + n, _hp.NumKvHeads, _hp.HeadDim, _hp.NumLayers);
        if (kvBytes > CpuPrefillKvBudgetBytes)
            return $"temporary CPU KV cache for the whole model ({kvBytes >> 20} MiB) exceeds the {CpuPrefillKvBudgetBytes >> 20} MiB budget";
        if (_settings.Kv.RoundBf16 || _settings.Kv.Store != KvStoreMode.Fp32)
            return "bf16 KV store requested: exact F32 K/V cannot be guaranteed";
        return KvHandoff.FamilyRefusal(_model, _settings.Prefill.HybridCpuPrefill, HandoffPath.VulkanLayerSplit);
    }

    /// <summary>
    /// CPU-prefill handoff for the layer split: the CPU pass prefills every layer into its own F32 KV cache (the CPU layers'
    /// cache already, and the source for the GPU layers), then the GPU layers' rows are uploaded. Decode continues split.
    /// Compared with the per-token split prefill (one GPU round trip per token plus the MoE router per layer) this is a batched
    /// CPU prefill; the contract is tested on a real checkpoint before a family is admitted (receipts, per path).
    /// </summary>
    private bool TryPrefillViaCpuHandoff(IReadOnlyList<int> tokens, int startPos, out ReadOnlySpan<float> logits)
    {
        logits = default;
        // Models with the attention output gate (afmoe) have no batched CPU trunk: their CPU Prefill is the per-token trunk, which is exactly the
        // path whose K/V the parity test checks, so it is still a valid (if not faster) source of rows. Everything else needs the batched trunk.
        var capability = _cpuPart.GetBatchedPrefillCapability();
        if (!capability.Available && !_hp.AttentionOutputGate)
        {
            _cpuPrefillBroken = true;
            LastCpuPrefillRefusal = "CPU batched prefill unavailable: " + capability.Detail;
            return false;
        }
        // A later turn: the CPU part already holds rows [0, startPos) for its own layers (decode wrote them); the GPU layers' rows are read
        // back into the same cache so the CPU pass prefills the new tokens against the full history. Whole-model cache, F32 on both sides.
        if (startPos > 0) SeedGpuLayerRows(_cpuPart.KvCacheForHandoff, startPos);
        else _cpuPart.TruncateTo(0);
        ReadOnlySpan<float> cpuLogits;
        try { cpuLogits = _cpuPart.Prefill(tokens, startPos); }
        catch (NotSupportedException ex) { _cpuPrefillBroken = true; LastCpuPrefillRefusal = "CPU pass refused: " + ex.Message; return false; }
        if (_gpuPart.ImportSplitKv(_cpuPart.KvCacheForHandoff, startPos + tokens.Count) is { } why)
        {
            LastCpuPrefillRefusal = "GPU side: " + why;
            _cpuPart.TruncateTo(startPos);
            return false;
        }
        logits = cpuLogits;
        return true;
    }

    private void SeedGpuLayerRows(PagedKvCache cache, int count)
    {
        int kvDim = _hp.NumKvHeads * _hp.HeadDim;
        var k = new float[(long)count * kvDim];
        var v = new float[(long)count * kvDim];
        for (int layer = 0; layer < _split; layer++)
        {
            _gpuPart.ReadKvRows(layer, count, k, v);
            for (int pos = 0; pos < count; pos++)
                cache.AppendAt(layer, pos, k.AsSpan(pos * kvDim, kvDim), v.AsSpan(pos * kvDim, kvDim));
        }
    }

    /// <summary>The dtype the GPU layers store KV in (tests pick the comparison from it).</summary>
    internal DType GpuKvDType => _gpuPart.KvDTypeForTest;

    /// <summary>Diagnostic: K/V rows [0, count) of one GPU-resident layer (tests compare them with the CPU pass).</summary>
    internal void ReadGpuKvRows(int layer, int count, float[] k, float[] v) => _gpuPart.ReadKvRows(layer, count, k, v);

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        LastPrefillUsedCpuHandoff = false;
        LastCpuPrefillRefusal = CpuPrefillRefusal(tokens.Count, startPos);
        if (LastCpuPrefillRefusal is null && TryPrefillViaCpuHandoff(tokens, startPos, out var handed))
        {
            LastPrefillUsedCpuHandoff = true;
            return handed;
        }
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
