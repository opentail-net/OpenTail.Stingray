using System.Diagnostics;
using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// CUDA counterpart of <c>HybridPrefillHandoff.cs</c> (Phase 1 of
/// <c>docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md</c>): a fresh long prompt is prefilled by the CPU batched
/// <see cref="ForwardPass"/> and its K/V rows are copied into this hybrid's caches. Shared logic (family admission, row
/// extraction, memory estimate) lives in <see cref="KvHandoff"/>.
///
/// <para><b>Status: written for the Vulkan-verified design, NOT RUN. There is no NVIDIA GPU on the development machine,
/// so this path is opt-in (<c>STINGRAY_CUDA_HYBRID_CPU_PREFILL=1</c>) and untested; its tests
/// (<c>CudaHybridCpuPrefillHandoffTests</c>) are written and must be run on CUDA hardware before the default changes.</b>
/// The CUDA hybrid already has its own batched trunk (<see cref="PrefillBatchedTrunk"/>) with host-routed MoE, so whether
/// the CPU pass is faster is a measurement to make on the target machine, not an assumption.</para>
///
/// <para>Differences from the Vulkan path: GPU K/V tensors honour <c>STINGRAY_KV_DTYPE</c> (F32, BF16, Q8_0), so the
/// handoff is admitted only for F32 (anything else would need a conversion that must match the CUDA append kernels bit
/// for bit); per-layer head dimensions and KV head counts (Gemma 4) are refused; the upload goes through
/// <c>UploadRawInto</c> because <c>UploadInto</c> requires the source to fill the whole tensor.</para>
/// </summary>
public sealed unsafe partial class CudaHybridForwardPass
{
    private static readonly string? s_cpuPrefillSetting = Environment.GetEnvironmentVariable("STINGRAY_CUDA_HYBRID_CPU_PREFILL");
    private static readonly bool s_prefillTiming = Environment.GetEnvironmentVariable("STINGRAY_PREFILL_TIMING") == "1";

    private ForwardPass? _cpuPrefillPass;
    private OpenTail.Stingray.Cpu.CpuBackend? _cpuPrefillBackend;
    private readonly object _cpuPrefillGate = new();
    private bool _cpuPrefillBroken;
    private string _cpuPrefillBrokenReason = "";

    /// <summary>Off by default; <c>STINGRAY_CUDA_HYBRID_CPU_PREFILL=1</c> (or <c>all</c>) turns it on. Settable for tests.</summary>
    internal bool CpuPrefillEnabled { get; set; } = s_cpuPrefillSetting is "1" or "all";

    internal int CpuPrefillMinTokens { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS"), out int m) && m > 0 ? m : 32;

    internal long CpuPrefillKvBudgetBytes { get; set; } =
        (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB"), out int mb) && mb > 0 ? mb : 4096) * 1024L * 1024L;

    /// <summary>True when the last <see cref="Prefill"/> ran on the CPU and handed its KV over.</summary>
    internal bool LastPrefillUsedCpuHandoff { get; private set; }

    /// <summary>Why the last <see cref="Prefill"/> did not use the handoff, or null when it did (or tried to).</summary>
    internal string? LastCpuPrefillRefusal { get; private set; }

    private string? CpuPrefillRefusal(int n, int startPos)
    {
        if (!CpuPrefillEnabled) return "disabled (opt in with STINGRAY_CUDA_HYBRID_CPU_PREFILL=1)";
        if (_cpuPrefillBroken) return _cpuPrefillBrokenReason;
        if (startPos != 0) return "startPos != 0 (a CPU pass built here does not hold the earlier turns' K/V)";
        if (n < CpuPrefillMinTokens) return $"prompt shorter than {CpuPrefillMinTokens} tokens";
        if (n > _maxSeqLen) return "prompt longer than the hybrid context";
        if (_tqEnabled) return "TurboQuant KV is active";
        if (_isGemma4Like || _hp.LayerHeadDim is not null || _hp.LayerKvHeads is not null) return "per-layer head dimensions or KV head counts";
        if (_kvDType != DType.Float32) return $"GPU KV dtype {_kvDType}: the handoff would need a conversion that must match the CUDA append kernels exactly";
        if (PagedKvCache.Bf16RoundingRequested) return "STINGRAY_KV_DTYPE=bf16 rounds every KV write; exact F32 K/V cannot be guaranteed";
        if (_nGpuLayers + _nCpuLayers != _hp.NumLayers) return "layer placement does not cover the model";
        if (KvHandoff.FamilyRefusal(_model, s_cpuPrefillSetting, HandoffPath.CudaHybrid) is { } familyRefusal) return familyRefusal;
        long kvBytes = KvHandoff.TemporaryKvBytes(n, _numKvHeads, _headDim, _hp.NumLayers);
        if (kvBytes > CpuPrefillKvBudgetBytes)
            return $"temporary CPU KV cache ({kvBytes >> 20} MiB) exceeds the {CpuPrefillKvBudgetBytes >> 20} MiB budget";
        return null;
    }

    private bool TryPrefillViaCpuHandoff(IReadOnlyList<int> tokens)
    {
        lock (_cpuPrefillGate)
        {
            long t0 = Stopwatch.GetTimestamp();
            try
            {
                if (_cpuPrefillPass is null && !BuildCpuPrefillPass()) return false;
                long t1 = Stopwatch.GetTimestamp();

                int n = tokens.Count;
                using var cache = new PagedKvCache(_hp.NumLayers, _numKvHeads, _headDim,
                    bf16Store: false, autoBf16: false, layerHeadDim: null);
                ReadOnlySpan<float> logits = _cpuPrefillPass!.PrefillWithCache(tokens, cache, 0);
                logits[.._logitsBuf.Length].CopyTo(_logitsBuf);
                long t2 = Stopwatch.GetTimestamp();

                // The sequential path would leave the pass faulted if it threw mid-way; here a failure before this point
                // has touched nothing, and the handoff itself only overwrites positions [0, n).
                HandoffKv(cache, n);
                _kvLength = n;
                _gpuTqCompressedLen = 0;
                long t3 = Stopwatch.GetTimestamp();

                LastPrefillUsedCpuHandoff = true;
                LastPrefillWasBatched = false;
                if (s_prefillTiming)
                {
                    double construct = Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds;
                    double cpu = Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds;
                    double hand = Stopwatch.GetElapsedTime(t2, t3).TotalMilliseconds;
                    double total = Stopwatch.GetElapsedTime(t0, t3).TotalMilliseconds;
                    Console.Error.WriteLine(
                        $"[cuda-hybrid-prefill] cpu handoff: {n} tokens; construct {construct:F0} ms (0 once warm); " +
                        $"cpu prefill {cpu:F0} ms ({n * 1000.0 / Math.Max(cpu, 1e-3):F1} tok/s); kv handoff {hand:F0} ms; " +
                        $"end-to-end {total:F0} ms ({n * 1000.0 / Math.Max(total, 1e-3):F1} tok/s)");
                }
                return true;
            }
            catch (NotSupportedException ex)
            {
                _cpuPrefillBroken = true;
                _cpuPrefillBrokenReason = "CPU pass refused: " + ex.Message;
                return false;
            }
        }
    }

    private bool BuildCpuPrefillPass()
    {
        var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        ForwardPass? pass = null;
        try
        {
            pass = new ForwardPass(_model, backend, _hp, maxContextLength: _maxSeqLen);
            var capability = pass.GetBatchedPrefillCapability();
            if (!capability.Available)
            {
                _cpuPrefillBroken = true;
                _cpuPrefillBrokenReason = "CPU batched prefill unavailable: " + capability.Detail;
                pass.Dispose();
                backend.Dispose();
                return false;
            }
        }
        catch
        {
            pass?.Dispose();
            backend.Dispose();
            throw;
        }
        _cpuPrefillBackend = backend;
        _cpuPrefillPass = pass;
        return true;
    }

    private void HandoffKv(PagedKvCache source, int n)
    {
        int kvDim = _numKvHeads * _headDim;
        float[] kHost = new float[(long)n * kvDim];
        float[] vHost = new float[(long)n * kvDim];

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            KvHandoff.ExtractRows(source, layer, n, _numKvHeads, _headDim, kHost, vHost);
            if (layer < _nGpuLayers)
            {
                _gpu.UploadRawInto(_gpuKCache[layer], MemoryMarshal.AsBytes(kHost.AsSpan()));
                _gpu.UploadRawInto(_gpuVCache[layer], MemoryMarshal.AsBytes(vHost.AsSpan()));
            }
            else
            {
                KvHandoff.StoreRows(_cpuKvCache, layer - _nGpuLayers, n, kvDim, kHost, vHost);
            }
        }

        _cpuKvCache.TruncateTo(n);
    }

    /// <summary>Diagnostic read of K/V rows [0, <paramref name="count"/>) of one layer, row-major <c>[count, kvDim]</c>. Not a hot path.</summary>
    internal void ReadKvRows(int layer, int count, float[] k, float[] v)
    {
        int kvDim = _numKvHeads * _headDim;
        if (layer < _nGpuLayers)
        {
            _gpu.Download(_gpuKCache[layer], k.AsSpan(0, count * kvDim));
            _gpu.Download(_gpuVCache[layer], v.AsSpan(0, count * kvDim));
        }
        else
        {
            KvHandoff.LoadRows(_cpuKvCache, layer - _nGpuLayers, count, kvDim, k, v);
        }
    }

    private void DisposeCpuPrefill()
    {
        lock (_cpuPrefillGate)
        {
            _cpuPrefillPass?.Dispose();
            _cpuPrefillBackend?.Dispose();
            _cpuPrefillPass = null;
            _cpuPrefillBackend = null;
        }
    }
}
