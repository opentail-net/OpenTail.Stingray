using System.Diagnostics;
using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Phase 1b of <c>docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md</c>: the CPU-prefill KV handoff
/// (<c>HybridPrefillHandoff.cs</c>) for the full-offload <see cref="GpuForwardPass"/> (<c>-g -1</c>), which is the default
/// when a model fits. Measured on OLMoE (2026-10-03): full-GPU decode is the fastest path on the development machine
/// (about 28 tok/s against 24 CPU-only and 18-22 hybrid) but its MoE prefill is the slowest (about 30 tok/s, a router
/// round trip per layer per token against 140+ for the CPU batched pass), so the two are combined: prefill on the CPU,
/// K/V handed to the GPU caches, decode on the GPU.
///
/// <para><b>Scope</b>: MoE models only. Dense models keep the GPU's own batched trunk (<c>_canBatchedTrunk</c>); on a fast
/// discrete GPU that trunk would beat a CPU prefill, and nothing here is measured for that case. MoE has no batched GPU
/// trunk at all (<c>ComputeCanBatchedTrunk</c> excludes it), so its sequential per-token prefill is what the CPU pass
/// replaces; the CPU pass is faster than that on any hardware I can reason about until the GPU grouped-expert prefill
/// (Phase 3 of the plan) exists, but only the integrated-GPU case is measured.</para>
///
/// <para><b>Precision</b>: this pass stores KV as F32 or as IEEE fp16 packed two per word, low half first (the
/// "BFloat16" dtype is fp16 in Vulkan). F32 is copied exactly. For fp16 the CPU packs with the same layout the
/// <c>KvAppendBf16</c> shader writes (round-to-nearest-even via <see cref="Half"/>; GLSL leaves <c>packHalf2x16</c>
/// rounding to the implementation, which is round-to-nearest-even on the hardware I know). That is the same narrowing the
/// sequential GPU prefill applies, but it is a conversion, so tests compare the downloaded words with an independent
/// <see cref="Half"/> packing of the CPU cache, and decode agreement is tested end to end. Q8_0 KV is refused.</para>
///
/// <para>All the other rules of the hybrid handoff apply: fresh sequences only, F32 CPU prefill cache, evidence-based family
/// list, memory budget, single owner (see <see cref="HybridForwardPass"/>'s handoff file).</para>
/// </summary>
public sealed unsafe partial class GpuForwardPass
{
    private static readonly string? s_cpuPrefillSetting = Environment.GetEnvironmentVariable("STINGRAY_GPU_CPU_PREFILL");
    private static readonly bool s_prefillTiming = Environment.GetEnvironmentVariable("STINGRAY_PREFILL_TIMING") == "1";

    private ForwardPass? _cpuPrefillPass;
    private OpenTail.Stingray.Cpu.CpuBackend? _cpuPrefillBackend;
    private readonly object _cpuPrefillGate = new();
    private bool _cpuPrefillBroken;
    private string _cpuPrefillBrokenReason = "";

    /// <summary>On by default for MoE; <c>STINGRAY_GPU_CPU_PREFILL=0</c> turns it off (<c>all</c> also lifts the family list).</summary>
    internal bool CpuPrefillEnabled { get; set; } = s_cpuPrefillSetting != "0";

    internal int CpuPrefillMinTokens { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS"), out int m) && m > 0 ? m : 32;

    internal long CpuPrefillKvBudgetBytes { get; set; } =
        (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB"), out int mb) && mb > 0 ? mb : 4096) * 1024L * 1024L;

    /// <summary>The dtype this pass stores KV in (F32 or packed fp16 as "BFloat16"); tests need it to pick the right comparison.</summary>
    internal DType KvDTypeForTest => _kvDType;

    /// <summary>True when the last <see cref="Prefill"/> ran on the CPU and handed its KV over.</summary>
    internal bool LastPrefillUsedCpuHandoff { get; private set; }

    /// <summary>Why the last <see cref="Prefill"/> did not use the handoff, or null when it did (or tried to).</summary>
    internal string? LastCpuPrefillRefusal { get; private set; }

    private string? CpuPrefillRefusal(int n, int startPos)
    {
        if (!CpuPrefillEnabled) return "disabled (STINGRAY_GPU_CPU_PREFILL=0)";
        if (_cpuPrefillBroken) return _cpuPrefillBrokenReason;
        if (startPos != 0) return "startPos != 0 (a CPU pass built here does not hold the earlier turns' K/V)";
        if (n < CpuPrefillMinTokens) return $"prompt shorter than {CpuPrefillMinTokens} tokens";
        if (n > _maxSeqLen) return "prompt longer than the context";
        if (!_isMoE) return "dense model: the GPU batched trunk handles prefill (nothing here is measured for dense on a fast GPU)";
        if (_canBatchedTrunk) return "the GPU batched trunk is available";
        if (_tqEnabled) return "TurboQuant KV is active";
        if (_isGemma4 || _hp.LayerHeadDim is not null) return "per-layer head dimensions";
        if (_residentLayers != _hp.NumLayers) return "layer split: not every layer is GPU resident";
        if (_kvEvicted) return "KV positions were evicted (SnapKV)";
        if (_snapKvEffectiveBudget > 0 && n > _snapKvEffectiveBudget) return "SnapKV would evict after this prefill";
        if (_mropePairPos is not null) return "M-RoPE positions";
        if (_kvDType is not (DType.Float32 or DType.BFloat16)) return $"GPU KV dtype {_kvDType} has no handoff conversion";
        if (PagedKvCache.Bf16RoundingRequested) return "STINGRAY_KV_DTYPE=bf16 rounds every KV write; exact F32 K/V cannot be guaranteed";
        if (KvHandoff.FamilyRefusal(_model, s_cpuPrefillSetting, HandoffPath.VulkanFullGpu) is { } familyRefusal) return familyRefusal;
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

                HandoffKv(cache, n);
                TruncateTo(n); // _kvLength, the host KvCache counter and the M-RoPE image list, exactly as a rewind would
                long t3 = Stopwatch.GetTimestamp();

                LastPrefillUsedCpuHandoff = true;
                if (s_prefillTiming)
                {
                    double construct = Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds;
                    double cpu = Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds;
                    double hand = Stopwatch.GetElapsedTime(t2, t3).TotalMilliseconds;
                    double total = Stopwatch.GetElapsedTime(t0, t3).TotalMilliseconds;
                    Console.Error.WriteLine(
                        $"[gpu-prefill] cpu handoff: {n} tokens; construct {construct:F0} ms (0 once warm); " +
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
        uint[]? kPacked = _kvDType == DType.BFloat16 ? new uint[(long)n * kvDim / 2] : null;
        uint[]? vPacked = _kvDType == DType.BFloat16 ? new uint[(long)n * kvDim / 2] : null;

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            KvHandoff.ExtractRows(source, layer, n, _numKvHeads, _headDim, kHost, vHost);
            if (kPacked is null)
            {
                _gpu.UploadInto(_gpuKCache[layer], kHost);
                _gpu.UploadInto(_gpuVCache[layer], vHost);
            }
            else
            {
                PackHalf(kHost, kPacked);
                PackHalf(vHost, vPacked!);
                // Bit pattern copy: UploadInto moves float-sized words and never interprets them.
                _gpu.UploadInto(_gpuKCache[layer], MemoryMarshal.Cast<uint, float>(kPacked));
                _gpu.UploadInto(_gpuVCache[layer], MemoryMarshal.Cast<uint, float>(vPacked!));
            }
        }
    }

    /// <summary>Two IEEE fp16 per word, element 2w in the low half: the layout <c>KvAppendBf16</c> writes and attention reads.</summary>
    internal static void PackHalf(float[] src, uint[] dst)
    {
        for (int w = 0; w < dst.Length; w++)
        {
            uint lo = BitConverter.HalfToUInt16Bits((Half)src[2 * w]);
            uint hi = BitConverter.HalfToUInt16Bits((Half)src[2 * w + 1]);
            dst[w] = lo | (hi << 16);
        }
    }

    /// <summary>Diagnostic read of K/V rows [0, <paramref name="count"/>) of one layer as floats, row-major <c>[count, kvDim]</c>.</summary>
    internal void ReadKvRows(int layer, int count, float[] k, float[] v)
    {
        int kvDim = _numKvHeads * _headDim;
        int elems = count * kvDim;
        if (_kvDType == DType.Float32)
        {
            _gpu.Download(_gpuKCache[layer], k.AsSpan(0, elems));
            _gpu.Download(_gpuVCache[layer], v.AsSpan(0, elems));
            return;
        }
        var kw = new uint[elems / 2];
        var vw = new uint[elems / 2];
        _gpu.Download(_gpuKCache[layer], MemoryMarshal.Cast<uint, float>(kw.AsSpan()));
        _gpu.Download(_gpuVCache[layer], MemoryMarshal.Cast<uint, float>(vw.AsSpan()));
        for (int w = 0; w < kw.Length; w++)
        {
            k[2 * w] = (float)BitConverter.UInt16BitsToHalf((ushort)(kw[w] & 0xFFFF));
            k[2 * w + 1] = (float)BitConverter.UInt16BitsToHalf((ushort)(kw[w] >> 16));
            v[2 * w] = (float)BitConverter.UInt16BitsToHalf((ushort)(vw[w] & 0xFFFF));
            v[2 * w + 1] = (float)BitConverter.UInt16BitsToHalf((ushort)(vw[w] >> 16));
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
