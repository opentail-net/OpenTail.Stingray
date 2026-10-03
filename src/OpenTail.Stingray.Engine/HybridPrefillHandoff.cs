using System.Diagnostics;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Phase 1 of <c>docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md</c>: a fresh long prompt is prefilled by the
/// CPU <see cref="ForwardPass"/>, whose batched trunk (and, for MoE, grouped-by-expert matmuls) is far faster than
/// the hybrid's token-at-a-time loop (OLMoE, 631 tokens: 141.6 tok/s CPU-only against 25.1 hybrid, 2026-10-03), and the
/// resulting K/V rows are copied into the hybrid's own caches so decode continues on the hybrid unchanged.
///
/// <para><b>What this is not</b>: it does not make the Vulkan path batched. It is a scheduling and cache-transfer
/// optimisation. It changes no KV precision (both sides are FP32), no attention, router or FFN arithmetic, and adds no
/// GPU kernels.</para>
///
/// <para><b>Hard rules</b> (each has a refusal reason in <see cref="CpuPrefillRefusal"/>):
/// fresh sequences only (<c>startPos == 0</c>; a CPU pass built here does not hold the hybrid's earlier K/V);
/// the prefill cache is constructed as F32 explicitly (<c>PagedKvCache</c> would otherwise follow
/// <c>STINGRAY_KV_STORE</c> and could narrow itself to BF16 at 1,024 tokens), and the process-wide BF16 rounding flag
/// (<c>STINGRAY_KV_DTYPE=bf16</c>) refuses the path; only model families with a hybrid-versus-CPU logit parity test
/// are admitted; the temporary CPU KV cache must fit a memory budget.</para>
///
/// <para><b>Memory model</b>: weights are shared by mapping (<see cref="GgufModel"/> owns the memory-mapped tensors);
/// this class's retained <see cref="ForwardPass"/> owns only its scratch. The KV cache of each prefill is temporary and
/// freed after the handoff. <b>Single owner</b>: the retained pass is stateful and not re-entrant, so the fast path is
/// serialised by <c>_cpuPrefillGate</c>; the hybrid itself is single-sequence.</para>
/// </summary>
public sealed unsafe partial class HybridForwardPass
{
    private static readonly string? s_cpuPrefillSetting = Environment.GetEnvironmentVariable("STINGRAY_HYBRID_CPU_PREFILL");
    private static readonly bool s_prefillTiming = Environment.GetEnvironmentVariable("STINGRAY_PREFILL_TIMING") == "1";

    private ForwardPass? _cpuPrefillPass;
    private OpenTail.Stingray.Cpu.CpuBackend? _cpuPrefillBackend;
    private readonly object _cpuPrefillGate = new();
    private bool _cpuPrefillBroken;
    private string _cpuPrefillBrokenReason = "";

    /// <summary>Master switch; <c>STINGRAY_HYBRID_CPU_PREFILL=0</c> turns it off. Settable so tests can compare both paths.</summary>
    internal bool CpuPrefillEnabled { get; set; } = s_cpuPrefillSetting != "0";

    /// <summary>Shortest prompt that takes the handoff (llama.cpp's op-offload gate is 32 too). <c>STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS</c>.</summary>
    internal int CpuPrefillMinTokens { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS"), out int m) && m > 0 ? m : 32;

    /// <summary>Largest temporary CPU KV cache the fast path may allocate. <c>STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB</c>, default 4 GiB.</summary>
    internal long CpuPrefillKvBudgetBytes { get; set; } =
        (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB"), out int mb) && mb > 0 ? mb : 4096) * 1024L * 1024L;

    /// <summary>
    /// Preload the experts the CPU prefill used most into the GPU expert cache after a handoff (default on; MoE models with a
    /// slot cache only). <c>STINGRAY_HYBRID_CPU_PREFILL_WARM=0</c> turns it off.
    /// </summary>
    internal bool CpuPrefillWarmExperts { get; set; } = Environment.GetEnvironmentVariable("STINGRAY_HYBRID_CPU_PREFILL_WARM") != "0";

    /// <summary>True when the last <see cref="Prefill"/> ran on the CPU and handed its KV over.</summary>
    internal bool LastPrefillUsedCpuHandoff { get; private set; }

    /// <summary>Why the last <see cref="Prefill"/> did not use the handoff, or null when it did (or tried to).</summary>
    internal string? LastCpuPrefillRefusal { get; private set; }

    private string? CpuPrefillRefusal(int n, int startPos)
    {
        if (!CpuPrefillEnabled) return "disabled (STINGRAY_HYBRID_CPU_PREFILL=0)";
        if (_cpuPrefillBroken) return _cpuPrefillBrokenReason;
        if (startPos != 0) return "startPos != 0 (a CPU pass built here does not hold the earlier turns' K/V)";
        if (n < CpuPrefillMinTokens) return $"prompt shorter than {CpuPrefillMinTokens} tokens";
        if (n > _maxSeqLen) return "prompt longer than the hybrid context";
        if (_tqEnabled) return "TurboQuant KV is active";
        if (_hp.LayerHeadDim is not null) return "per-layer head dimensions";
        if (PagedKvCache.Bf16RoundingRequested) return "STINGRAY_KV_DTYPE=bf16 rounds every KV write; exact F32 K/V cannot be guaranteed";
        if (_nGpuLayers + _nCpuLayers != _hp.NumLayers) return "layer placement does not cover the model";
        if (KvHandoff.FamilyRefusal(_model, s_cpuPrefillSetting) is { } familyRefusal) return familyRefusal;
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
                // Which GPU-layer experts did this prompt use? The sequential prefill would have loaded them into the GPU expert
                // cache as a side effect; the CPU pass does not touch it, so decode would start cold (measured: first 16 tokens
                // at about 13-15 tok/s against 18-20 after a sequential prefill). Count them while the CPU pass routes.
                int[][]? hot = _expertSlotManager is not null && CpuPrefillWarmExperts ? new int[_nGpuLayers][] : null;
                if (hot is not null)
                    _cpuPrefillPass!.MoeRoutingObserver = (layer, counts) =>
                    {
                        if (layer >= _nGpuLayers) return;
                        var acc = hot[layer] ??= new int[counts.Length];
                        for (int e = 0; e < counts.Length; e++) acc[e] += counts[e];
                    };
                ReadOnlySpan<float> logits;
                try { logits = _cpuPrefillPass!.PrefillWithCache(tokens, cache, 0); }
                finally { _cpuPrefillPass!.MoeRoutingObserver = null; }
                logits[.._logitsBuf.Length].CopyTo(_logitsBuf);
                long t2 = Stopwatch.GetTimestamp();

                HandoffKv(cache, n);
                _kvLength = n;
                long t3 = Stopwatch.GetTimestamp();
                int warmed = hot is null ? 0 : WarmExpertCache(hot);
                long t4 = Stopwatch.GetTimestamp();

                LastPrefillUsedCpuHandoff = true;
                if (s_prefillTiming)
                {
                    double construct = Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds;
                    double cpu = Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds;
                    double hand = Stopwatch.GetElapsedTime(t2, t3).TotalMilliseconds;
                    double warm = Stopwatch.GetElapsedTime(t3, t4).TotalMilliseconds;
                    double total = Stopwatch.GetElapsedTime(t0, t4).TotalMilliseconds;
                    Console.Error.WriteLine(
                        $"[hybrid-prefill] cpu handoff: {n} tokens; construct {construct:F0} ms (0 once warm); " +
                        $"cpu prefill {cpu:F0} ms ({n * 1000.0 / Math.Max(cpu, 1e-3):F1} tok/s); kv handoff {hand:F0} ms; " +
                        $"expert warm-up {warmed} experts {warm:F0} ms; " +
                        $"end-to-end {total:F0} ms ({n * 1000.0 / Math.Max(total, 1e-3):F1} tok/s)");
                }
                return true;
            }
            catch (NotSupportedException ex)
            {
                // The CPU pass refuses this configuration (for example a per-token-only trunk). Remember it so the next
                // prompt does not pay for a second attempt, and let the sequential path rewrite every cache position.
                _cpuPrefillBroken = true;
                _cpuPrefillBrokenReason = "CPU pass refused: " + ex.Message;
                return false;
            }
        }
    }

    /// <summary>
    /// Loads the most-used (layer, expert) pairs of the prompt into the slot cache, hottest first, up to its capacity.
    /// Returns how many were loaded. Failures are not fatal: a missing warm-up only means decode warms the cache itself.
    /// </summary>
    private int WarmExpertCache(int[][] hot)
    {
        var slots = _expertSlotManager!;
        var ranked = new List<(int Layer, int Expert, int Count)>();
        for (int layer = 0; layer < hot.Length; layer++)
            if (hot[layer] is { } counts)
                for (int e = 0; e < counts.Length; e++)
                    if (counts[e] > 0) ranked.Add((layer, e, counts[e]));
        ranked.Sort((a, b) => b.Count.CompareTo(a.Count));

        int loaded = 0;
        foreach (var (layer, expert, _) in ranked)
        {
            if (loaded >= slots.Capacity) break;
            try { slots.Preload(layer, expert); loaded++; }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Console.Error.WriteLine($"[hybrid-prefill] expert warm-up stopped at {loaded}: {ex.Message}");
                break;
            }
        }
        return loaded;
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

    /// <summary>
    /// Copies K/V rows [0, n) of every layer from the CPU prefill cache into the hybrid's caches (row extraction is
    /// shared, see <see cref="KvHandoff"/>). GPU layers are written with one transfer per tensor; CPU layers into the
    /// hybrid's own <see cref="KvCache"/>.
    /// </summary>
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
                _gpu.UploadInto(_gpuKCache[layer], kHost);
                _gpu.UploadInto(_gpuVCache[layer], vHost);
            }
            else
            {
                KvHandoff.StoreRows(_cpuKvCache, layer - _nGpuLayers, n, kvDim, kHost, vHost);
            }
        }

        _cpuKvCache.TruncateTo(n);
        _routePredictor?.Reset();
    }

    /// <summary>
    /// Diagnostic read of the K/V rows [0, <paramref name="count"/>) of one layer from wherever they live (GPU tensor or CPU
    /// cache), as row-major <c>[count, kvDim]</c>. Used by the handoff exactness test; not a hot path.
    /// </summary>
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
