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
    private static readonly bool s_prefillTiming = Environment.GetEnvironmentVariable("STINGRAY_PREFILL_TIMING") == "1";

    private ForwardPass? _cpuPrefillPass;
    private OpenTail.Stingray.Cpu.CpuBackend? _cpuPrefillBackend;
    private readonly object _cpuPrefillGate = new();
    private bool _cpuPrefillBroken;
    private string _cpuPrefillBrokenReason = "";

    /// <summary>Master switch; <c>STINGRAY_HYBRID_CPU_PREFILL=0</c> turns it off. Settable so tests can compare both paths.</summary>
    internal bool CpuPrefillEnabled { get; set; }

    /// <summary>Shortest prompt that takes the handoff (llama.cpp's op-offload gate is 32 too). <c>STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS</c>.</summary>
    internal int CpuPrefillMinTokens { get; set; }

    /// <summary>Largest temporary CPU KV cache the fast path may allocate. <c>STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB</c>, default 4 GiB.</summary>
    internal long CpuPrefillKvBudgetBytes { get; set; }

    /// <summary>
    /// Preload the experts the CPU prefill used most into the GPU expert cache after a handoff (default on; MoE models with a
    /// slot cache only). <c>STINGRAY_HYBRID_CPU_PREFILL_WARM=0</c> turns it off.
    /// </summary>
    internal bool CpuPrefillWarmExperts { get; set; }

    private void InitCpuPrefillHandoff()
    {
        var p = _settings.Prefill;
        CpuPrefillEnabled = p.HybridCpuPrefill != "0";
        CpuPrefillMinTokens = p.HybridCpuPrefillMinTokens;
        CpuPrefillKvBudgetBytes = p.HybridCpuPrefillKvBudgetMb * 1024L * 1024L;
        CpuPrefillWarmExperts = p.HybridCpuPrefillWarmExperts;
    }

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
        if (_settings.Kv.RoundBf16) return "STINGRAY_KV_DTYPE=bf16 rounds every KV write; exact F32 K/V cannot be guaranteed";
        if (_nGpuLayers + _nCpuLayers != _hp.NumLayers) return "layer placement does not cover the model";
        if (KvHandoff.FamilyRefusal(_model, _settings.Prefill.HybridCpuPrefill, HandoffPath.VulkanHybrid) is { } familyRefusal) return familyRefusal;
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
                // Which GPU-layer experts does this prompt use? The sequential prefill loaded them into the GPU expert cache as a
                // side effect; the CPU pass does not touch it, so decode would start cold (measured: 62-70% of lookups missing in
                // the first 16 tokens, decode 14 tok/s against 19). Each layer's routing is known the moment that layer's batched
                // MoE finishes, so its hot experts are queued for upload then, while the CPU is still busy with later layers and
                // the GPU is idle: by the time the prefill returns, the cache is warm and no upload competes with decode.
                var warm = _expertSlotManager is not null && CpuPrefillWarmExperts ? StartExpertWarmup() : null;
                if (warm is not null)
                    _cpuPrefillPass!.MoeRoutingObserver = (layer, counts) => { if (layer < _nGpuLayers) warm.Add(layer, counts); };
                ReadOnlySpan<float> logits;
                try { logits = _cpuPrefillPass!.PrefillWithCache(tokens, cache, 0); }
                finally { _cpuPrefillPass!.MoeRoutingObserver = null; warm?.Complete(); }
                logits[.._logitsBuf.Length].CopyTo(_logitsBuf);
                long t2 = Stopwatch.GetTimestamp();

                HandoffKv(cache, n);
                _kvLength = n;
                long t3 = Stopwatch.GetTimestamp();
                long t4 = Stopwatch.GetTimestamp();

                LastPrefillUsedCpuHandoff = true;
                if (s_prefillTiming)
                {
                    double construct = Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds;
                    double cpu = Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds;
                    double hand = Stopwatch.GetElapsedTime(t2, t3).TotalMilliseconds;
                    double total = Stopwatch.GetElapsedTime(t0, t4).TotalMilliseconds;
                    Console.Error.WriteLine(
                        $"[hybrid-prefill] cpu handoff: {n} tokens; construct {construct:F0} ms (0 once warm); " +
                        $"cpu prefill {cpu:F0} ms ({n * 1000.0 / Math.Max(cpu, 1e-3):F1} tok/s); kv handoff {hand:F0} ms; " +
                        (warm is null ? "" : "expert warm-up overlapped with the prefill; ") +
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

    private ExpertWarmup? _warmup;

    private ExpertWarmup StartExpertWarmup()
    {
        StopExpertWarmup();
        return _warmup = new ExpertWarmup(_expertSlotManager!, Math.Max(1, _expertSlotManager!.Capacity / Math.Max(1, _nGpuLayers)));
    }

    /// <summary>The GPU expert cache the hybrid owns, null for dense models (tests read its profiler).</summary>
    internal ExpertSlotManager? ExpertSlots => _expertSlotManager;

    /// <summary>Waits for the post-handoff expert warm-up to finish (tests only).</summary>
    internal void WaitForExpertWarmup() => _warmup?.Wait();

    private void StopExpertWarmup()
    {
        _warmup?.Stop();
        _warmup = null;
    }

    /// <summary>
    /// Uploads, on one background thread, the hottest experts of each GPU layer as the CPU prefill reports them: per layer the
    /// top <c>perLayer</c> by token count, coldest first so the hottest are the most recently used, each promoted to the
    /// protected segment as it lands (<see cref="ExpertSlotManager.PromoteOnPreload"/>; new entries otherwise land in a small
    /// probationary segment and a bulk preload evicts its own earlier entries). A prompt processed in several chunks reports a
    /// layer several times; experts already resident are skipped by <c>Preload</c>. Failures are not fatal: without a warm-up
    /// decode just warms the cache itself.
    /// </summary>
    private sealed class ExpertWarmup
    {
        private readonly ExpertSlotManager _slots;
        private readonly int _perLayer;
        private readonly System.Collections.Concurrent.BlockingCollection<(int Layer, int Expert)> _queue = new();
        private readonly CancellationTokenSource _cts = new();
        private readonly Task _worker;

        public ExpertWarmup(ExpertSlotManager slots, int perLayer)
        {
            _slots = slots;
            _perLayer = perLayer;
            _slots.PromoteOnPreload = true;
            _worker = Task.Run(Run);
        }

        /// <summary>Queues the hottest experts of one layer (called on the CPU prefill's thread, once per layer per chunk).</summary>
        public void Add(int layer, int[] counts)
        {
            var chosen = Enumerable.Range(0, counts.Length).Where(e => counts[e] > 0)
                .OrderByDescending(e => counts[e]).Take(_perLayer).Reverse();
            foreach (int e in chosen) _queue.Add((layer, e));
        }

        /// <summary>Blocks until every queued upload is done (tests; production decode never waits).</summary>
        public void Wait() { try { _worker.Wait(); } catch { /* reported by the worker */ } }

        /// <summary>No more layers will be reported; the worker drains what is queued and finishes.</summary>
        public void Complete() => _queue.CompleteAdding();

        public void Stop()
        {
            _cts.Cancel();
            _queue.CompleteAdding();
            try { _worker.Wait(TimeSpan.FromSeconds(10)); } catch { /* a faulted warm-up was already reported */ }
        }

        private void Run()
        {
            try
            {
                foreach (var (layer, expert) in _queue.GetConsumingEnumerable(_cts.Token))
                    _slots.Preload(layer, expert);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Console.Error.WriteLine($"[hybrid-prefill] expert warm-up stopped: {ex.Message}");
            }
            finally
            {
                _slots.PromoteOnPreload = false;
            }
        }
    }

    private bool BuildCpuPrefillPass()
    {
        var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        ForwardPass? pass = null;
        try
        {
            pass = new ForwardPass(_model, backend, _hp, maxContextLength: _maxSeqLen, settings: _settings);
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
        StopExpertWarmup();
        lock (_cpuPrefillGate)
        {
            _cpuPrefillPass?.Dispose();
            _cpuPrefillBackend?.Dispose();
            _cpuPrefillPass = null;
            _cpuPrefillBackend = null;
        }
    }
}
