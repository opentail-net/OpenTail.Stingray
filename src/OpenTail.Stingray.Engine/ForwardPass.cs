using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.TurboQuant;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Optimized CPU forward pass for a dense LLaMA-family transformer.
/// Uses AVX2 SIMD, fused dequant-matvec, and multi-threading.
/// </summary>
public sealed unsafe class ForwardPass : IForwardPass, IBatchedForwardPass, IPrefixCacheableBatchedForwardPass
{
    // Widened from GgufModel to the tensor-source seam: ForwardPass uses only FindTensor,
    // GetTensorData and GetTensorDataPtr, so a non-GGUF source can feed this unmodified loop.
    private readonly IModelTensorSource _model;
    private readonly ModelHyperparams _hp;
    private readonly PagedKvCache _kvCache;
    private readonly int _ctxLen; // scratch buffer sizing (attnScores, TurboQuant)

    // GGUF control/user-defined token IDs, when the model carries the optional tokenizer type
    // table. An all-control prompt is structurally unlike normal text and is a numerically hostile
    // input for activation Q8 prefill; PrefillDispatch keeps that narrow case on the sequential
    // F32 route. Null means the source did not supply token-type classification.
    private readonly bool[]? _controlTokenIds;

    // Norm weight cache: only tiny F32 weights (2048 floats = 8KB each)
    private readonly Dictionary<string, nint> _normCache = new();

    // Dequant-once BLAS weight cache (issue #189). The batched-prefill BLAS path
    // (SimdKernels.MatMulBatched) re-dequantizes each projection weight to F32 on every
    // call; chunked prompt admission re-walks all layers per chunk, so small chunks re-pay
    // the full dequant N times. We cache the F32 dequant per weight tensor (keyed by name)
    // and reuse it across chunks. Reuse distance is a full model sweep, so the cache only
    // pays off if it can hold *every* batched projection weight — hence _dequantCacheCovers.
    // Populated lazily on the single batcher thread (same no-lock assumption as _normCache).
    private readonly Dictionary<string, nint> _dequantWeightCache = new();

    // Repacked Q4_K weights in the 8-row interleaved block_q4_Kx8 form (perf-loop iteration 42).
    // Separate from the dequant cache above: that one produces F32 for the BLAS path and costs 8x
    // the weight memory, this one stays 4-bit and costs ~5.6% (1216 B per 8 rows per block vs
    // 8 x 144). Populated lazily on first use of a tensor and freed in Dispose.
    private readonly Dictionary<string, nint> _q4kx8Cache = new();
    private long _q4kx8CacheUsedBytes;

    /// <summary>
    /// Ceiling on repacked-weight memory, in bytes. <b>DEFAULT: auto-sized from available memory.</b>
    /// Override with <c>STINGRAY_Q4KX8_CACHE_MB=&lt;megabytes&gt;</c>; <c>0</c> disables the
    /// repacked path entirely.
    ///
    /// <para><b>Was opt-in (default 0) until 2026-08-02.</b> The three reasons recorded for that
    /// are now resolved or superseded:</para>
    /// <list type="bullet">
    /// <item><b>"2.6x isolated but only +14% end-to-end"</b> — that figure was measured while
    /// <c>MatMulBatchedDualCached</c> routed FFN gate+up past the repacked path, starving it of
    /// ~55% of the model's matmul FLOPs. With that precedence fixed the same comparison is
    /// <b>1.80x</b> end-to-end (86.3 -> 155.1 t/s at 931 tokens).</item>
    /// <item><b>The PrefillPackedMulti tolerance failures</b> were caused by the dual-Q8 gate's
    /// <c>N &gt;= MinBatchForQ8Prefill</c> threshold making kernel choice batch-size-dependent, not
    /// by the repacked kernel's summation order as previously diagnosed. All configurations now
    /// pass the full suite.</item>
    /// <item><b>Quality is measured, not assumed.</b> wikitext-2 test split, 8191 scored tokens
    /// through the batched path: baseline PPL 16.0488, repacked+Path 2 <b>16.0484</b> (-0.002%),
    /// with no degradation in the <c>[1024,+)</c> position bucket (16.9116 vs 16.9189).</item>
    /// </list>
    ///
    /// <para><b>The memory cost is real and is why this auto-sizes rather than taking a flat
    /// default.</b> The repacked copy is an ADDITIONAL allocation — 729 MiB against a 1006.7 MiB
    /// model here, ~+72% — and unlike the mmap'd original it is anonymous and not reclaimable. On a
    /// 70B Q4_K_M that would be tens of GB, which can turn "fits" into "does not fit": a worse
    /// failure than being slow.</para>
    ///
    /// <para>Auto-sizing is safe because <see cref="GetRepackedQ4Kx8"/> declines individual tensors
    /// once the budget is exhausted and the caller falls back to the row-major path. Partial
    /// repacking therefore costs speed, never correctness — so an under-estimate degrades
    /// gracefully rather than failing.</para>
    /// </summary>
    private readonly long _q4kx8CacheBudgetBytes = ResolveQ4Kx8CacheBudget();

    /// <summary>
    /// Auto-sizes the repacked-weight budget from memory the runtime reports as available,
    /// honouring an explicit <c>STINGRAY_Q4KX8_CACHE_MB</c> override (including <c>0</c> to
    /// disable). Uses <see cref="GCMemoryInfo.TotalAvailableMemoryBytes"/>, which respects cgroup
    /// and job-object limits, so a container sees its own ceiling rather than the host's.
    /// </summary>
    private static long ResolveQ4Kx8CacheBudget()
    {
        string? raw = Environment.GetEnvironmentVariable("STINGRAY_Q4KX8_CACHE_MB");
        if (long.TryParse(raw, out long mb)) return mb > 0 ? mb * 1024 * 1024 : 0;

        // A quarter of available memory. Deliberately conservative: the repacked copy sits
        // alongside the mmap'd weights AND the KV cache, and this runs before either is sized, so
        // the budget cannot account for them. A quarter comfortably covers small and mid models
        // outright while capping the damage on large ones, where partial repacking still captures
        // most of the win if the largest tensors are taken first.
        long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        return available > 0 ? available / 4 : 0;
    }
    private readonly bool _dequantCacheEnabled;
    private readonly bool _dequantCacheCovers;     // budget holds the whole model
    private readonly long _dequantCacheBudgetBytes; // 0 = off, long.MaxValue = unlimited
    private long _dequantCacheUsedBytes;

    // Preallocated scratch buffers
    private readonly float* _hidden;     // [embDim]
    private readonly float* _residual;   // [embDim]
    private readonly float* _normBuf;    // [embDim]
    private readonly float* _q;          // [numHeads * headDim]
    private readonly float* _k;          // [numKvHeads * headDim]
    private readonly float* _v;          // [numKvHeads * headDim]
    private readonly float* _attnOut;    // [numHeads * headDim]
    private readonly float* _ffnGate;    // [intermDim]
    private readonly float* _ffnUp;      // [intermDim]
    private readonly float* _logits;     // [vocabSize]
    private readonly float* _attnScores; // [numHeads * ctxLen] per-head score scratch

    private readonly int _embDim;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headsPerKvGroup;
    private readonly int _intermDim;

    // Per-layer head-dim variance (Gemma 4: SWA layers use 256, global layers use 512).
    // _layerHeadDim is non-null only when hp.LayerHeadDim is non-null; otherwise the
    // hot path uses the scalar _headDim. _maxHeadDim sizes scratch + the PagedKvCache
    // so per-layer slots fit; per-layer Attention reads/writes the leading
    // head_dim[layer] of each head, with the trailing bytes left as zeros via an
    // explicit Clear before the Q/K/V matvecs.
    private readonly int[]? _layerHeadDim;
    private readonly int[]? _layerRopeDim;
    private readonly int[]? _layerKvSrc;
    private readonly bool[]? _isSwaLayer;
    private readonly int _maxHeadDim;

    // Precomputed tensor metadata for hot-path access
    private readonly TensorRef _embTensor;
    private readonly TensorRef[] _attnNorm;
    private readonly TensorRef[] _wq, _wk, _wv, _wo;
    private readonly TensorRef[] _ffnNorm;
    private readonly TensorRef[] _wGate, _wUp, _wDown;
    private readonly TensorRef _outputNorm;
    private readonly TensorRef _outputWeight;

    // Optional attention biases (Qwen models)
    private readonly bool _hasAttnBias;
    private readonly bool _hasAttnOutputBias;
    private readonly float*[] _bq, _bk, _bv, _bo;

    // Optional per-head Q/K RMSNorm (Qwen3-style shared weights of size headDim,
    // or OLMoE-style per-channel weights of size numHeads*headDim / numKvHeads*headDim).
    private readonly bool _hasQkNorm;
    private readonly bool _perChannelQkNorm;
    private readonly float*[] _qNorm, _kNorm;

    // Wide-vector (AVX-512) RmsNorm fast path. Enabled only when the model has
    // per-layer head_dim (Gemma 4) — for other models the byte-parity oracles
    // (e.g. MtpDecoder_GreedyParity_LlamaCpp on Qwen3.6-27B-MTP) are sensitive
    // to the ~ULP reduction-order shift between the AVX2 and AVX-512 sum-of-
    // squares paths. See feedback_qkv_matvecdual_breaks_mtp_parity.
    private readonly bool _useWideNorms;

    // MoE (Mixture of Experts) — Phase 5
    private readonly TensorRef[]? _wGateInp;      // router weights [numExperts, embDim] per layer
    private readonly TensorRef[]? _wGateShexp, _wUpShexp, _wDownShexp; // shared expert per layer
    private readonly TensorRef[]? _wGateExps, _wUpExps, _wDownExps;   // packed expert weights per layer
    private readonly float* _routerLogits;  // [numExperts] scratch
    private readonly float* _sharedOut;     // [embDim] shared expert output
    private readonly float* _expertGate;    // [expertIntermDim] expert gate scratch
    private readonly float* _expertUp;      // [expertIntermDim] expert up scratch
    // Per-expert down-projection scratch — sized embDim because that's the row count
    // of the down MatVec. Most MoE models have intermDim >= embDim so _ffnUp would
    // suffice, but OLMoE has embDim=2048 / intermDim=1024 and overflows it.
    private readonly float* _moeDownTemp;

    // ── Batched MoE prefill scratch (docs/cpu-architecture-kernel-opportunities.md item 6) ──
    // Routing is per token, so a batched MoE FFN cannot simply widen the dense GEMMs: token 0
    // may pick experts {3,17} while token 1 picks {3,42}. These buffers hold the CSR bucketing
    // that turns "N tokens x k experts each" into "for each expert, one contiguous batch of the
    // tokens that chose it" — which is what lets each expert's weight rows be read ONCE per
    // batch instead of once per token, the same amortisation the dense batched path gets.
    //
    // A single expert can be chosen by at most one slot of each token (SelectTopK never repeats
    // an index), so no bucket exceeds N rows — that, not N*numActive, is what the per-expert
    // buffers below are sized for.
    private int    _moeBatchCap;      // token rows the buffers below are sized for (0 = unallocated)
    private float* _moeBatchRouter;   // [cap x numExperts]      per-token router probabilities
    private int*   _moeBatchSel;      // [cap x numActive]       selected expert ids
    private float* _moeBatchWts;      // [cap x numActive]       selection weights
    private int*   _moeExpStart;      // [numExperts + 1]        CSR bucket offsets
    private int*   _moeExpCursor;     // [numExperts]            fill cursors
    private int*   _moeExpTokI;       // [cap x numActive]       token index, grouped by expert
    private int*   _moeExpTokK;       // [cap x numActive]       top-k slot,   grouped by expert
    private float* _moeBatchGathered; // [cap x embDim]          one expert's input rows, contiguous;
                                      //   reused for that expert's down output once the gather
                                      //   has been consumed by the gate/up GEMMs
    private float* _moeBatchGate;     // [cap x expertDim]
    private float* _moeBatchUp;       // [cap x expertDim]
    private float* _moeBatchDown;     // [cap x numActive x embDim]  UNWEIGHTED down partials,
                                      //   indexed by (token, slot) so the final reduce can run in
                                      //   top-k order — see MoeFfnBatched's phase 4.

    // Optional TurboQuant KV cache (Phase 3)
    private TurboQuantKvCache? _tqKvCache;
    private float* _rotatedQuery;  // scratch for WHT-rotated query [headDim]
    private float* _decompBuf;     // scratch for decompressed TQ value [headDim]

    // SnapKV (issue #51) prefill-time KV eviction. Activated via the
    // STINGRAY_SNAPKV_BUDGET env var; the selector is lazily allocated on the
    // first prefill long enough to require eviction.
    private readonly SnapKvConfig _snapKvCfg;
    private SnapKvSelector? _snapKv;

    // Hidden-state taps (DSpark / EAGLE-3-style draft conditioning, PR #413 spec).
    // Non-null once EnableHiddenTaps has run; see HiddenTapBuffer for layout.
    private HiddenTapBuffer? _taps;

    // Diagnostic: per-layer residual L2-norm trace (env: STINGRAY_TRACE_NORMS=1).
    private static readonly bool _traceNorms =
        Environment.GetEnvironmentVariable("STINGRAY_TRACE_NORMS") == "1";
    private float[]? _normTraceAttn;   // [numLayers] post-attn-residual L2 norm
    private float[]? _normTraceFfn;    // [numLayers] post-ffn-residual L2 norm
    // Optional MoE router probe (env: STINGRAY_TRACE_ROUTERS=1 dumps top-k experts for every
    // MoE layer). To restrict to a single position (large MoE models), set STINGRAY_TRACE_POS=<n>.
    private static readonly bool _traceRouters =
        Environment.GetEnvironmentVariable("STINGRAY_TRACE_ROUTERS") == "1";
    private static readonly int _traceRouterPos = ParseInt("STINGRAY_TRACE_POS", -1);
    private static int ParseInt(string env, int def)
    {
        var s = Environment.GetEnvironmentVariable(env);
        return int.TryParse(s, System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : def;
    }
    // Tracks the position of the in-flight forward pass so MoeFfn can decide whether to log.
    private int _currentPos;

    // Precomputed RoPE cos/sin tables [maxSeqLen * halfDim]
    private readonly float* _ropeCosTable;
    private readonly float* _ropeSinTable;
    private readonly int _ropeHalfDim;

    // Second RoPE table for Gemma 4 SWA layers (theta = RopeThetaSwa, e.g. 10K).
    // Non-null only when hp.RopeThetaSwa > 0. Halfdim follows the SWA layers' rope dim.
    private readonly float* _ropeCosTableSwa;
    private readonly float* _ropeSinTableSwa;
    private readonly int _ropeHalfDimSwa;

    // Gemma 4 per-layer norms + scale (null/empty on non-Gemma 4 models).
    private readonly TensorRef[]? _postAttnNorm;
    private readonly TensorRef[]? _postFfwNorm;
    private readonly float[]? _layerOutputScale;

    // Gemma 4 Per-Layer-Embedding (PLE) injection. Non-null only when
    // _hp.HasPerLayerTokenEmbd is true. _pleTokenEmbed stays mmap-resident
    // (~3 GB Q8_0). _perLayerModelProj is preloaded BF16→F32 once.
    private readonly TensorRef _pleTokenEmbed;
    private readonly float* _perLayerModelProj;
    private readonly TensorRef _perLayerProjNormTensor;
    private readonly TensorRef[]? _pleInpGate;
    private readonly TensorRef[]? _plePostProj;
    private readonly TensorRef[]? _plePostNorm;
    private readonly int _pleWidth;
    private readonly float* _pleRowBuf;       // [NumLayers * PleWidth] gathered+dequant per token
    private readonly float* _projPerLayer;    // [NumLayers * PleWidth] proj_layer cache per token
    private readonly float* _pleX;            // [PleWidth] inner scratch
    private readonly float* _pleY;            // [embDim] inner scratch

    public ForwardPass(IModelTensorSource model, IComputeBackend backend, ModelHyperparams hp,
        int maxContextLength = 0, long prefillDequantCacheBytes = long.MinValue)
    {
        _model = model;
        _hp = hp;
        if (model.Metadata.TryGetValue("tokenizer.ggml.token_type", out object? tokenTypesObj)
            && tokenTypesObj is object[] tokenTypes)
        {
            _controlTokenIds = new bool[tokenTypes.Length];
            for (int i = 0; i < tokenTypes.Length; i++)
            {
                int type = Convert.ToInt32(tokenTypes[i], System.Globalization.CultureInfo.InvariantCulture);
                _controlTokenIds[i] = type is TokenizerSource.ControlTokenType or TokenizerSource.UserDefinedTokenType;
            }
        }
        // ctxLen only governs scratch buffer sizes; PagedKvCache allocates pages lazily.
        int ctxLen = maxContextLength > 0
            ? Math.Min(maxContextLength, hp.ContextLength)
            : Math.Min(hp.ContextLength, 32768);
        _ctxLen = ctxLen;
        _snapKvCfg = SnapKvConfig.FromEnvironment();

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;

        // Per-layer head-dim plumbing. Materialise the per-layer arrays once so the
        // hot loop reads from a plain int[] rather than IReadOnlyList<int>.
        if (hp.LayerHeadDim is { } lhd)
        {
            _layerHeadDim = new int[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) _layerHeadDim[i] = lhd[i];
        }
        if (hp.LayerRopeDim is { } lrd)
        {
            _layerRopeDim = new int[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) _layerRopeDim[i] = lrd[i];
        }
        if (hp.KvSourceLayer is { } ksl)
        {
            _layerKvSrc = new int[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) _layerKvSrc[i] = ksl[i];
        }
        if (hp.IsSwaLayer is { } swa)
        {
            _isSwaLayer = new bool[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) _isSwaLayer[i] = swa[i];
        }

        _maxHeadDim = _headDim;
        int maxRopeDim = hp.RopeDim > 0 ? hp.RopeDim : _headDim;
        if (_layerHeadDim is not null)
            for (int i = 0; i < hp.NumLayers; i++)
                if (_layerHeadDim[i] > _maxHeadDim) _maxHeadDim = _layerHeadDim[i];

        // Gemma 4 (per-layer head_dim) gets the wider AVX-512 RmsNorm path because
        // its parity test is internal CPU-vs-CUDA argmax, which is invariant to the
        // ~ULP reduction-order difference. Other architectures stick to AVX2.
        _useWideNorms = _layerHeadDim is not null && Avx512F.IsSupported;
        if (_layerRopeDim is not null)
            for (int i = 0; i < hp.NumLayers; i++)
                if (_layerRopeDim[i] > maxRopeDim) maxRopeDim = _layerRopeDim[i];

        // The dense CPU pass is the only forward pass with BF16 KV readers, so it is the only one
        // that may honour STINGRAY_KV_STORE. See PagedKvCache.Bf16StoreRequested.
        // Pages are sized at _maxHeadDim, but a per-layer-head_dim model (Gemma 4: 256 on SWA
        // layers, 512 on global ones) writes each layer's V heads at THAT layer's stride — the
        // projections pack them contiguously at layerHd. Without this the cache scattered V at the
        // max stride while the K path read at the per-layer one, so every KV head above the first
        // landed in unwritten memory (Gemma 4 layer 0: q heads 4-7 read zeros).
        _kvCache = new PagedKvCache(hp.NumLayers, hp.NumKvHeads, _maxHeadDim,
            bf16Store: PagedKvCache.Bf16StoreRequested,
            autoBf16: PagedKvCache.Bf16AutoRequested,
            layerHeadDim: _layerHeadDim);

        // Allocate scratch
        _hidden = Alloc(_embDim);
        _residual = Alloc(_embDim);
        _normBuf = Alloc(_embDim);
        _q = Alloc(_numHeads * _maxHeadDim);
        _k = Alloc(_numKvHeads * _maxHeadDim);
        _v = Alloc(_numKvHeads * _maxHeadDim);
        _attnOut = Alloc(_numHeads * _maxHeadDim);
        _ffnGate = Alloc(_intermDim);
        _ffnUp = Alloc(_intermDim);
        _logits = Alloc(hp.VocabSize);
        _attnScores = Alloc(_numHeads * ctxLen);

        if (_traceNorms)
        {
            _normTraceAttn = new float[hp.NumLayers];
            _normTraceFfn = new float[hp.NumLayers];
        }

        // Precompute RoPE cos/sin tables for all positions [0, ctxLen).
        // For Gemma 4 the global table is sized for the largest rope dim across layers;
        // the SWA table is built separately at RopeThetaSwa with its own (smaller) halfDim.
        _ropeHalfDim = maxRopeDim / 2;
        _ropeCosTable = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDim * sizeof(float)));
        _ropeSinTable = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDim * sizeof(float)));

        // Gemma 4 stores a top-level `rope_freqs.weight` (size halfDim) that masks
        // the global-layer RoPE frequencies: first 64 pairs = 1.0 (rotate), last
        // 192 = 1e30 (identity). Mirrors llama.cpp gemma4.cpp:191 which passes
        // `rope_freqs` only for non-SWA layers. The SWA table is built unscaled.
        float* globalFreqFactors = null;
        var ropeFreqsInfo = model.FindTensor("rope_freqs.weight");
        float[]? ropeFreqsBuf = null;
        if (ropeFreqsInfo is GgufTensorInfo rfi && rfi.DType == DType.Float32 && rfi.ElementCount == _ropeHalfDim)
        {
            var src = MemoryMarshal.Cast<byte, float>(model.GetTensorData(rfi));
            ropeFreqsBuf = new float[_ropeHalfDim];
            src.Slice(0, _ropeHalfDim).CopyTo(ropeFreqsBuf);
        }
        fixed (float* p = ropeFreqsBuf)
        {
            globalFreqFactors = p;
            SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable, ctxLen, maxRopeDim, hp.RopeTheta, globalFreqFactors);
        }

        if (hp.RopeThetaSwa > 0f && _layerRopeDim is not null)
        {
            int swaRopeDim = maxRopeDim;
            for (int i = 0; i < hp.NumLayers; i++)
                if (_isSwaLayer![i]) { swaRopeDim = _layerRopeDim[i]; break; }
            _ropeHalfDimSwa = swaRopeDim / 2;
            _ropeCosTableSwa = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDimSwa * sizeof(float)));
            _ropeSinTableSwa = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDimSwa * sizeof(float)));
            SimdKernels.BuildRopeTable(_ropeCosTableSwa, _ropeSinTableSwa, ctxLen, swaRopeDim, hp.RopeThetaSwa);
        }

        // Pre-resolve all tensor references (avoids dictionary lookups in hot loop)
        _embTensor = ResolveTensor("token_embd.weight");

        int L = hp.NumLayers;
        _attnNorm = new TensorRef[L];
        _wq = new TensorRef[L]; _wk = new TensorRef[L];
        _wv = new TensorRef[L]; _wo = new TensorRef[L];
        _ffnNorm = new TensorRef[L];
        _wGate = new TensorRef[L]; _wUp = new TensorRef[L]; _wDown = new TensorRef[L];

        _hasAttnBias = hp.HasAttnBias;
        _hasAttnOutputBias = hp.HasAttnOutputBias;
        _bq = new float*[L]; _bk = new float*[L];
        _bv = new float*[L]; _bo = new float*[L];

        _hasQkNorm = hp.HasQkNorm;
        _perChannelQkNorm = hp.IsPerChannelQkNorm;
        _qNorm = new float*[L]; _kNorm = new float*[L];

        // MoE weight arrays
        if (hp.IsMoE)
        {
            _wGateInp = new TensorRef[L];
            _wGateExps = new TensorRef[L]; _wUpExps = new TensorRef[L]; _wDownExps = new TensorRef[L];
            if (hp.HasSharedExpert)
            {
                _wGateShexp = new TensorRef[L]; _wUpShexp = new TensorRef[L]; _wDownShexp = new TensorRef[L];
            }
            _routerLogits = Alloc(hp.NumExperts);
            _sharedOut = Alloc(_embDim);
            _expertGate = Alloc(hp.ExpertIntermediateDim);
            _expertUp = Alloc(hp.ExpertIntermediateDim);
            _moeDownTemp = Alloc(_embDim);
        }

        if (hp.HasPostAttnNorm) _postAttnNorm = new TensorRef[L];
        if (hp.HasPostFfwNorm) _postFfwNorm = new TensorRef[L];
        if (hp.HasLayerOutputScale) _layerOutputScale = new float[L];

        for (int i = 0; i < L; i++)
        {
            int layerHd = _layerHeadDim?[i] ?? _headDim;
            bool kvShared = _layerKvSrc is not null && _layerKvSrc[i] >= 0;
            // Gemma 4 12B global layers carry no attn_v (attention_k_eq_v): V reuses
            // the K projection, so the tensor is genuinely absent.
            bool kEqVLayer = _model.FindTensor($"blk.{i}.attn_v.weight") is null
                          && _model.FindTensor($"blk.{i}.attn_k.weight") is not null;

            _attnNorm[i] = ResolveTensor($"blk.{i}.attn_norm.weight");
            _wq[i] = ResolveTensor($"blk.{i}.attn_q.weight");
            _wo[i] = ResolveTensor($"blk.{i}.attn_output.weight");
            _ffnNorm[i] = ResolveTensor($"blk.{i}.ffn_norm.weight");

            // KV-share layers (Gemma 4 tail) don't carry their own attn_k/attn_v weights —
            // they alias the source layer's K/V pages. Skip the tensor lookup so missing-
            // tensor errors don't fire; the runtime path also skips these projections.
            if (!kvShared)
            {
                _wk[i] = ResolveTensor($"blk.{i}.attn_k.weight");
                // k_eq_v global layers have no attn_v; the runtime reuses K as V.
                if (!kEqVLayer)
                    _wv[i] = ResolveTensor($"blk.{i}.attn_v.weight");
            }

            if (hp.IsMoE)
            {
                _wGateInp![i] = ResolveTensor($"blk.{i}.ffn_gate_inp.weight");
                _wGateExps![i] = ResolveTensor($"blk.{i}.ffn_gate_exps.weight");
                _wUpExps![i] = ResolveTensor($"blk.{i}.ffn_up_exps.weight");
                _wDownExps![i] = ResolveTensor($"blk.{i}.ffn_down_exps.weight");
                if (hp.HasSharedExpert)
                {
                    _wGateShexp![i] = ResolveTensor($"blk.{i}.ffn_gate_shexp.weight");
                    _wUpShexp![i] = ResolveTensor($"blk.{i}.ffn_up_shexp.weight");
                    _wDownShexp![i] = ResolveTensor($"blk.{i}.ffn_down_shexp.weight");
                }
            }
            else
            {
                _wGate[i] = ResolveTensor($"blk.{i}.ffn_gate.weight");
                _wUp[i] = ResolveTensor($"blk.{i}.ffn_up.weight");
                _wDown[i] = ResolveTensor($"blk.{i}.ffn_down.weight");
            }

            if (_hasAttnBias)
            {
                _bq[i] = LoadBias($"blk.{i}.attn_q.bias", _numHeads * layerHd);
                if (!kvShared)
                {
                    _bk[i] = LoadBias($"blk.{i}.attn_k.bias", _numKvHeads * layerHd);
                    _bv[i] = LoadBias($"blk.{i}.attn_v.bias", _numKvHeads * layerHd);
                }
                // Output-projection bias is optional (Qwen2 omits it; left null when absent).
                if (_hasAttnOutputBias)
                    _bo[i] = LoadBias($"blk.{i}.attn_output.bias", _embDim);
            }

            if (_hasQkNorm && !hp.UseL2QkNorm)
            {
                int qNormSize = _perChannelQkNorm ? _numHeads * layerHd : layerHd;
                _qNorm[i] = LoadBias($"blk.{i}.attn_q_norm.weight", qNormSize);
                // KV-share layers (Gemma 4 shared_kv_layers tail) reuse the source layer's
                // already-normed K, so they carry no attn_k_norm — the QAT q4_0 GGUF omits it
                // (the Q8_0 ships a dead, never-read copy). ApplyQkNormLayer passes k=null for
                // these layers, so _kNorm[i] is never dereferenced; leave it null. Mirrors the
                // attn_k/attn_v guard above and the CUDA/hybrid loaders (#211).
                if (!kvShared)
                {
                    int kNormSize = _perChannelQkNorm ? _numKvHeads * layerHd : layerHd;
                    _kNorm[i] = LoadBias($"blk.{i}.attn_k_norm.weight", kNormSize);
                }
            }

            if (_postAttnNorm is not null)
                _postAttnNorm[i] = ResolveTensor($"blk.{i}.post_attention_norm.weight");
            if (_postFfwNorm is not null)
                _postFfwNorm[i] = ResolveTensor($"blk.{i}.post_ffw_norm.weight");
            if (_layerOutputScale is not null)
                _layerOutputScale[i] = LoadScalarF32($"blk.{i}.layer_output_scale.weight");
        }

        _outputNorm = ResolveTensor("output_norm.weight");
        _outputWeight = model.FindTensor("output.weight") is not null
            ? ResolveTensor("output.weight")
            : _embTensor; // tied embeddings

        if (hp.HasPerLayerTokenEmbd)
        {
            if (model.FindTensor("per_layer_token_embd.weight") is null
                || model.FindTensor("per_layer_model_proj.weight") is null
                || model.FindTensor("per_layer_proj_norm.weight") is null)
            {
                throw new InvalidOperationException(
                    "ModelHyperparams.HasPerLayerTokenEmbd is true but one or more PLE tensors " +
                    "(per_layer_token_embd / per_layer_model_proj / per_layer_proj_norm) are missing.");
            }

            _pleWidth = hp.PerLayerEmbeddingWidth;
            int stackedDim = L * _pleWidth;

            _pleTokenEmbed = ResolveTensor("per_layer_token_embd.weight");

            var projInfo = model.FindTensor("per_layer_model_proj.weight")!.Value;
            var projData = model.GetTensorData(projInfo);
            int projCount = (int)projInfo.ElementCount;
            _perLayerModelProj = Alloc(projCount);
            Dequantize.ToFloat32(projData, new Span<float>(_perLayerModelProj, projCount),
                projInfo.DType, projCount);

            _perLayerProjNormTensor = ResolveTensor("per_layer_proj_norm.weight");

            _pleInpGate = new TensorRef[L];
            _plePostProj = new TensorRef[L];
            _plePostNorm = new TensorRef[L];
            for (int i = 0; i < L; i++)
            {
                _pleInpGate[i] = ResolveTensor($"blk.{i}.inp_gate.weight");
                _plePostProj[i] = ResolveTensor($"blk.{i}.proj.weight");
                _plePostNorm[i] = ResolveTensor($"blk.{i}.post_norm.weight");
            }

            _pleRowBuf = Alloc(stackedDim);
            _projPerLayer = Alloc(stackedDim);
            _pleX = Alloc(_pleWidth);
            _pleY = Alloc(_embDim);
        }

        // ── Dequant-once BLAS weight cache budget (issue #189) ──────────────────────
        // Only dense models route batched prefill through MatMulBatched: MoE uses its own
        // expert path and Gemma 4 per-layer head_dim falls back to sequential Forward, so
        // neither would ever consult the cache. And without OpenBLAS the batched path stays
        // on the fused register-dequant MatVec, where a separate F32 cache is a net loss.
        bool cacheable = SimdKernels.BlasAvailable && !_hp.IsMoE && _layerHeadDim is null;
        long fullF32Bytes = 0;
        if (cacheable)
        {
            // Sum only weights that will actually be cached: skip unset tensors (a default
            // TensorRef has a null Dimensions array, whose ElementCount throws), e.g. the
            // absent attn_v on k_eq_v layers. MatMulBatchedCached caches the same set.
            static long F32Bytes(in TensorRef t) =>
                t.DataPtr is null ? 0 : t.Info.ElementCount * sizeof(float);
            for (int l = 0; l < _hp.NumLayers; l++)
                fullF32Bytes += F32Bytes(_wq[l]) + F32Bytes(_wk[l]) + F32Bytes(_wv[l])
                    + F32Bytes(_wo[l]) + F32Bytes(_wGate[l]) + F32Bytes(_wUp[l]) + F32Bytes(_wDown[l]);
        }
        _dequantCacheBudgetBytes = ResolveDequantCacheBudget(prefillDequantCacheBytes, fullF32Bytes, cacheable);
        _dequantCacheEnabled = _dequantCacheBudgetBytes > 0;
        _dequantCacheCovers = _dequantCacheEnabled && fullF32Bytes > 0
            && _dequantCacheBudgetBytes >= fullF32Bytes;

        PrefaultWeights();
    }

    /// <summary>
    /// Resolve the issue #189 dequant-cache byte budget. <paramref name="requested"/> is the
    /// programmatic override in <i>bytes</i> (<c>long.MinValue</c> = resolve from the
    /// <c>STINGRAY_PREFILL_DEQUANT_MB</c> env var); for both sources <c>0</c> = off,
    /// negative = unlimited, positive = explicit budget. The env "auto" default (unset or
    /// <c>auto</c>) enables the cache only when a full F32 copy of the projection weights
    /// fits within a quarter of available RAM (as reported by <see cref="GC.GetGCMemoryInfo"/>,
    /// which reflects the container/cgroup limit when the runtime detects one), mirroring the
    /// engine's KV-budget auto-sizing.
    /// </summary>
    private static long ResolveDequantCacheBudget(long requested, long fullF32Bytes, bool cacheable)
    {
        if (!cacheable) return 0;

        if (requested != long.MinValue)
            return requested == 0 ? 0 : requested < 0 ? long.MaxValue : requested;

        var raw = Environment.GetEnvironmentVariable("STINGRAY_PREFILL_DEQUANT_MB");
        if (raw is null || raw.Equals("auto", StringComparison.OrdinalIgnoreCase)
            || !long.TryParse(raw, System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out long mb))
        {
            long avail = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            return fullF32Bytes > 0 && fullF32Bytes <= avail / 4 ? fullF32Bytes : 0;
        }
        return MbToBudgetBytes(mb);
    }

    /// <summary>MiB→byte budget with the #189 sign convention (0=off, &lt;0=unlimited), saturating
    /// to <see cref="long.MaxValue"/> instead of overflowing on an absurdly large MiB value.</summary>
    public static long MbToBudgetBytes(long mb) =>
        mb == 0 ? 0
        : mb < 0 || mb > long.MaxValue / (1024 * 1024) ? long.MaxValue
        : mb * 1024 * 1024;

    /// <summary>
    /// Pre-fault every weight page so the first request doesn't stall on demand paging
    /// (issue #221). This is the fully-CPU pass — the whole model is mmap-resident, the
    /// user chose to run it from RAM, so <see cref="MmapPrefault.RamGate.Always"/> skips
    /// the RAM-fit heuristic (subject only to the <c>STINGRAY_PREFAULT=0</c> kill switch).
    /// </summary>
    private void PrefaultWeights()
    {
        var regions = new List<(nint, long)>();
        void Add(TensorRef t)
        {
            if (t.DataPtr != null) regions.Add(((nint)t.DataPtr, t.Info.ByteSize));
        }

        Add(_embTensor); Add(_outputNorm); Add(_outputWeight);
        int L = _hp.NumLayers;
        for (int i = 0; i < L; i++)
        {
            bool kvShared = _layerKvSrc is not null && _layerKvSrc[i] >= 0;
            Add(_attnNorm[i]);
            Add(_wq[i]); Add(_wo[i]);
            // k_eq_v global layers have no attn_v (_wv[i] is default/unset; Add skips null).
            if (!kvShared) { Add(_wk[i]); Add(_wv[i]); }
            Add(_ffnNorm[i]);
            if (_postAttnNorm is not null) Add(_postAttnNorm[i]);
            if (_postFfwNorm is not null) Add(_postFfwNorm[i]);

            if (_hp.IsMoE)
            {
                Add(_wGateInp![i]);
                Add(_wGateExps![i]); Add(_wUpExps![i]); Add(_wDownExps![i]);
                if (_hp.HasSharedExpert)
                {
                    Add(_wGateShexp![i]); Add(_wUpShexp![i]); Add(_wDownShexp![i]);
                }
            }
            else
            {
                Add(_wGate[i]); Add(_wUp[i]); Add(_wDown[i]);
            }
        }

        MmapPrefault.Run("ForwardPass", regions, MmapPrefault.RamGate.Always);
    }

    public PagedKvCache Cache => _kvCache;

    /// <summary>Vocabulary size of this model.</summary>
    public int VocabSize => _hp.VocabSize;

    /// <summary>Maximum supported sequence length.</summary>
    /// <summary>
    /// The longest sequence this pass can actually run — the SMALLEST of the real limits, not
    /// merely the KV cache's.
    ///
    /// <para>It used to return <c>_kvCache.MaxSeqLen</c> alone, which is the paged cache's block
    /// capacity (<c>maxBlocks * PageSize</c> = 131,072 by default) and has nothing to do with the
    /// RoPE tables, allocated at <c>_ctxLen</c> = min(requested, trained) positions. Every batching
    /// caller trusts this number — <c>ContinuousBatchingEngine</c> clamps admission to it and
    /// <c>HotSession</c> refuses turns that exceed it — so on an 8192-context model it advertised
    /// 16x more headroom than the RoPE tables had, and prefilling into that headroom died with an
    /// <c>AccessViolationException</c> in <c>ApplyRopeLayer</c> rather than a clean refusal.
    /// Pinned by <c>MaxSeqLenContractTests</c>.</para>
    /// </summary>
    public int MaxSeqLen => Math.Min(_kvCache.MaxSeqLen, _ctxLen);

    /// <summary>
    /// Bytes of (fp32) KV cache one token occupies across all layers. Used by
    /// <see cref="ContinuousBatchingEngine"/> to convert a memory budget into a token
    /// budget for admission backpressure (issue #183). Page-granularity slack in
    /// <see cref="PagedKvCache"/> (16-position pages) is not included — the budget is
    /// a planning estimate, not an exact accounting.
    /// </summary>
    public long KvBytesPerToken => (long)_hp.NumLayers * _numKvHeads * _headDim * 2 * sizeof(float);

    /// <summary>
    /// Whether SnapKV prefill eviction is configured (via STINGRAY_SNAPKV*). The
    /// continuous-batching engine checks this to disable chunked/packed prefill:
    /// SnapKV scoring only runs on a fresh full-prompt prefill (startPos == 0), so
    /// splitting the prompt into chunks would silently skip eviction.
    /// </summary>
    public bool SnapKvEnabled => _snapKvCfg.Enabled;

    /// <summary>
    /// Whether the issue #189 dequant-once BLAS weight cache is active <i>and</i> budgeted to
    /// hold every batched-prefill projection weight. When true, chunked prefill at small chunk
    /// sizes re-pays no dequant after the first sweep, so <see cref="ContinuousBatchingEngine"/>
    /// can default to a small prefill chunk (low decode-stall) without the throughput collapse
    /// the per-chunk re-dequant would otherwise cause.
    /// </summary>
    public bool PrefillDequantCacheActive => _dequantCacheCovers;

    /// <summary>Resolved dequant-cache byte budget (issue #189): 0 = off, long.MaxValue = unlimited.</summary>
    public long PrefillDequantCacheBudgetBytes => _dequantCacheBudgetBytes;

    /// <summary>
    /// Truncate the KV cache to the given length, discarding positions >= length.
    /// Used by speculative decoding to rewind rejected draft tokens.
    /// Not supported when TurboQuant is enabled and the target length falls in the compressed range.
    /// </summary>
    public void TruncateTo(int length)
    {
        if (_tqKvCache != null)
            _tqKvCache.TruncateTo(length);
        else
            _kvCache.TruncateTo(length);
    }

    /// <inheritdoc />
    public bool SupportsPartialRewind => true;

    /// <inheritdoc />
    /// <remarks>TurboQuant compresses KV in place once it leaves the FP32 recent window, so
    /// those positions cannot be rewound into. Reported per read because it grows as decoding
    /// proceeds.</remarks>
    public int MinRewindLength => _tqKvCache?.MaxTqLength ?? 0;

    public void ResetCache()
    {
        if (_tqKvCache != null)
            _tqKvCache.Reset();
        else
            _kvCache.Reset();
    }

    /// <summary>
    /// Enables TurboQuant KV cache compression. Must be called before any forward pass.
    /// <paramref name="quantizer"/> selects the compressed-region codec: Lloyd-Max
    /// FastScan (this method's default, 3-4 bit; <paramref name="bits"/> applies) or
    /// KVarN (issue #180: 4-bit K / 2-bit V in 128-token tiles; <paramref name="bits"/>
    /// is ignored). Prefer KVarN when it fits your configuration — Lloyd-Max 3-bit
    /// severely degrades quality on QK-norm models such as Qwen3 (issue #432:
    /// Qwen3-0.6B wikitext-2 PPL 15.47 fp32 / 15.67 KVarN / 945.6 Lloyd-Max 3-bit;
    /// the CLI and server default to KVarN where supported). KVarN does not compose
    /// with SnapKV eviction yet — the combo is rejected here rather than corrupting
    /// the cache at Compact time.
    /// KVarN also shrinks the guaranteed <see cref="TruncateTo"/> rewind depth to
    /// <paramref name="fp32WindowSize"/> − 127 (whole-tile promotion): keep the
    /// window well above the draft length when combining with speculative decoding.
    /// </summary>
    public void EnableTurboQuant(int fp32WindowSize = 256, int bits = 3,
        TqQuantizer quantizer = TqQuantizer.LloydMax)
    {
        if (quantizer == TqQuantizer.KVarN && _snapKvCfg.Enabled)
            throw new NotSupportedException(
                "SnapKV eviction (STINGRAY_SNAPKV_BUDGET) is not yet supported with the KVarN quantizer " +
                "(issue #180 follow-up: Compact-time re-quantization needs whole-tile re-assembly). " +
                "Unset STINGRAY_SNAPKV_BUDGET or use the Lloyd-Max quantizer.");

        _tqKvCache = new TurboQuantKvCache(
            _hp.NumLayers, _ctxLen, _numKvHeads, _headDim,
            Math.Min(fp32WindowSize, _ctxLen), bits,
            layerIndexBase: 0, totalLayerCountForSeeds: _hp.NumLayers,
            quantizer: quantizer);
        _rotatedQuery = Alloc(_numHeads * _headDim);
        _decompBuf = Alloc(_numHeads * _headDim);
    }

    /// <summary>The TurboQuant KV cache, if enabled.</summary>
    public TurboQuantKvCache? TqCache => _tqKvCache;

    /// <summary>
    /// Reports whether this model can use the regular CPU batched-prefill trunk for an ordinary
    /// multi-token prompt. This is a load-time capability, not a claim that every request or
    /// projection takes the activation-Q8 kernel.
    /// </summary>
    public CpuBatchedPrefillCapability GetBatchedPrefillCapability() =>
        CpuBatchedPrefillCapability.Evaluate(
            turboQuantEnabled: _tqKvCache is not null,
            isMoe: _hp.IsMoE,
            moeBatchedPrefillSupported: MoeBatchedPrefillSupported,
            // STINGRAY_PER_LAYER_HD_PREFILL can deliberately force an experimental, known-wrong
            // path for measurement. A production-facing receipt must report supported capability,
            // not merely whether that escape hatch will execute code.
            perLayerHeadDimUnsupported: _layerHeadDim is not null);

    /// <summary>
    /// Prefill: process all prompt tokens layer-by-layer.
    /// Weights stay hot in L3 cache across tokens within each layer,
    /// amortizing DRAM reads ~N× vs sequential Forward() calls.
    /// Returns logits for the last token.
    /// </summary>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) =>
        PrefillDispatch(tokens, startPos, onAllPositionLogits: null);

    /// <summary>
    /// Delegate for <see cref="PrefillWithPerPositionLogits"/>. Not <c>Action&lt;int,
    /// ReadOnlySpan&lt;float&gt;&gt;</c> because <see cref="ReadOnlySpan{T}"/> is a ref struct
    /// and cannot be a generic type argument — a plain delegate declaration has no such
    /// restriction on its parameter types.
    /// </summary>
    public delegate void PositionLogitsCallback(int position, ReadOnlySpan<float> logits);

    /// <summary>
    /// Diagnostic sibling of <see cref="Prefill"/> (docs/cpu-prefill-plan.md §14): invokes
    /// <paramref name="onAllPositionLogits"/> once per prompt position (0-based within this
    /// call) with that position's full-vocab logits, instead of only the last. Exists so a
    /// perplexity/quality tool can score every position through the SAME batched-prefill path
    /// production prefill actually uses (i.e. through <see cref="SimdKernels.MatMulBatched"/>,
    /// so it is sensitive to <see cref="SimdKernels.Q8PrefillEnabled"/>) instead of the
    /// token-by-token <see cref="Forward"/> loop, which never touches that path at all.
    ///
    /// <para>
    /// Deliberately a separate method rather than an optional parameter on <see cref="Prefill"/>:
    /// <see cref="Prefill"/> implements <see cref="IForwardPass.Prefill"/>, whose signature is
    /// fixed by the interface, so adding a parameter there breaks interface satisfaction.
    /// </para>
    ///
    /// <para>
    /// The callback's span aliases a reused buffer — read it before returning, it is
    /// invalidated by the next callback or by this method returning.
    /// </para>
    /// </summary>
    public ReadOnlySpan<float> PrefillWithPerPositionLogits(
        IReadOnlyList<int> tokens, int startPos, PositionLogitsCallback onAllPositionLogits)
    {
        ArgumentNullException.ThrowIfNull(onAllPositionLogits);
        return PrefillDispatch(tokens, startPos,
            (n, logits) => onAllPositionLogits(n, logits));
    }

    private ReadOnlySpan<float> PrefillDispatch(
        IReadOnlyList<int> tokens, int startPos, PositionLogitsCallback? onAllPositionLogits)
    {
        int N = tokens.Count;
        if (N == 0) throw new ArgumentException("Token list is empty");
        if (N == 1)
        {
            var single = Forward(tokens[0], startPos);
            onAllPositionLogits?.Invoke(0, single);
            return single;
        }

        // Quantised activation prefill is well behaved on ordinary prompts, but an all-control
        // two-token prompt was observed to produce a negative final-logit cosine versus the F32
        // decode route. Control-only sequences are structural probes, not user prose; retaining
        // their F32 path costs negligible normal-prompt performance and removes an unsafe default
        // divergence. A mixed prompt (including the usual BOS + text) remains eligible for Q8.
        if (IsAllControlTokenPrompt(tokens) || IsSingleDistinctTokenPrompt(tokens))
        {
            ReadOnlySpan<float> logits = default;
            for (int i = 0; i < N; i++)
            {
                logits = Forward(tokens[i], startPos + i);
                onAllPositionLogits?.Invoke(i, logits);
            }
            return logits;
        }

        // MoE models: batched prefill runs the CSR-bucketed per-expert FFN (MoeFfnBatched) when
        // MoeBatchedPrefillSupported admits the model; the configurations it excludes (TurboQuant
        // cache, router/norm traces, Gemma-family post-layer transforms) still prefill per token.
        // Per-layer head-dim models (Gemma 4): the batched PrefillCore path assumes a
        // single qDim/kvDim across layers, so fall back to sequential Forward until
        // Phase 8 plumbs per-layer head_dim through the batched paths.
        //
        // Note for onAllPositionLogits callers: this fallback never calls MatMulBatched,
        // so it cannot exercise Q8PrefillEnabled — a caller diagnosing that path specifically
        // should confirm the model isn't MoE / doesn't have per-layer head dims first.
        // Per-layer head dims (issue #351) are now plumbed through the batched blocks: buffers are
        // sized from _maxHeadDim with per-layer qDim/kvDim strides, Q/K norms and RoPE take the
        // layer's dim via ApplyRopeLayer (which also picks the SWA rope table), and
        // PrefillCoreAttention derives headDim per layer. SnapKV is NOT covered — SnapKvSelector is
        // still constructed from a single model-wide head dim — so per-layer models with SnapKV
        // eviction active keep taking the sequential path.
        // Per-layer head dims (issue #351) STILL take the sequential trunk, and the reason is no
        // longer the "three quantities" (those are now separated — see below). Opening the gate and
        // RUNNING it on gemma-4-E4B showed the batched core is missing most of the per-layer
        // feature set the sequential trunk carries, not just its strides:
        //
        //   * per-layer KV HEAD COUNT (_hp.LayerKvHeads) — gemma4 mixes MQA global layers with GQA
        //     SWA layers; the batched block uses _numKvHeads everywhere
        //   * KV-layer SHARING (_layerKvSrc): shared layers skip the K/V projection entirely and
        //     attention reads a DIFFERENT layer's cache (effLayer)
        //   * attention_k_eq_v: global layers carry no attn_v at all
        //   * gemma4's per-head PureRmsNorm on V before the cache write
        //   * SLIDING-WINDOW attention — PrefillCoreAttention has no windowSize parameter of any
        //     kind. This is the real blocker: it is a missing feature, not missing plumbing.
        //
        // STINGRAY_PER_LAYER_HD_PREFILL=1 used to force the batched path here, documented as
        // producing "wrong output on gemma4" and existing "to make the remaining work measurable".
        //
        // MEASURED 2026-08-07: it does not produce wrong output — it produces
        // AccessViolationException, "attempted to read or write protected memory". Forcing the
        // batched path makes it index KV with the model-wide head dim (512) on layers that actually
        // carry 256, which walks off the end of the buffers. That is an unsafe native write, not a
        // numerical inaccuracy.
        //
        // So the switch could never serve its stated purpose: a path that corrupts memory cannot be
        // timed. It now fails fast with an explanation instead, which is strictly better than
        // crashing — and the honest sizing of the remaining work has to come from implementing the
        // per-layer plumbing, not from forcing a route that was never bounds-safe.
        if (s_perLayerHeadDimPrefillForced && _layerHeadDim is not null)
        {
            throw new NotSupportedException(
                "STINGRAY_PER_LAYER_HD_PREFILL=1 cannot force batched prefill for a per-layer " +
                "head_dim model (gemma4). The batched path indexes KV with the model-wide head dim, " +
                "so on layers with a smaller head_dim it reads and writes past the buffer — measured " +
                "as an AccessViolationException, not merely incorrect output. Unset the variable; " +
                "the sequential route is correct. See docs/done/gemma4-12b-evidence.md.");
        }
        bool perLayerHdUnsupported = _layerHeadDim is not null;
        bool moeUnsupported = _hp.IsMoE && !MoeBatchedPrefillSupported;
        if (moeUnsupported || perLayerHdUnsupported)
        {
            ReadOnlySpan<float> logits = default;
            for (int i = 0; i < N; i++)
            {
                logits = Forward(tokens[i], startPos + i);
                onAllPositionLogits?.Invoke(i, logits);
            }
            return logits;
        }

        // TurboQuant uses a sibling batched path that populates _tqKvCache
        // (with FastScan tile + staging compression along the way) instead of
        // _kvCache. Without this, decode reads from _tqKvCache and only sees
        // the decode-token K/V — the prompt's K/V would land in _kvCache where
        // TqAttention can't reach it.
        if (onAllPositionLogits != null && _tqKvCache != null)
            throw new NotSupportedException(
                "PrefillWithPerPositionLogits does not support TurboQuant yet " +
                "(PrefillCoreTq has its own separate batched path, not extended for this).");

        if (_tqKvCache != null)
            return PrefillCoreTq(tokens, startPos);

        return PrefillCore(tokens, _kvCache, startPos, onAllPositionLogits);
    }

    /// <summary>
    /// True when every token in the prompt is the SAME token id. Such a prompt is degenerate for
    /// int8 activation prefill: measured on SmolLM2-1.7B-Q4_K_M, a single repeated token drives the
    /// final-logit cosine against the exact F32 route to 0.40-0.48 at every length from 2 to 64
    /// tokens, and to -0.12 for a repeated space. Adding ONE differing token restores it to 0.995+.
    ///
    /// <para>The boundary is that sharp, which is why this is a distinct-count test rather than a
    /// magnitude or dynamic-range test. Two earlier hypotheses were measured and disproved: the
    /// prompt's embedding outlier ratio does not separate the failing class (healthy code scores
    /// worse than collapsing whitespace), and neither does the activation outlier ratio taken at
    /// the point of quantisation (healthy prose contains rows with the same maximum as the
    /// collapsing case). See docs/done/cpu-prefill-quality-gate.md.</para>
    ///
    /// <para>Why identical tokens break it: with one repeated token the rows entering each matmul
    /// differ only by positional effects, so the information is carried entirely in small
    /// differences riding on a large common component. Per-row int8 scales to the common
    /// component and quantises those differences away. Ordinary prompts carry their signal in the
    /// differences BETWEEN tokens, which are large enough to survive.</para>
    ///
    /// <para>Deliberately a function of the token ids alone, matching
    /// <see cref="IsAllControlTokenPrompt(IReadOnlyList{int})"/>: a statistic computed over
    /// activations or over whatever else shares a batch would make a token's numerics depend on
    /// its neighbours, which is the property PrefillPackedMulti and the chunked-prefill tests
    /// exist to protect.</para>
    /// </summary>
    private static bool IsSingleDistinctTokenPrompt(IReadOnlyList<int> tokens)
    {
        if (tokens.Count < 2) return false;
        int first = tokens[0];
        for (int i = 1; i < tokens.Count; i++)
            if (tokens[i] != first) return false;
        return true;
    }

    /// <summary>Span overload of <see cref="IsSingleDistinctTokenPrompt(IReadOnlyList{int})"/>.</summary>
    private static bool IsSingleDistinctTokenPrompt(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length < 2) return false;
        int first = tokens[0];
        for (int i = 1; i < tokens.Length; i++)
            if (tokens[i] != first) return false;
        return true;
    }

    private bool IsAllControlTokenPrompt(IReadOnlyList<int> tokens)
    {
        if (_controlTokenIds is null) return false;
        for (int i = 0; i < tokens.Count; i++)
        {
            int token = tokens[i];
            if ((uint)token >= (uint)_controlTokenIds.Length || !_controlTokenIds[token])
                return false;
        }
        return true;
    }

    private bool IsAllControlTokenPrompt(ReadOnlySpan<int> tokens)
    {
        if (_controlTokenIds is null) return false;
        for (int i = 0; i < tokens.Length; i++)
        {
            int token = tokens[i];
            if ((uint)token >= (uint)_controlTokenIds.Length || !_controlTokenIds[token])
                return false;
        }
        return true;
    }

    /// <summary>
    /// Batched prefill core: processes N tokens layer-by-layer into the given cache.
    /// Used by <see cref="Prefill"/> (with _kvCache) and <see cref="PrefillWithCache"/> (with an external cache).
    /// </summary>
    /// <summary>
    /// Batched projection used by the prefill cores. When the issue #189 dequant cache is
    /// active and this is a BLAS-engaged quantized GEMM, it dequantizes the weight once into
    /// the cache and reuses it across chunks; otherwise it falls back to the standard
    /// <see cref="SimdKernels.MatMulBatched"/> (dequant-per-call). The cached path is
    /// bit-identical: same F32 weights, same SGEMM.
    ///
    /// <para>Every caller is a prefill core, so this passes <c>allowQ8: true</c> — the batch rows
    /// are positions within one prompt, not independent user sequences. See
    /// <see cref="SimdKernels.MatMulBatched"/>'s <c>allowQ8</c> parameter for why that distinction
    /// is made by the caller rather than inferred from batch size.</para>
    /// </summary>
    private void MatMulBatchedCached(float* output, in TensorRef w, float* input,
        int N, int rows, int cols)
    {
        if (_dequantCacheEnabled && w.DType != DType.Float32 && N >= SimdKernels.MinBatchForBlas)
        {
            float* wf32 = GetDequantWeightF32(in w, rows, cols);
            if (wf32 != null)
            {
                SimdKernels.MatMulBatchedF32(output, wf32, input, N, rows, cols);
                return;
            }
        }
        // Repacked 8-row Q4_K path (perf-loop iteration 42): measured 2.6x over the row-major
        // _8In at the trunk's Q4_K shape.
        //
        // Deliberately NOT gated on a minimum N. An "N >= 8" gate looks harmless — below it there
        // is no token amortisation to pair with the row amortisation — but it is a NUMERICS
        // boundary: a prompt admitted in chunks whose tail falls below the threshold would have
        // some positions computed by this path and others by the row-major one, so chunked and
        // unchunked prefill of the same prompt disagree. That is exactly the defect
        // MinBatchForQ8Prefill had (see its doc comment), caught here by
        // ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull. TryMatMulBatchedQ4Kx8
        // handles any batch size — its ragged tail uses the single-token repacked kernel, which
        // still gets the 8-row win — so there is no reason to have a threshold at all.
        if (SimdKernels.Q8PrefillEnabled)
        {
            byte* packed = GetRepackedQ4Kx8(in w, rows, cols);
            if (packed != null &&
                SimdKernels.TryMatMulBatchedQ4Kx8(output, packed, input, N, rows, cols))
                return;
        }

        SimdKernels.MatMulBatched(output, w.DataPtr, input, N, rows, cols, w.DType, allowQ8: true);
    }

    /// <summary>
    /// Perf-loop iteration 13 (docs/perf-loop-progress.md): dual-weight sibling of
    /// <see cref="MatMulBatchedCached"/> for two weight matrices sharing the same input (FFN
    /// gate+up) -- routes through <see cref="SimdKernels.TryMatMulBatchedDualQ8"/> (shares one
    /// Q8 quantization pass and one Parallel.For dispatch across both) when neither weight is
    /// going through the dequant-cache/BLAS path; falls back to two separate
    /// <see cref="MatMulBatchedCached"/> calls (preserving existing, unchanged behavior) whenever
    /// the dequant cache IS active for either tensor, or the dual Q8 dispatch doesn't support the
    /// dtype (Q5_K/Q2_K/Float32/Q8_0 have no _8In/_4In dot -- same set TryMatMulBatchedQ8 itself
    /// falls back for).
    /// </summary>
    private void MatMulBatchedDualCached(float* output1, in TensorRef w1, float* output2, in TensorRef w2,
        float* input, int N, int rows, int cols)
    {
        bool useCache1 = _dequantCacheEnabled && w1.DType != DType.Float32 && N >= SimdKernels.MinBatchForBlas;
        bool useCache2 = _dequantCacheEnabled && w2.DType != DType.Float32 && N >= SimdKernels.MinBatchForBlas;

        // The repacked Q4_K path OUTRANKS the dual-Q8 path when both are available.
        //
        // Dual-Q8's advantage is real but small: it shares one activation quantisation pass across
        // two weights. The repacked path is far faster per matmul, and gate+up together are ~55% of
        // this model's matmul FLOPs (2 x 8192x2048 against 16.8M for down and 10.5M for q/k/v/o), so
        // leaving them on the dual path starved the repack of most of its benefit. Measured at 931
        // tokens: 93.2 -> 141.5 t/s, a 1.52x end-to-end win, which took the gap to llama.cpp from
        // 1.67x to ~1.14x.
        //
        // This ordering bug survived because the dual-Q8 gate below was audited for CORRECTNESS
        // (see its comment) and left alone, while the repacked path arrived later (iteration 42).
        // Nobody re-asked whether dual-Q8 should still win the race once something faster existed.
        //
        // MatMulBatchedCached already prefers the repacked path internally, so deferring to it twice
        // is all that is needed. Both weights must be repackable or we fall through: taking the
        // repacked path for one and dual-Q8 for the other would mix quantisation schemes within a
        // single FFN.
        if (!useCache1 && !useCache2 && SimdKernels.Q8PrefillEnabled
            && GetRepackedQ4Kx8(in w1, rows, cols) != null
            && GetRepackedQ4Kx8(in w2, rows, cols) != null)
        {
            MatMulBatchedCached(output1, in w1, input, N, rows, cols);
            MatMulBatchedCached(output2, in w2, input, N, rows, cols);
            return;
        }

        // This must mirror MatMulBatched's own Q8 gate exactly, or gate+threshold changes silently
        // desync the two. An earlier version called TryMatMulBatchedDualQ8 unconditionally,
        // bypassing the gate entirely -- caught by
        // ContinuousBatchingTests.PrefillPackedMulti_MatchesSequentialPrefill (a genuine numerics
        // divergence from the F32 MatVec path, not a kernel correctness bug: isolated synthetic
        // tests call TryMatMulBatchedQ8/TryMatMulBatchedDualQ8 directly on both sides and match
        // bit-identically, since neither side of that comparison passes through this gate).
        // Only prefill reaches this method, so the allowQ8 decision is implicitly true here.
        if (!useCache1 && !useCache2 && w1.DType == w2.DType
            && SimdKernels.Q8PrefillEnabled && N >= SimdKernels.MinBatchForQ8Prefill
            && Environment.GetEnvironmentVariable("STINGRAY_DISABLE_DUAL_Q8") != "1")
        {
            if (SimdKernels.TryMatMulBatchedDualQ8(output1, w1.DataPtr, output2, w2.DataPtr, input, N, rows, cols, w1.DType))
                return;
        }
        MatMulBatchedCached(output1, in w1, input, N, rows, cols);
        MatMulBatchedCached(output2, in w2, input, N, rows, cols);
    }

    /// <summary>
    /// Return the F32 dequant of a weight tensor from the issue #189 reuse cache, populating
    /// it on a miss. Returns <c>null</c> when adding this tensor would exceed the byte budget
    /// (fill-and-stop, no eviction — the reuse distance is a whole model sweep, so an LRU
    /// smaller than the model just thrashes) so the caller re-dequants per call instead.
    /// </summary>
    /// <summary>
    /// Repacked 8-row Q4_K weights for this tensor, or null if the shape/ISA does not qualify or
    /// the budget is exhausted. Repacking is done once per tensor and reused for every prefill.
    ///
    /// <para>Budget is shared with the dequant cache's: whatever that path did not claim. The
    /// repacked copy is only ~5.6% larger than the source weights, so this is far cheaper than the
    /// F32 dequant cache — but it is still a second copy, and it loses the mmap sharing of the
    /// original GGUF pages, which is why it is budgeted rather than unconditional.</para>
    /// </summary>
    private byte* GetRepackedQ4Kx8(in TensorRef w, int rows, int cols)
    {
        if (w.DType != DType.Q4_K || !SimdKernels.CanRepackQ4Kx8(rows, cols))
            return null;
        if (_q4kx8Cache.TryGetValue(w.Name, out var hit))
            return (byte*)hit;

        long bytes = SimdKernels.Q4Kx8PackedBytes(rows, cols);
        if (bytes <= 0 || _q4kx8CacheUsedBytes + bytes > _q4kx8CacheBudgetBytes)
            return null;

        var buf = (byte*)NativeMemory.Alloc((nuint)bytes);
        try
        {
            SimdKernels.RepackQ4KMatrix(w.DataPtr, buf, rows, cols);
        }
        catch
        {
            NativeMemory.Free(buf);   // don't orphan the allocation if the repack throws
            throw;
        }
        _q4kx8Cache[w.Name] = (nint)buf;
        _q4kx8CacheUsedBytes += bytes;
        return buf;
    }

    private float* GetDequantWeightF32(in TensorRef w, int rows, int cols)
    {
        if (_dequantWeightCache.TryGetValue(w.Name, out var hit))
            return (float*)hit;

        long elems = (long)rows * cols;
        long totalBytes = DTypeInfo.ByteSize(elems, w.DType);
        // The quant source span and the F32 destination span are addressed with int lengths.
        // A weight too large for that can't be cached (it would also break the per-call
        // MatMulBatched path) — fall back to per-call dequant rather than truncate silently.
        if (elems > int.MaxValue || totalBytes > int.MaxValue)
            return null;

        long bytes = elems * sizeof(float);
        if (_dequantCacheUsedBytes + bytes > _dequantCacheBudgetBytes)
            return null;

        // Zeroed (not Alloc) so a partial dequant can never leave uninitialized tail bytes;
        // Dequantize.ToFloat32 writes the full element count for block-aligned GGUF tensors.
        var buf = (float*)NativeMemory.AllocZeroed((nuint)bytes);
        try
        {
            Dequantize.ToFloat32(
                new ReadOnlySpan<byte>(w.DataPtr, (int)totalBytes),
                new Span<float>(buf, (int)elems),
                w.DType, elems);
        }
        catch
        {
            NativeMemory.Free(buf); // don't orphan the allocation if dequant throws
            throw;
        }
        _dequantWeightCache[w.Name] = (nint)buf;
        _dequantCacheUsedBytes += bytes;
        return buf;
    }

    private ReadOnlySpan<float> PrefillCore(IReadOnlyList<int> tokens, PagedKvCache cache, int startPos,
        PositionLogitsCallback? onAllPositionLogits = null)
    {
        int N = tokens.Count;

        // SnapKV gating (issue #51): only run eviction when this is a fresh
        // prefill (startPos==0), the budget is configured, AND the prompt is
        // long enough that eviction would actually drop something. On short
        // prompts the scoring cost outweighs the savings.
        bool snapKvActive = _snapKvCfg.Enabled
                         && startPos == 0
                         && N > _snapKvCfg.Budget
                         && N > _snapKvCfg.Window;
        if (snapKvActive)
        {
            _snapKv ??= new SnapKvSelector(_numHeads, _numKvHeads, _headDim);
            _snapKv.Reset(N);
        }

        // Batch hidden states: [N, embDim]
        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            // 1. Embed all tokens
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim);

            // Temp buffers for batched operations
            int qDimMax = _numHeads * _maxHeadDim;
            int kvDimMax = _numKvHeads * _maxHeadDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMax * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDimMax * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDimMax * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMax * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            // MoE needs a separate FFN output buffer: the dense path writes the down projection
            // straight back over batchNorm, but every expert re-reads batchNorm, so they cannot
            // share. Only allocated for MoE — dense keeps the in-place buffer it always had.
            // Per-layer head dims only: one zeroed staging row each for K and V, widened from
            // the layer's compact head packing to the cache's _maxHeadDim head stride. Zeroed
            // once — the padding between heads is never re-dirtied, since every scatter writes
            // exactly the same head slots.
            var kStage = _layerHeadDim is not null
                ? (float*)NativeMemory.AllocZeroed((nuint)(kvDimMax * sizeof(float))) : null;
            var vStage = _layerHeadDim is not null
                ? (float*)NativeMemory.AllocZeroed((nuint)(kvDimMax * sizeof(float))) : null;
            bool batchedMoe = _hp.IsMoE;
            var batchMoeOut = batchedMoe
                ? (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)))
                : null;

            try
            {
                bool profPrefill = PrefillProfileTimers.Enabled;
                if (profPrefill) PrefillProfileTimers.CountTokens(N);
                long pStage;

                // 2. Process layer-by-layer
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Per-layer head dim (gemma4 issue #351): every shape below is derived from
                    // THIS layer's head dim, not a model-wide one. The batch buffers above are
                    // sized from _maxHeadDim, so a smaller layer simply packs its rows tighter;
                    // qDim/kvDim are the per-token STRIDE into those buffers for this layer only.
                    // Mirrors GpuForwardPass.RunGemma4Layers, which cuts per-layer views the same way.
                    int layerHd = _layerHeadDim?[layer] ?? _headDim;
                    // Quantity 2: the BUFFER / MATMUL shape, which is the weight's actual row
                    // count — _wq[layer] has _numHeads * layerHd rows and no more, so asking a
                    // narrow layer's projection for qDimMax rows reads past the tensor and faults
                    // inside SimdKernels.DotF32. PrefillCoreAttention independently derives this
                    // same compact value as its own qDim, so making the buffers compact is what
                    // makes producer and consumer agree.
                    //
                    // Quantity 1 — the CACHE's stride — is deliberately NOT this. It stays
                    // _maxHeadDim-wide, and K/V are widened to it at the single point that needs
                    // it, the Append below. Using the compact width there is what handed WriteKv a
                    // short span and threw at PagedKvCache.cs:455.
                    int qDim = _numHeads * layerHd;
                    int kvDim = _numKvHeads * layerHd;
                    long pLayerStart = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    long pNamedTicks = 0;

                    cache.TruncateTo(startPos);
                    var normW = GetNormWeight(_attnNorm[layer]);

                    // Batch RMS norm for all tokens
                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.RmsNorm, d);
                        pNamedTicks += d;
                    }

                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    // Batched Q/K/V projections (single GEMM per weight matrix)
                    MatMulBatchedCached(batchQ, in _wq[layer], batchNorm, N, qDim, _embDim);
                    MatMulBatchedCached(batchK, in _wk[layer], batchNorm, N, kvDim, _embDim);
                    MatMulBatchedCached(batchV, in _wv[layer], batchNorm, N, kvDim, _embDim);

                    // Apply QKV biases per token (Qwen models)
                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.QkvProj, d);
                        pNamedTicks += d;
                    }

                    // Per-head Q/K RMSNorm and RoPE — ordering and NoPE layers
                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    long pRopeTicks = 0, pAttnTicks = 0;
                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        // Qwen3 (weighted QK-norm): norm BEFORE RoPE
                        if (_hasQkNorm && !_hp.UseL2QkNorm)
                        {
                            ApplyQkNorm(qn, kn, layer);
                        }

                        if (useRoPE)
                        {
                            ApplyRopeLayer(qn, startPos + n, _numHeads, layer, layerHd);
                            ApplyRopeLayer(kn, startPos + n, _numKvHeads, layer, layerHd);
                        }

                        // L2 QK-norm (Llama-4): norm AFTER RoPE, only on RoPE layers
                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, layerHd, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, layerHd, _hp.RmsNormEps);
                        }

                        if (kStage is null)
                        {
                            cache.Append(layer,
                                new ReadOnlySpan<float>(kn, kvDim),
                                new ReadOnlySpan<float>(vn, kvDim));
                        }
                        else
                        {
                            // Quantity 1. A flat copy into a kvDimMax-long span would have the
                            // right LENGTH and therefore throw nothing, while placing every head
                            // but the first at the wrong offset — the cache strides heads by
                            // _maxHeadDim, so this is a per-head SCATTER, not a pad.
                            ScatterToCacheStride(kStage, kn, _numKvHeads, layerHd, _maxHeadDim);
                            ScatterToCacheStride(vStage!, vn, _numKvHeads, layerHd, _maxHeadDim);
                            cache.Append(layer,
                                new ReadOnlySpan<float>(kStage, kvDimMax),
                                new ReadOnlySpan<float>(vStage, kvDimMax));
                        }
                        cache.IncrementPosition();
                    }
                    if (profPrefill) pRopeTicks = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;

                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    PrefillCoreAttention(batchQ, cache, layer, N, startPos, batchAttnOut);
                    if (profPrefill) pAttnTicks = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;

                    if (profPrefill)
                    {
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.RoPE, pRopeTicks);
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.Attention, pAttnTicks);
                        pNamedTicks += pRopeTicks + pAttnTicks;
                    }

                    // SnapKV (issue #51): accumulate per-layer last-W query
                    // attention into the global score buffer. batchQ here is
                    // post-RoPE / post-Q-norm — the same vectors that just
                    // wrote scores against the K cache in the per-token loop
                    // above, so the scoring math is internally consistent.
                    if (snapKvActive)
                    {
                        _snapKv!.AccumulateLayer(batchQ, N, cache, layer, startPos,
                            _snapKvCfg.Window);
                    }

                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    // Batched output projection
                    MatMulBatchedCached(batchNorm, in _wo[layer], batchAttnOut, N, _embDim, qDim);

                    // Apply output projection bias (Qwen models)
                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.OutProj, d);
                        pNamedTicks += d;
                    }

                    // Add output projection + residual → batchHidden
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    // FFN: batch norm, batched gate/up GEMM, per-token SiLU, batched down GEMM
                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    pStage = profPrefill ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.RmsNorm, d);
                        pNamedTicks += d;
                        pStage = System.Diagnostics.Stopwatch.GetTimestamp();
                    }

                    // Where this layer's FFN output lands: dense reuses batchNorm in place,
                    // MoE needs its own buffer (see batchMoeOut's declaration).
                    float* ffnOut = batchedMoe ? batchMoeOut : batchNorm;
                    if (batchedMoe)
                    {
                        MoeFfnBatched(layer, batchNorm, batchMoeOut, N);
                    }
                    else
                    {
                        MatMulBatchedDualCached(batchFfnGate, in _wGate[layer], batchFfnUp, in _wUp[layer], batchNorm, N, _intermDim, _embDim);

                        // Per-token SiLU(gate) * up
                        for (int n = 0; n < N; n++)
                            SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                                batchFfnUp + (long)n * _intermDim, _intermDim);

                        MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnGate, N, _embDim, _intermDim);
                    }
                    if (profPrefill)
                    {
                        long d = System.Diagnostics.Stopwatch.GetTimestamp() - pStage;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.Ffn, d);
                        pNamedTicks += d;
                    }

                    // Residual add
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, ffnOut + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }

                    // Hidden-state taps: batchHidden rows are this layer's outputs.
                    if (_taps is { } taps && taps.SlotOf(layer) is int tapSlot && tapSlot >= 0)
                        for (int n = 0; n < N; n++)
                            CaptureTap(startPos + n, tapSlot, batchHidden + (long)n * _embDim);

                    if (profPrefill)
                    {
                        long layerTotal = System.Diagnostics.Stopwatch.GetTimestamp() - pLayerStart;
                        PrefillProfileTimers.Add(PrefillProfileTimers.Category.Other, Math.Max(0, layerTotal - pNamedTicks));
                    }
                }

                // Set KV cache length to startPos + N for subsequent decode calls.
                cache.TruncateTo(startPos + N);

                // SnapKV (issue #51): compact the cache to the selected keep
                // set. Runs once per prefill — the per-token decode path is
                // untouched and pays no extra cost. After compaction
                // cache.Length is the kept-slot count and cache.LogicalLength
                // is the original prompt length, so decode RoPE continues from
                // the right reference frame.
                if (snapKvActive)
                {
                    var keep = _snapKv!.SelectKeepSet(N, _snapKvCfg.Budget, _snapKvCfg.Recency);
                    if (keep.Length < N)
                    {
                        cache.Compact(keep);
                    }
                }
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
                if (batchMoeOut != null) NativeMemory.Free(batchMoeOut);
                if (kStage != null) NativeMemory.Free(kStage);
                if (vStage != null) NativeMemory.Free(vStage);
            }

            // 3. Final norm + output projection. Normally last token only; when
            // onAllPositionLogits is set (diagnostic use, see Prefill's doc comment) every
            // position is projected instead, reusing the same _logits buffer per position
            // (the callback must consume it before the next iteration overwrites it) so this
            // stays a streaming O(vocab) buffer rather than an O(N*vocab) allocation.
            var outNormW = GetNormWeight(_outputNorm);

            if (onAllPositionLogits != null)
            {
                for (int n = 0; n < N; n++)
                {
                    float* hn = batchHidden + (long)n * _embDim;
                    SimdKernels.RmsNorm(hn, hn, outNormW, _embDim, _hp.RmsNormEps);
                    FusedMatVec(_logits, _outputWeight, hn, _hp.VocabSize, _embDim);
                    onAllPositionLogits(n, new ReadOnlySpan<float>(_logits, _hp.VocabSize));
                }
                return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
            }

            float* lastHidden = batchHidden + (long)(N - 1) * _embDim;
            SimdKernels.RmsNorm(lastHidden, lastHidden, outNormW, _embDim, _hp.RmsNormEps);
            FusedMatVec(_logits, _outputWeight, lastHidden, _hp.VocabSize, _embDim);

            return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// TurboQuant variant of <see cref="PrefillCore"/>: identical batched matmul
    /// structure (QKV, attn output, FFN gate/up/down all run as one GEMM per
    /// weight matrix across N tokens), but routes K/V into the TQ cache and
    /// uses TqAttention per token. Between layers the global TQ position
    /// counter snaps back to <paramref name="startPos"/> while per-layer
    /// FastScan tile/staging/FP32 state stays intact — each layer's TQ window
    /// evolves independently as the N tokens stream through it.
    ///
    /// <para>Unlike <see cref="PrefillCore"/>, this deliberately does not opt into
    /// <see cref="SimdKernels.MatMulBatched"/>'s int8 path (it leaves <c>allowQ8</c> at its
    /// default of <c>false</c>). TurboQuant already trades accuracy for KV footprint, and
    /// stacking int8 activation quantization on top of it is a separate quality question that no
    /// perplexity/greedy-parity measurement covers yet. A deliberate scope boundary, not an
    /// oversight — TQ prefill forgoes the ~+47% int8 win until that is measured.</para>
    /// </summary>
    private ReadOnlySpan<float> PrefillCoreTq(IReadOnlyList<int> tokens, int startPos)
    {
        var cache = _tqKvCache!;
        int N = tokens.Count;

        // SnapKV (issue #60) gating: fresh prefill, explicit budget, prompt
        // long enough to drop something. TQ is fine to compose with — the
        // selector reads from the same cache the per-token TqAttention writes
        // and the compaction promotes the oldest FP32-window survivors into
        // the TQ region as needed.
        bool snapKvActive = _snapKvCfg.Enabled
                         && startPos == 0
                         && N > _snapKvCfg.Budget
                         && N > _snapKvCfg.Window;
        if (snapKvActive)
        {
            _snapKv ??= new SnapKvSelector(_numHeads, _numKvHeads, _headDim);
            _snapKv.Reset(N);
        }

        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim);

            int qDimMax = _numHeads * _maxHeadDim;
            int kvDimMax = _numKvHeads * _maxHeadDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMax * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDimMax * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDimMax * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDimMax * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));

            try
            {
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Per-layer head dim (gemma4 issue #351): every shape below is derived from
                    // THIS layer's head dim, not a model-wide one. The batch buffers above are
                    // sized from _maxHeadDim, so a smaller layer simply packs its rows tighter;
                    // qDim/kvDim are the per-token STRIDE into those buffers for this layer only.
                    // Mirrors GpuForwardPass.RunGemma4Layers, which cuts per-layer views the same way.
                    int layerHd = _layerHeadDim?[layer] ?? _headDim;
                    // STRIDES STAY UNIFORM. PagedKvCache holds one model-wide _kvDim and copies
                    // exactly that many floats in WriteKv, so a per-layer stride hands it a short
                    // span and throws (PagedKvCache.cs:455). layerHd belongs in the ARITHMETIC —
                    // norms, RoPE, attention head dim — never in the addressing. Narrow layers
                    // simply leave the row tail unused; attention only ever reads h*layerHd for
                    // h < the layer's head count, so the tail is written to cache but never read.
                    int qDim = qDimMax;
                    int kvDim = kvDimMax;
                    // Snap the shared global position counter back to startPos
                    // for this layer's per-token loop. Per-layer TQ tile + FP32
                    // window state from prior layers is untouched.
                    cache.ResetTotalLengthForBatchedPrefill(startPos);
                    var normW = GetNormWeight(_attnNorm[layer]);

                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }

                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        if (_hasQkNorm && !_hp.UseL2QkNorm)
                            ApplyQkNorm(qn, kn, layer);

                        if (useRoPE)
                        {
                            ApplyRopeLayer(qn, startPos + n, _numHeads, layer, layerHd);
                            ApplyRopeLayer(kn, startPos + n, _numKvHeads, layer, layerHd);
                        }

                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, layerHd, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, layerHd, _hp.RmsNormEps);
                        }

                        cache.Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        cache.IncrementPosition();

                        Copy(_q, qn, qDim);
                        TqAttention(layer, startPos + n);

                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                    }

                    // SnapKV (issue #60): same shape as PrefillCore's call but
                    // against the TQ cache. batchQ is post-RoPE / post-Q-norm —
                    // the same vectors TqAttention just used to write scores
                    // against the TQ-compressed + FP32-ring K state.
                    if (snapKvActive)
                    {
                        _snapKv!.AccumulateLayer(batchQ, N, cache, layer, startPos,
                            _snapKvCfg.Window);
                    }

                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);

                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);

                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);

                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }

                // _totalLength was advanced to startPos + N by the last layer's
                // per-token loop, which is the state subsequent decode calls expect.

                // SnapKV (issue #60): compact the TQ cache to the selected keep
                // set. Runs once per prefill — per-token decode is untouched.
                // After compaction Length is the kept-slot count; decode RoPE
                // for the next token continues from `startPos + N` (the caller's
                // position counter is unchanged), which is the right post-eviction
                // reference frame because RoPE depends on absolute position not
                // on cache slot index.
                if (snapKvActive)
                {
                    var keep = _snapKv!.SelectKeepSet(N, _snapKvCfg.Budget, _snapKvCfg.Recency);
                    if (keep.Length < N)
                    {
                        cache.Compact(keep, N);
                    }
                }
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }

            float* lastHidden = batchHidden + (long)(N - 1) * _embDim;
            var outNormW = GetNormWeight(_outputNorm);
            SimdKernels.RmsNorm(lastHidden, lastHidden, outNormW, _embDim, _hp.RmsNormEps);
            FusedMatVec(_logits, _outputWeight, lastHidden, _hp.VocabSize, _embDim);

            return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// Whether <see cref="BatchVerify"/> can run (issue #207): everything except the two
    /// configurations it throws for — the TurboQuant KV cache (compressed ring can't take
    /// the batched appends) and gemma4-style per-layer head_dim (not wired into the batched
    /// trunk) — and a SnapKV-compacted cache. After <c>Compact</c> the physical slot count
    /// (<see cref="PagedKvCache.Length"/>) sits below the logical RoPE position
    /// (<see cref="PagedKvCache.LogicalLength"/>), but <see cref="BatchVerify"/> appends at
    /// the LOGICAL position via <c>TruncateTo(startPos)</c>, which would declare slots
    /// past the compacted length valid and read garbage K/V — same #130 gate the CUDA and
    /// GDN passes already have; the sequential <see cref="Forward"/> fallback handles the
    /// compacted frame correctly. MoE stays <c>true</c>: <see cref="BatchVerify"/> itself
    /// falls back to sequential <see cref="Forward"/> calls for MoE, which is still correct.
    /// </summary>
    // ── Hidden-state taps (DSpark draft conditioning, PR #413 spec) ──

    /// <summary>
    /// Taps require stable absolute positions, which SnapKV compaction breaks,
    /// and are captured on the standard dense paths only (no TurboQuant KV).
    /// </summary>
    public bool SupportsHiddenTaps => !_snapKvCfg.Enabled && _tqKvCache is null;

    public int HiddenTapDim => _taps?.TapDim ?? 0;

    public void EnableHiddenTaps(ReadOnlySpan<int> layerIds)
    {
        if (!SupportsHiddenTaps)
            throw new NotSupportedException(
                "Hidden-state taps are not supported with SnapKV eviction or a TurboQuant KV cache " +
                "(both break the absolute-position indexing taps rely on).");
        // Gemma-family post-layer transforms (post-FFW norm, PLE injection, per-layer
        // output scale) run only on the sequential RunTrunk path; the batched
        // Prefill/BatchVerify cores capture at the plain FFN-residual point. Until the
        // batched cores mirror those transforms, taps on such models would record
        // different values per path — reject rather than desync silently. (Gemma 4
        // per-layer head_dim models already route every batched call to sequential
        // Forward, but the guard keeps the contract explicit.)
        // The exception is a model whose batched calls ALL route to sequential Forward anyway —
        // per-layer head_dim, i.e. Gemma 4. There is then no second capture point to disagree
        // with, which is the only thing this guard exists to prevent, so rejecting it buys
        // nothing and costs the ability to diff Gemma 4 layer-by-layer against another backend.
        if ((_postFfwNorm is not null || _layerOutputScale is not null || _hp.HasPerLayerTokenEmbd)
            && _layerHeadDim is null)
            throw new NotSupportedException(
                "Hidden-state taps are not supported on models with post-FFW norm / per-layer " +
                "output scale / PLE (capture points differ between sequential and batched paths).");

        _taps?.Dispose();
        _taps = new HiddenTapBuffer(layerIds, _hp.NumLayers, _embDim, _hp.ContextLength);
    }

    public ReadOnlySpan<float> HiddenTapsAt(int position) =>
        _taps is { } tb ? tb.At(position) : default;

    /// <summary>Copy one tapped layer output (embDim floats) into position/slot.</summary>
    private void CaptureTap(int position, int slot, float* layerOutput)
    {
        new ReadOnlySpan<float>(layerOutput, _embDim).CopyTo(_taps!.RowSlot(position, slot));
    }

    public bool SupportsBatchVerify =>
        _tqKvCache is null
        && _layerHeadDim is null
        && _kvCache.Length == _kvCache.LogicalLength;

    /// <summary>
    /// Batched verification for speculative decoding: processes <paramref name="tokens"/> starting
    /// at <paramref name="startPos"/> using the existing KV cache as context.
    /// All K/V entries are appended to the cache; caller must call TruncateTo to rewind on rejection.
    /// Returns <c>result[i]</c> = logits after processing <c>tokens[i]</c>.
    /// </summary>
    /// <exception cref="NotSupportedException">If TurboQuant KV cache is enabled.</exception>
    public float[][] BatchVerify(int[] tokens, int startPos)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("BatchVerify is not supported when TurboQuant KV cache is enabled.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on the batched BatchVerify path.");

        int N = tokens.Length;
        if (N == 0) return Array.Empty<float[]>();

        if (N == 1 || _hp.IsMoE)
        {
            // Single token or MoE: fall back to sequential Forward calls
            var seq = new float[N][];
            for (int i = 0; i < N; i++)
            {
                var logits = Forward(tokens[i], startPos + i);
                seq[i] = new float[_hp.VocabSize];
                logits.CopyTo(seq[i]);
            }
            return seq;
        }

        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            // 1. Embed all tokens
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim);

            int qDim = _numHeads * _headDim;
            int kvDim = _numKvHeads * _headDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));

            try
            {
                // 2. Process layer-by-layer (same batch structure as Prefill, starting at startPos)
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Restore cache length to startPos so K/V appends land at the right positions
                    _kvCache.TruncateTo(startPos);

                    var normW = GetNormWeight(_attnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }

                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    // Sequential: RoPE (at startPos+n), K/V append, causal attention
                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        int pos = startPos + n;

                        // Qwen3 (weighted QK-norm): norm BEFORE RoPE
                        if (_hasQkNorm && !_hp.UseL2QkNorm)
                        {
                            ApplyQkNorm(qn, kn, layer);
                        }

                        if (useRoPE)
                        {
                            ApplyRope(qn, pos, _numHeads);
                            ApplyRope(kn, pos, _numKvHeads);
                        }

                        // L2 QK-norm (Llama-4): norm AFTER RoPE, only on RoPE layers
                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                        }

                        _kvCache.Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        _kvCache.IncrementPosition();  // _length = startPos + n + 1

                        Copy(_q, qn, qDim);
                        Attention(_kvCache, layer, pos);  // seqLen = startPos + n + 1, uses K/V for 0..pos

                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                    }

                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);

                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);

                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);

                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }

                    // Hidden-state taps: batchHidden rows are this layer's outputs.
                    if (_taps is { } taps && taps.SlotOf(layer) is int tapSlot && tapSlot >= 0)
                        for (int n = 0; n < N; n++)
                            CaptureTap(startPos + n, tapSlot, batchHidden + (long)n * _embDim);
                }

                // Ensure cache length is startPos + N
                _kvCache.TruncateTo(startPos);
                for (int i = 0; i < N; i++) _kvCache.IncrementPosition();
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }

            // 3. Final norm + output projection per position
            var outNormW = GetNormWeight(_outputNorm);
            var result = new float[N][];
            for (int n = 0; n < N; n++)
            {
                float* h = batchHidden + (long)n * _embDim;
                SimdKernels.RmsNorm(h, h, outNormW, _embDim, _hp.RmsNormEps);
                FusedMatVec(_logits, _outputWeight, h, _hp.VocabSize, _embDim);
                result[n] = new float[_hp.VocabSize];
                new ReadOnlySpan<float>(_logits, _hp.VocabSize).CopyTo(result[n]);
            }
            return result;
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// Run one token through the full transformer. Returns logits span.
    /// </summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        _currentPos = position;

        // 1. Embedding lookup (single-row dequant, no full table materialization)
        EmbedToken(token);

        // Gemma family scales embeddings by sqrt(EmbeddingDim) before the trunk.
        if (_hp.EmbeddingScale != 1f)
            SimdKernels.ScaleInPlace(_hidden, _hp.EmbeddingScale, _embDim);

        if (_hp.HasPerLayerTokenEmbd)
            BuildPerLayerProjections(token);

        return RunTrunk(position, token);
    }

    /// <summary>
    /// Forward a single position from a PRECOMPUTED embedding (e.g. a vision soft token)
    /// instead of a token-table lookup, returning the next-token logits.
    ///
    /// Differs from <see cref="Forward"/> in two model-faithful ways for multimodal input
    /// (llama.cpp src/models/gemma4.cpp):
    ///   • does NOT apply the Gemma sqrt(EmbeddingDim) embedding scale — raw image/audio
    ///     embeddings arrive already final (gemma4.cpp:182, "do not normalize weights for
    ///     raw embeddings input"); and
    ///   • uses the padding token (id 0) for the per-layer-embedding (PLE) table lookup,
    ///     while still projecting the supplied embedding (gemma4.cpp build_inp_per_layer
    ///     multimodal branch).
    ///
    /// Attention is causal, consistent with the existing sequential Gemma path. (Gemma's
    /// reference toggles bidirectional attention within an image span; replicating that
    /// needs the batched layer-by-layer path, which is gated off for per-layer-head-dim
    /// models — tracked as a follow-up on issue #250.)
    /// </summary>
    /// <inheritdoc/>
    public bool SupportsEmbeddingInput => true;

    public ReadOnlySpan<float> ForwardEmbedding(ReadOnlySpan<float> embedding, int position)
    {
        if (embedding.Length != _embDim)
            throw new ArgumentException(
                $"embedding length {embedding.Length} != model embedding dim {_embDim}.");

        _currentPos = position;
        embedding.CopyTo(new Span<float>(_hidden, _embDim));

        // Note: no EmbeddingScale here (see remarks). PLE uses the padding token row.
        if (_hp.HasPerLayerTokenEmbd)
            BuildPerLayerProjections(0);

        return RunTrunk(position, traceToken: -1);
    }

    /// <summary>
    /// Shared transformer trunk for <see cref="Forward"/> and <see cref="ForwardEmbedding"/>:
    /// assumes <c>_hidden</c> (and, for PLE models, the per-layer projections) are already
    /// populated for <paramref name="position"/>. <paramref name="traceToken"/> is used only
    /// for the optional norm trace (−1 for embedding input).
    /// </summary>
    private ReadOnlySpan<float> RunTrunk(int position, int traceToken)
    {
        float embNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        StageCapture.Record("cpu", -1, StageCapture.Stages.Embed,
            new ReadOnlySpan<float>(_hidden, _embDim));

        bool profDecode = DecodeProfileTimers.Enabled;
        if (profDecode) DecodeProfileTimers.CountToken();

        // 2. Transformer layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            long layerStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            long namedTicks = 0;
            long stageStart;

            int layerHd = _layerHeadDim?[layer] ?? _headDim;
            // Per-layer KV head count (Gemma 4 12B: 8 GQA on SWA, 1 MQA on the global
            // k_eq_v layers). Falls back to the model-level count for every other arch.
            int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[layer] : _numKvHeads;
            int qDimL = _numHeads * layerHd;
            int kvDimL = layerKv * layerHd;
            int kvSrc = _layerKvSrc is not null ? _layerKvSrc[layer] : -1;
            bool kvShared = kvSrc >= 0;
            int effLayer = kvShared ? kvSrc : layer;
            bool isSwa = _isSwaLayer is not null && _isSwaLayer[layer];
            int windowSize = isSwa ? _hp.SlidingWindowSize : -1;
            // Gemma 4 12B global layers carry no attn_v (attention_k_eq_v): V reuses the
            // raw K projection (pre QK-norm, pre-RoPE). These layers always own their KV.
            bool kEqV = _hp.AttentionKEqV && !isSwa && _wv[layer].DataPtr is null;

            // Save residual
            Copy(_residual, _hidden, _embDim);

            // Pre-attention RMS norm
            var normW = GetNormWeight(_attnNorm[layer]);
            stageStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            FastRmsNorm(_normBuf, _hidden, normW, _embDim, _hp.RmsNormEps);
            StageCapture.Record("cpu", layer, StageCapture.Stages.AttnNorm,
                new ReadOnlySpan<float>(_normBuf, _embDim));
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            // Q projection always runs on the active layer's weights.
            // For per-layer head_dim models the trailing bytes of _q/_k/_v are stale
            // from a wider prior layer; zero them so subsequent Attention reads (which
            // stride by the active layerHd) don't pick up garbage on heads beyond
            // numHeads*layerHd, and KV cache pages don't carry inter-layer pollution.
            if (_layerHeadDim is not null)
            {
                int qBytes = _numHeads * _maxHeadDim;
                int kvBytes = _numKvHeads * _maxHeadDim;
                new Span<float>(_q, qBytes).Clear();
                new Span<float>(_k, kvBytes).Clear();
                new Span<float>(_v, kvBytes).Clear();
            }
            FusedMatVec(_q, _wq[layer], _normBuf, qDimL, _embDim);
            if (!kvShared)
            {
                if (kEqV)
                {
                    // Gemma 4 12B global layers: no attn_v weight — V is the raw K
                    // projection (copied BEFORE QK-norm and RoPE, then plain-RMS-normed
                    // below). Mirrors CudaForwardPass CopyDevice(vView, kView).
                    FusedMatVec(_k, _wk[layer], _normBuf, kvDimL, _embDim);
                    Copy(_v, _k, kvDimL);
                }
                else if (_layerHeadDim is not null)
                {
                    // Gemma 4: K and V share row count and dtype — fuse via
                    // MatVecDual so the row loops interleave and the input
                    // vector reads amortize. The row-interleave changes the FP
                    // ordering vs sequential matvecs by ~ULP; gated to the
                    // per-layer head_dim path because cumulative trunk drift
                    // breaks Qwen3.6-27B-MTP byte parity (see
                    // feedback_qkv_matvecdual_breaks_mtp_parity).
                    SimdKernels.MatVecDual(_k, _wk[layer].DataPtr, _v, _wv[layer].DataPtr,
                        _normBuf, kvDimL, _embDim, _wk[layer].DType, _wv[layer].DType);
                }
                else
                {
                    FusedMatVec(_k, _wk[layer], _normBuf, kvDimL, _embDim);
                    FusedMatVec(_v, _wv[layer], _normBuf, kvDimL, _embDim);
                }
            }
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.QkvProj, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            StageCapture.Record("cpu", layer, StageCapture.Stages.VProj,
                new ReadOnlySpan<float>(_v, kvDimL));

            if (_hasAttnBias)
            {
                SimdKernels.AddInPlace(_q, _bq[layer], qDimL);
                if (!kvShared)
                {
                    SimdKernels.AddInPlace(_k, _bk[layer], kvDimL);
                    SimdKernels.AddInPlace(_v, _bv[layer], kvDimL);
                }
            }

            // NoPE: skip RoPE for NoPE layers
            bool useRoPE = _hp.NoRopeLayerStep == 0
                || (layer + 1) % _hp.NoRopeLayerStep != 0;

            // Qwen3 (weighted QK-norm): apply norm BEFORE RoPE (per reference implementation)
            // Llama-4 (L2 QK-norm): apply norm AFTER RoPE (per llama.cpp)
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                ApplyQkNormLayer(_q, kvShared ? null : _k, layer, layerHd, layerKv);
            }

            // Gemma 4: V is plain per-head RmsNorm (no learned weight) before cache.
            // Matches llama.cpp src/models/gemma4.cpp line 227:
            //   Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps)
            if (_layerHeadDim is not null && !kvShared)
            {
                PerHeadPureRmsNorm(_v, layerKv, layerHd, _hp.RmsNormEps);
            }

            StageCapture.Record("cpu", layer, StageCapture.Stages.VNorm,
                new ReadOnlySpan<float>(_v, kvDimL));

            if (useRoPE)
            {
                ApplyRopeLayer(_q, position, _numHeads, layer, layerHd);
                if (!kvShared)
                    ApplyRopeLayer(_k, position, layerKv, layer, layerHd);
            }

            // L2 QK-norm (Llama-4): only on RoPE layers, applied after RoPE
            if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
            {
                PerHeadPureRmsNorm(_q, _numHeads, layerHd, _hp.RmsNormEps);
                if (!kvShared)
                    PerHeadPureRmsNorm(_k, _numKvHeads, layerHd, _hp.RmsNormEps);
            }

            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.RoPE, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            // Store K, V in cache. KV-share layers don't append — the source layer's
            // cache slot is shared via effLayer in the Attention call below.
            // PagedKvCache.Append requires exactly cache.KvDim floats; for per-layer
            // head_dim models the trailing (KvDim - kvDimL) floats were just zeroed.
            if (!kvShared)
            {
                int appendLen = _layerHeadDim is not null ? _kvCache.KvDim : kvDimL;
                if (_tqKvCache != null)
                {
                    _tqKvCache.Append(layer,
                        new ReadOnlySpan<float>(_k, appendLen),
                        new ReadOnlySpan<float>(_v, appendLen));
                }
                else
                {
                    _kvCache.Append(layer,
                        new ReadOnlySpan<float>(_k, appendLen),
                        new ReadOnlySpan<float>(_v, appendLen));
                }
            }

            // Attention
            if (profDecode) stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            if (_tqKvCache != null)
                TqAttention(layer, position);
            else
                Attention(_kvCache, effLayer, layer, position, layerHd, windowSize, layerKv);
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.Attention, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            // Output projection (input width is per-layer qDim).
            StageCapture.Record("cpu", layer, StageCapture.Stages.AttnOut,
                new ReadOnlySpan<float>(_attnOut, qDimL));

            FusedMatVec(_hidden, _wo[layer], _attnOut, _embDim, qDimL);
            if (_hasAttnOutputBias)
                SimdKernels.AddInPlace(_hidden, _bo[layer], _embDim);
            StageCapture.Record("cpu", layer, StageCapture.Stages.OProj,
                new ReadOnlySpan<float>(_hidden, _embDim));
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.OutProj, d);
                namedTicks += d;
            }

            // Gemma 4: post-attention RmsNorm BEFORE the residual add.
            if (_postAttnNorm is not null)
            {
                var paNormW = GetNormWeight(_postAttnNorm[layer]);
                FastRmsNorm(_hidden, _hidden, paNormW, _embDim, _hp.RmsNormEps);
            }

            // Residual
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            if (_traceNorms) _normTraceAttn![layer] = L2Norm(_hidden, _embDim);

            StageCapture.Record("cpu", layer, StageCapture.Stages.PostAttnResidual,
                new ReadOnlySpan<float>(_hidden, _embDim));

            // Save residual for FFN
            Copy(_residual, _hidden, _embDim);

            // Pre-FFN RMS norm
            var ffnNormW = GetNormWeight(_ffnNorm[layer]);
            stageStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            FastRmsNorm(_normBuf, _hidden, ffnNormW, _embDim, _hp.RmsNormEps);
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm, d);
                namedTicks += d;
                stageStart = System.Diagnostics.Stopwatch.GetTimestamp();
            }

            if (_hp.IsMoE)
                MoeFfn(layer);
            else
                DenseFfn(layer);
            if (profDecode)
            {
                long d = System.Diagnostics.Stopwatch.GetTimestamp() - stageStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.Ffn, d);
                namedTicks += d;
            }

            // Gemma 4: post-FFN RmsNorm before the residual add.
            if (_postFfwNorm is not null)
            {
                var pfNormW = GetNormWeight(_postFfwNorm[layer]);
                FastRmsNorm(_hidden, _hidden, pfNormW, _embDim, _hp.RmsNormEps);
            }

            // Residual (post-attn output that includes its own residual).
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            StageCapture.Record("cpu", layer, StageCapture.Stages.PostFfnResidual,
                new ReadOnlySpan<float>(_hidden, _embDim));

            if (_hp.HasPerLayerTokenEmbd)
                ApplyPerLayerEmbedding(layer);

            StageCapture.Record("cpu", layer, StageCapture.Stages.PostPle,
                new ReadOnlySpan<float>(_hidden, _embDim));

            // Gemma 4: per-layer learned output scale applies AFTER the PLE injection
            // (matches llama.cpp gemma4 build order — applying it before PLE breaks the
            // residual balance and produces unbounded hidden L2 growth).
            if (_layerOutputScale is not null)
                SimdKernels.ScaleInPlace(_hidden, _layerOutputScale[layer], _embDim);

            // Hidden-state tap: _hidden now holds this layer's output (= next layer's input).
            if (_taps is { } taps && taps.SlotOf(layer) is int tapSlot && tapSlot >= 0)
                CaptureTap(position, tapSlot, _hidden);

            StageCapture.Record("cpu", layer, StageCapture.Stages.LayerOutput,
                new ReadOnlySpan<float>(_hidden, _embDim));

            if (_traceNorms) _normTraceFfn![layer] = L2Norm(_hidden, _embDim);

            if (profDecode)
            {
                long layerTotal = System.Diagnostics.Stopwatch.GetTimestamp() - layerStart;
                DecodeProfileTimers.Add(DecodeProfileTimers.Category.Other, Math.Max(0, layerTotal - namedTicks));
            }
        }

        // Increment KV cache position
        if (_tqKvCache != null)
            _tqKvCache.IncrementPosition();
        else
            _kvCache.IncrementPosition();

        float preFinalNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        // 3. Final RMS norm
        var outNormW = GetNormWeight(_outputNorm);
        long finalNormStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        FastRmsNorm(_hidden, _hidden, outNormW, _embDim, _hp.RmsNormEps);
        if (profDecode) DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm, System.Diagnostics.Stopwatch.GetTimestamp() - finalNormStart);

        float postFinalNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        // 4. Output projection → logits (fused, no 400MB intermediate buffer)
        long logitsStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);
        if (profDecode) DecodeProfileTimers.Add(DecodeProfileTimers.Category.OutProj, System.Diagnostics.Stopwatch.GetTimestamp() - logitsStart);

        // Gemma 4 final-logit softcap: x = tanh(x/cap) * cap.
        if (_hp.FinalLogitSoftcap > 0f)
            SimdKernels.SoftcapInPlace(_logits, _hp.VocabSize, _hp.FinalLogitSoftcap);

        if (_traceNorms)
            EmitNormTrace(traceToken, position, embNorm, preFinalNorm, postFinalNorm);

        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    private void ApplyRope(float* x, int pos, int heads)
    {
        var cos = _ropeCosTable + (long)pos * _ropeHalfDim;
        var sin = _ropeSinTable + (long)pos * _ropeHalfDim;
        if (_hp.IsNeoxRope)
            SimdKernels.ApplyRoPECachedNeox(x, cos, sin, heads, _headDim);
        else
            SimdKernels.ApplyRoPECached(x, cos, sin, heads, _headDim);
    }

    /// <summary>
    /// Per-layer-aware RoPE: selects the global or SWA cos/sin table for Gemma 4 and
    /// rotates each head's leading <paramref name="layerHd"/> dims.
    /// </summary>
    private void ApplyRopeLayer(float* x, int pos, int heads, int layer, int layerHd)
    {
        bool useSwa = _ropeCosTableSwa != null && _isSwaLayer is not null && _isSwaLayer[layer];
        int halfDim = useSwa ? _ropeHalfDimSwa : _ropeHalfDim;
        float* cosTab = useSwa ? _ropeCosTableSwa : _ropeCosTable;
        float* sinTab = useSwa ? _ropeSinTable    : _ropeSinTable;
        if (useSwa) sinTab = _ropeSinTableSwa;
        var cos = cosTab + (long)pos * halfDim;
        var sin = sinTab + (long)pos * halfDim;
        if (_hp.IsNeoxRope)
            SimdKernels.ApplyRoPECachedNeox(x, cos, sin, heads, layerHd);
        else
            SimdKernels.ApplyRoPECached(x, cos, sin, heads, layerHd);
    }

    private static float L2Norm(float* x, int n)
    {
        double s = 0;
        for (int i = 0; i < n; i++) { double v = x[i]; s += v * v; }
        return (float)Math.Sqrt(s);
    }

    private void EmitNormTrace(int token, int position,
        float embNorm, float preFinalNorm, float postFinalNorm)
    {
        // Top-1 logit + index
        int topIdx = 0; float topVal = float.MinValue;
        for (int i = 0; i < _hp.VocabSize; i++)
            if (_logits[i] > topVal) { topVal = _logits[i]; topIdx = i; }

        var sb = new System.Text.StringBuilder(2048);
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        sb.Append("[norms pos=").Append(position)
          .Append(" tok=").Append(token)
          .Append(" emb=").Append(embNorm.ToString("F2", inv));
        for (int i = 0; i < _hp.NumLayers; i++)
        {
            sb.Append(" L").Append(i).Append(":a=")
              .Append(_normTraceAttn![i].ToString("F1", inv))
              .Append("/f=").Append(_normTraceFfn![i].ToString("F1", inv));
        }
        sb.Append(" preFN=").Append(preFinalNorm.ToString("F2", inv))
          .Append(" postFN=").Append(postFinalNorm.ToString("F2", inv))
          .Append(" top=").Append(topIdx)
          .Append('@').Append(topVal.ToString("F2", inv));
        Console.Error.WriteLine(sb.ToString());
    }

    // ================================================================
    //  Batched Prefill Attention
    // ================================================================

    public static unsafe void ComputeBatchedCausalAttention(
        float* batchQ, float* batchK, float* batchV, float* batchAttnOut,
        int N, int startPos, int numHeads, int numKvHeads, int headDim, float scale)
    {
        int qDim = numHeads * headDim;
        int kvDim = numKvHeads * headDim;
        int hpkg = numHeads / numKvHeads;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / hpkg;
            int maxSeqLen = startPos + N;
            float* headScores = (float*)NativeMemory.AllocZeroed((nuint)(maxSeqLen * sizeof(float)));
            try
            {
                for (int n = 0; n < N; n++)
                {
                    float* qHead = batchQ + (long)n * qDim + h * headDim;
                    float* outHead = batchAttnOut + (long)n * qDim + h * headDim;

                    int scoreLen = startPos + n + 1;
                    for (int i = 0; i < scoreLen; i++)
                    {
                        float* kVec = batchK + (long)i * kvDim + kvHead * headDim;
                        headScores[i] = SimdKernels.DotF32(qHead, kVec, headDim) * scale;
                    }

                    SimdKernels.SoftmaxInPlace(headScores, scoreLen);

                    for (int d = 0; d < headDim; d++) outHead[d] = 0;

                    for (int i = 0; i < scoreLen; i++)
                    {
                        float* vVec = batchV + (long)i * kvDim + kvHead * headDim;
                        float w = headScores[i];
                        if (Fma.IsSupported && headDim >= 8)
                        {
                            var wv = Vector256.Create(w);
                            int d = 0;
                            for (; d + 8 <= headDim; d += 8)
                            {
                                var o = Avx.LoadVector256(outHead + d);
                                var v = Avx.LoadVector256(vVec + d);
                                Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                            }
                            for (; d < headDim; d++)
                                outHead[d] += w * vVec[d];
                        }
                        else
                        {
                            for (int d = 0; d < headDim; d++)
                                outHead[d] += w * vVec[d];
                        }
                    }
                }
            }
            finally
            {
                NativeMemory.Free(headScores);
            }
        });
    }

    /// <summary>
    /// Batched prefill attention, tiled over the token axis.
    ///
    /// <para>The previous shape was <c>for each head: for each token n: full K pass; softmax; full
    /// V pass</c>. Because <see cref="PrefillCore"/> passes the WHOLE prompt as N, that re-read the
    /// entire K and V cache once per token — for a 3218-token prompt, 3218 passes over ~824 KB per
    /// head, which is far past L2 and therefore goes to RAM every time. The FLOPs are inherently
    /// O(N²); the memory traffic did not have to be.</para>
    ///
    /// <para>Tiling the token loop by <c>TokenTile</c> streams each K[i] (then each V[i]) once per
    /// TILE tokens instead of once per token, so the vector stays hot in L1 while all tokens in the
    /// tile consume it — the same insight as the Vulkan flash-attention rewrite (perf-loop
    /// iteration 31), applied to the cache hierarchy rather than to VRAM bandwidth.</para>
    ///
    /// <para><b>Bit-identical to the untiled form</b>, deliberately: scores are computed by the same
    /// dot in the same order, softmax runs over the same row, and each output still accumulates
    /// over i ascending. Only the loop nesting changes, not the arithmetic order.</para>
    /// </summary>
    private void PrefillCoreAttention(float* batchQ, PagedKvCache cache, int layer, int N, int startPos, float* batchAttnOut)
    {
        int numHeads = _numHeads;
        int numKvHeads = _numKvHeads;
        // Per-layer head dim (issue #351). The attn scale on the next line was already
        // gemma4-aware; the dimension itself was not.
        int headDim = _layerHeadDim?[layer] ?? _headDim;
        int qDim = numHeads * headDim;
        // Quantity 1 (see the doc's "three quantities" table): the CACHE's head stride, which is
        // fixed at construction by _maxHeadDim and is NOT this layer's head dim. K rows are
        // _numKvHeads * _maxHeadDim wide with head h at h * _maxHeadDim, so a narrow layer must
        // still step by the wide stride to find its head — it just reads headDim of it.
        // (ValueAtHead already does this internally; KeyAt returns the row base, so the caller
        // owns the head offset and this is where it was being got wrong.)
        int cacheHeadStride = _layerHeadDim is not null ? _maxHeadDim : headDim;
        int hpkg = numHeads / numKvHeads;
        float scale = _layerHeadDim is not null ? 1.0f : 1.0f / MathF.Sqrt(headDim);
        int ctxLen = _ctxLen;
        bool enableRegisterValues = Environment.GetEnvironmentVariable(
            "STINGRAY_PREFILL_ATTN_REGISTER_VALUES") != "0";
        bool enableFlash64 = Environment.GetEnvironmentVariable(
            "STINGRAY_PREFILL_ATTN_FLASH64") != "0";

        // The 64x64 packing/setup cost loses on short prefills (the isolated crossover is well
        // above 128 tokens). Keeping small chunks on the bit-stable incumbent also preserves the
        // packed-multi parity contract, whose current attention loop is intentionally per-token.
        //
        // The threshold tests startPos + N — the sequence length prefilled so far — NOT N alone.
        // Flash-64 uses online softmax (running max, per-tile rescale) and is therefore NOT
        // bit-identical to the incumbent, so selecting between them on the CHUNK size makes a
        // prompt's logits depend on how it was admitted: at chunk 512 a 600-token prompt splits
        // 512 + 88, and on `N` alone the tail silently fell back to the incumbent while the same
        // prompt prefilled in one pass ran entirely on flash-64. Measured divergence was ~2.5% on a
        // logit (3.655 vs 3.556) — far outside chunk-boundary FP drift, and exactly the defect
        // SimdKernels.cs:76-86 calls out ("a user's logits depend on who else happened to be
        // batched alongside them"). Caught by
        // PrefillAttentionParityTests.ChunkedPrefill_MatchesUnchunked_AcrossFlash64Threshold.
        //
        // KNOWN RESIDUAL: this makes the decision monotonic in sequence position, which fixes every
        // case where the first chunk already reaches the threshold — i.e. any chunk size >= 256,
        // which covers the shipped STINGRAY_PREFILL_CHUNK values. It does NOT fix a prompt whose
        // total exceeds 256 while its individual chunks do not (e.g. 600 tokens at chunk 64): the
        // early chunks take the incumbent and later ones flash-64. Closing that needs the decision
        // threaded down from the caller, which alone knows the prompt's total length.
        // Flash-64 handles BF16 natively by widening each 64x64 tile once during packing, so it must
        // be offered the work FIRST. Routing BF16 around it was measured to cost 41% of prefill
        // throughput (122.2 -> 72.5 t/s at 5314 tokens) — the loss was the missing Flash-64, not the
        // narrower store.
        // Tile width is 64 queries/KVs. The machinery below is fully head-width generic (headDim is
        // a parameter throughout, scratch is sized Tile*headDim) and the strided AVX2 microkernel
        // supports 128/256-wide heads — but the 128/256 WIDTHS ARE HELD BACK HERE, deliberately.
        // Restoring `headDim is 64 or 128 or 256` is all it takes to re-enable them.
        //
        // Why: Flash128_MatchesMaterialisedAttention (Qwen3-8B, 32 heads / 8 KV, headDim 128,
        // 36 layers, Q4_K_M, 256 tokens) fails its own gate — final-logit maxAbs 0.310 against a
        // 0.01 tolerance. Flash-vs-materialised should differ only by FP reassociation, so 0.310 is
        // not obviously explainable as drift. Flash-64 is default-ON, so shipping the wider widths
        // would change prefill numerics for the most common head dim on evidence that is currently
        // ambiguous. Held back rather than reverted: the generalisation is very likely correct and
        // the open question is about the measurement, not the arithmetic.
        //
        // What is already ruled out (do not redo this work):
        //   * The GEMM kernel. GemmF32StridedParityTests covers the exact shapes this path uses —
        //     (64,128,64), (64,64,128), (64,128,128), the ragged query-tile tails, and the 256
        //     variants — and passes. Those tests are retained precisely to pin the kernel for this.
        //   * A generic headDim-128 defect. Qwen3-0.6B (16 heads / 8 KV, headDim 128) diverges by
        //     8.3e-6 with an identical greedy token, and the gate above genuinely activates there,
        //     so that is a real measurement rather than a skipped test.
        //   * An interaction with int8 activation prefill. The divergence survives with
        //     SimdKernels.Q8PrefillEnabled=false (0.258).
        //   * A scratch-sharing race. Ownership is per-iteration `using var` in the default
        //     schedule and ThreadLocal in the tile-jobs schedule; all nine buffers are sized and
        //     freed correctly.
        //   * The BF16 KV branch. Both BF16 store flags are env-driven, not size-driven, so neither
        //     model above took it.
        //
        // MEASURED 2026-08-07 — parity question RESOLVED; a perplexity gate is what remains.
        // On realistic prose (320 tokens, Qwen3-8B), flash-vs-materialised and the ACCEPTED
        // int8-activation-prefill baseline, captured in one process:
        //     flash-vs-materialised : cos 0.999345, maxAbs 0.762, greedy 63762
        //     q8-vs-f32 (ships ON)  : cos 0.999504, maxAbs 0.807, greedy 63762
        // Flash-128's divergence is the same order as an approximation this project already ships
        // enabled by default, and its maxAbs is SMALLER. Greedy token identical. The original 0.01
        // maxAbs bound was the wrong instrument, as suspected below. Note the OOD token sequence in
        // the old test read 0.310 while realistic prose reads 0.762 — a reminder that absolute
        // logit deltas are input-dependent and only meaningful against a calibrated baseline.
        //
        // PERPLEXITY GATE RUN 2026-08-07 — DECISION: keep off by default, ship as an opt-in trade.
        // wikitext-2 subset (120 KB), Qwen3-8B Q4_K_M, `perplexity --batched --batch-chunk-size 512`:
        //     flash-128 OFF : ppl 6.0579 [256,1024)   7.5318 [1024,+)   22.35 tok/s
        //     flash-128 ON  : ppl 6.0896 (+0.52%)     7.5672 (+0.47%)   25.52 tok/s (+14%)
        // Perplexity is deterministic for a fixed path, so +0.5% is reproducible, not noise. It also
        // lands WORSE than the exact sequential path (6.0789), not merely worse than the batched one,
        // so it is not inside the envelope of the approximations already shipped. This project's
        // precedent for a default-on numerics change is the Q4Kx8 repack at 16.0488 -> 16.0484,
        // i.e. ~0%; half a percent is two orders of magnitude larger. The +14% prefill is real but
        // does not buy that quality on someone's behalf — it is the model owner's call, so the
        // widths stay behind STINGRAY_PREFILL_ATTN_WIDE_HEADS=1.
        //
        // Note the earlier cosine/greedy check (cos 0.999345 vs the Q8 baseline's 0.999504, identical
        // greedy token) said "same envelope" and was NOT sufficient: a per-prompt cosine on final
        // logits missed a 0.5% corpus perplexity shift. Corpus gates outrank single-prompt similarity.
        //
        // Timing caveat: one sample per arm on a contended machine, so +14% is soft; the perplexity
        // figures are exact. Full wikitext-2 test split has not been run, only a 120 KB subset.
        // The superseded reasoning is kept below for context.
        //
        // SUPERSEDED: the unresolved question, and the next step to return this to active use: decide whether
        // 0.310 is a defect or an unrealistic tolerance. On the SAME model and input, toggling int8
        // activation prefill — which ships ENABLED BY DEFAULT — moves the same logits by 0.352,
        // i.e. more than the delta this test rejects. So measure the right quantity: cosine
        // similarity and greedy-token agreement for flash-vs-materialised, compared against the
        // accepted Q8-vs-F32 baseline on the same run (maxAbs on raw logits of a 36-layer model is
        // the wrong instrument, and cosine is what this project already uses for such judgements).
        // If flash-vs-materialised is at least as tight as the shipped Q8 baseline, retune the test
        // to cosine + greedy and re-enable. If it is materially worse, the defect is real and is
        // specific to the 32-head / headsPerKv=4 geometry, since 16/8 at the same width is clean —
        // bisect by feeding realistic tokens instead of BuildTokens' out-of-distribution
        // `1 + i*17 % 997` sequence, which can make attention near-degenerate and amplify.
        if (enableFlash64 && startPos + N >= 256 && _layerHeadDim is null
            && (headDim == 64 || (Flash64WideHeadDimsEnabled && headDim is 128 or 256)) &&
            Avx2.IsSupported && Fma.IsSupported)
        {
            PrefillFlashAttention64(batchQ, cache, layer, N, startPos, batchAttnOut,
                numHeads, numKvHeads, qDim, headDim, scale);
            return;
        }

        // Whatever Flash-64 declined (short prefill, non-64 head dim, per-layer head dims, no AVX2)
        // still needs a BF16-aware reader — the F32 loops below would misread 2-byte pages.
        if (cache.IsBf16Store)
        {
            PrefillCoreAttentionBf16(batchQ, cache, layer, N, startPos, batchAttnOut,
                numHeads, numKvHeads, headDim, qDim, cacheHeadStride, hpkg, scale);
            return;
        }

        // Tokens per tile. This is the K/V re-read amortisation factor: the whole K cache, then the
        // whole V cache, is streamed once per TILE tokens, so K/V traffic scales as N/TILE.
        //
        // Originally 8, chosen to keep the score scratch small. A direct sweep
        // (tools/attn-bench, N=3218, three independent runs) shows that was well short of the
        // optimum — the scratch concern was over-weighted and the traffic term keeps paying:
        //     tile      4      8     16     32     64    128    256
        //     speedup 0.70x  1.00x  1.17x  1.35x  1.48x  1.34x  1.04x
        // 64 is the measured optimum and the curve is flat enough either side (32 and 128 both
        // within ~10%) that it is not a knife-edge. Below 16 the per-tile fixed costs dominate;
        // above 128 the scratch itself (TILE * stride floats per head-thread) stops fitting.
        //
        // Scratch at TILE=64 is 64 * stride * 4 bytes per head-thread — 2 MB at ctxLen 8192, so
        // ~24 MB live across 12 concurrent head-threads. That is comfortably RAM-resident, which
        // is the real constraint; it was never going to be L1-resident at any useful tile size.
        const int TokenTile = 64;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / hpkg;
            int maxSeqLen = startPos + N;
            // Size the scratch to the sequence actually being prefilled, not to ctxLen. Every index
            // written below is `i < endSeq`, and endSeq = min(startPos + nBase + t + 1, cache.Length)
            // is bounded by startPos + N = maxSeqLen — so maxSeqLen rows are always sufficient, and
            // ctxLen over-allocated whenever the prompt is shorter than the configured context.
            // This matters more since TokenTile grew to 64: at ctxLen 8192 the old sizing would be
            // 2 MB per head-thread regardless of prompt length, and at a 128k context it would be
            // 33 MB per head-thread — hundreds of MB across the head threads, for scratch that is
            // mostly never touched. Measured perf-neutral on its own; this is a memory bound, not
            // a speed change.
            int stride = maxSeqLen;
            float* scores = (float*)NativeMemory.AllocZeroed((nuint)((long)TokenTile * stride * sizeof(float)));
            bool registerValues = enableRegisterValues && Fma.IsSupported && headDim >= 8 && headDim % 8 == 0;
            float** valueRows = registerValues
                ? (float**)NativeMemory.Alloc((nuint)(Math.Min(maxSeqLen, cache.Length) * sizeof(nint)))
                : null;

            // NOTE (2026-08-02, measured and reverted): phase 3 accumulates straight into
            // batchAttnOut, whose per-token stride is qDim = numHeads * headDim floats — 8192 BYTES
            // at 32 heads / 64 dim. Zen 3's L1d is 32 KB / 8-way / 64 B lines = 64 sets indexed by
            // address mod 4096, and 8192 mod 4096 == 0, so every token's output head maps to the
            // same L1 sets: 64 lines competing for 8 ways.
            //
            // That looks like textbook conflict thrashing, so phase 3 was rewritten to accumulate
            // into a contiguous TokenTile x headDim scratch (16 KB) and copy out once per tile —
            // bit-identical arithmetic, no numerics risk. Properly interleaved measurements found
            // it performance-neutral; the earlier claimed ~9% loss compared against one noisy
            // baseline and was retracted. Moving the same loads did not reduce their uop cost.
            //
            // The uop count, not the cache behaviour, is what made phase 3 expensive: 8 acc loads
            // + 8 V loads + 8 FMAs + 8 stores per (token, KV position), i.e. 4 uops per useful FMA.
            // The registerValues path below fixes that with an 8-token x 8-float microkernel. On
            // the production PagedKvCache shape it measured 1.17-1.20x for whole attention and
            // 1.075x / 1.088x end-to-end at 931 / 2431 tokens, bit-identical in the isolated harness
            // and the chunked-prefill tests. Set STINGRAY_PREFILL_ATTN_REGISTER_VALUES=0 only for
            // controlled A/B measurement. Do not re-try the scratch-buffer variant.
            try
            {
                if (valueRows is not null)
                    for (int i = 0; i < Math.Min(maxSeqLen, cache.Length); i++)
                        valueRows[i] = cache.ValueAtHead(layer, i, kvHead);

                for (int nBase = 0; nBase < N; nBase += TokenTile)
                {
                    int tn = Math.Min(TokenTile, N - nBase);
                    // Longest causal row in this tile — the K/V streaming bound.
                    int endSeqMax = Math.Min(startPos + nBase + tn, cache.Length);

                    // ── Phase 1: scores. Stream K once; every token in the tile consumes it. ──
                    //
                    // NOT batched via SimdKernels.DotF32_4In — see that kernel's remarks. Batching
                    // four query tokens per key vector was tried (2026-08-02) and reverted: it is
                    // worth only ~6% of attention (~2% end-to-end), and because the 4-wide kernel
                    // can only cover `t + 4 <= tn` the tail falls back to DotF32, which sums in a
                    // different order. A token's arithmetic would then depend on how many tokens
                    // share its tile — i.e. on N — so chunked and unchunked prefill of the same
                    // prompt disagree. That broke 4 ContinuousBatchingTests. Wiring it in safely
                    // needs every token on one kernel, which the tile remainder prevents.
                    for (int i = 0; i < endSeqMax; i++)
                    {
                        float* kVec = cache.KeyAt(layer, i) + kvHead * cacheHeadStride;
                        for (int t = 0; t < tn; t++)
                        {
                            int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                            if (i < endSeq)
                                scores[(long)t * stride + i] = SimdKernels.DotF32(
                                    batchQ + (long)(nBase + t) * qDim + h * headDim, kVec, headDim) * scale;
                        }
                    }

                    // ── Phase 2: per-token softmax over its own causal length ──
                    for (int t = 0; t < tn; t++)
                    {
                        int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                        SimdKernels.SoftmaxInPlace(scores + (long)t * stride, endSeq);
                    }

                    if (registerValues)
                    {
                        AccumulatePrefillValuesRegister8(valueRows, scores, stride, batchAttnOut,
                            nBase, tn, qDim, h * headDim, headDim, startPos, cache.Length);
                    }
                    else
                    {
                        for (int t = 0; t < tn; t++)
                        {
                            float* outHead = batchAttnOut + (long)(nBase + t) * qDim + h * headDim;
                            for (int d = 0; d < headDim; d++) outHead[d] = 0;
                        }

                        // Scalar/non-AVX fallback retains the original loop shape.
                        for (int i = 0; i < endSeqMax; i++)
                        {
                            float* vVec = cache.ValueAtHead(layer, i, kvHead);
                            for (int t = 0; t < tn; t++)
                            {
                                int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                                if (i >= endSeq) continue;
                                float* outHead = batchAttnOut + (long)(nBase + t) * qDim + h * headDim;
                                float w = scores[(long)t * stride + i];
                                if (Fma.IsSupported && headDim >= 8)
                                {
                                    var wv = Vector256.Create(w);
                                    int d = 0;
                                    for (; d + 8 <= headDim; d += 8)
                                    {
                                        var o = Avx.LoadVector256(outHead + d);
                                        var v = Avx.LoadVector256(vVec + d);
                                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                                    }
                                    for (; d < headDim; d++)
                                        outHead[d] += w * vVec[d];
                                }
                                else
                                {
                                    for (int d = 0; d < headDim; d++)
                                        outHead[d] += w * vVec[d];
                                }
                            }
                        }
                    }

                }
            }
            finally
            {
                NativeMemory.Free(valueRows);
                NativeMemory.Free(scores);
            }
        });
    }

    /// <summary>
    /// Attention against a BF16-store cache (<c>STINGRAY_KV_STORE=bf16</c>). Structural mirror of
    /// <see cref="PrefillCoreAttention"/>: same token tiling, same three phases, same causal bounds
    /// and the same score-then-softmax-then-weighted-V order — only the loads differ, widening 2-byte
    /// elements instead of reading 4-byte ones. Arithmetic stays fp32 throughout.
    /// </summary>
    /// <remarks>
    /// <para>Deliberately a separate method rather than a branch inside the hot loops. The two paths
    /// dereference different pointer types, and threading that through the tiled loops would put a
    /// predictable-but-real test in the innermost body of the engine's single hottest kernel. The
    /// duplication is the cheaper of the two costs, and the F32 path stays byte-for-byte as it was —
    /// which matters because it is the default and is covered by the parity suites.</para>
    ///
    /// <para><b>Not bit-identical to the F32 path, by construction</b> — the stored values have 8
    /// mantissa bits instead of 23. The reduction ORDER is identical (<c>DotF32Bf16</c> copies
    /// <c>DotF32</c>'s accumulator tree exactly), so the difference is attributable to storage
    /// precision alone. Perplexity is the gate, not bit-equality.</para>
    ///
    /// <para>The register-8 value microkernel is not used here. It is an F32 uop-count optimisation
    /// worth ~1.17x on prefill; this path exists for decode, where N=1 makes an 8-token microkernel
    /// degenerate and the bound is DRAM traffic rather than uops.</para>
    /// </remarks>
    private void PrefillCoreAttentionBf16(
        float* batchQ, PagedKvCache cache, int layer, int N, int startPos, float* batchAttnOut,
        int numHeads, int numKvHeads, int headDim, int qDim, int cacheHeadStride, int hpkg, float scale)
    {
        const int TokenTile = 64;

        Parallel.For(0, numHeads, h =>
        {
            int kvHead = h / hpkg;
            int maxSeqLen = startPos + N;
            int stride = maxSeqLen;
            float* scores = (float*)NativeMemory.AllocZeroed(
                (nuint)((long)TokenTile * stride * sizeof(float)));
            try
            {
                for (int nBase = 0; nBase < N; nBase += TokenTile)
                {
                    int tn = Math.Min(TokenTile, N - nBase);
                    int endSeqMax = Math.Min(startPos + nBase + tn, cache.Length);

                    // ── Phase 1: scores. Stream K once per tile. ──
                    for (int i = 0; i < endSeqMax; i++)
                    {
                        ushort* kVec = cache.Bf16KeyAt(layer, i) + kvHead * cacheHeadStride;
                        for (int t = 0; t < tn; t++)
                        {
                            int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                            if (i < endSeq)
                                scores[(long)t * stride + i] = SimdKernels.DotF32Bf16(
                                    batchQ + (long)(nBase + t) * qDim + h * headDim, kVec, headDim) * scale;
                        }
                    }

                    // ── Phase 2: per-token softmax over its own causal length ──
                    for (int t = 0; t < tn; t++)
                    {
                        int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                        SimdKernels.SoftmaxInPlace(scores + (long)t * stride, endSeq);
                    }

                    // ── Phase 3: weighted V. Stream V once per tile, same i-ascending order. ──
                    for (int t = 0; t < tn; t++)
                    {
                        float* outHead = batchAttnOut + (long)(nBase + t) * qDim + h * headDim;
                        for (int d = 0; d < headDim; d++) outHead[d] = 0;
                    }

                    for (int i = 0; i < endSeqMax; i++)
                    {
                        ushort* vVec = cache.Bf16ValueAtHead(layer, i, kvHead);
                        for (int t = 0; t < tn; t++)
                        {
                            int endSeq = Math.Min(startPos + nBase + t + 1, cache.Length);
                            if (i >= endSeq) continue;
                            SimdKernels.AccumulateScaledBf16(
                                batchAttnOut + (long)(nBase + t) * qDim + h * headDim,
                                vVec, scores[(long)t * stride + i], headDim);
                        }
                    }
                }
            }
            finally
            {
                NativeMemory.Free(scores);
            }
        });
    }

    /// <summary>
    /// Default AVX2 prefill path matching llama.cpp's CPU Flash-attention structure: 64x64 Q/KV
    /// tiles, online softmax, and the same 6x2 FP32 microkernel for both QK and probabilities*V.
    /// Set <c>STINGRAY_PREFILL_ATTN_FLASH64=0</c> to retain the materialised-score fallback.
    /// KV tiles are anchored at absolute position zero, so a query sees the same reduction order
    /// whether a prompt reaches this method in one call or several chunks.
    /// </summary>
    private static void PrefillFlashAttention64(
        float* batchQ, PagedKvCache cache, int layer, int tokenCount, int startPos,
        float* output, int numHeads, int numKvHeads, int qDim, int headDim, float scale)
    {
        const int Tile = 64;
        int queryTiles = (tokenCount + Tile - 1) / Tile;
        int headsPerKv = numHeads / numKvHeads;

        // Query-tile jobs improve isolated attention substantially, but the production result was
        // neutral at 900 tokens and only +2.0% best-of at 2400 tokens. Keep the simpler one-job-
        // per-head schedule as the default; this switch retains the verified experiment without
        // promoting a result that is still within this machine's end-to-end noise floor. Both
        // arms call the same tile worker, so the switch isolates scheduling from arithmetic.
        // KV-outer/query-inner reorder: packs each KV tile once per group of query tiles instead
        // of once per query tile. Default since it measured +1.6% alone / +4.0% with the SIMD
        // K-pack transpose, and it is bit-exact with the old schedule (Flash64KvOuterTests).
        if (Flash64KvOuterEnabled)
        {
            int groupTiles = s_flash64KvOuterGroupTiles;
            int maxQueries = Math.Min(tokenCount, groupTiles * Tile);
            var kvOuterScratch = new ThreadLocal<PrefillFlash64KvOuterScratch>(
                () => new PrefillFlash64KvOuterScratch(headDim, maxQueries), trackAllValues: true);
            try
            {
                Parallel.For(0, numHeads, h =>
                    ComputePrefillFlashAttention64KvOuterHead(batchQ, cache, layer, tokenCount, startPos,
                        output, qDim, headDim, scale, h, h / headsPerKv, kvOuterScratch.Value!, groupTiles));
            }
            finally
            {
                foreach (PrefillFlash64KvOuterScratch s in kvOuterScratch.Values) s.Dispose();
                kvOuterScratch.Dispose();
            }
            return;
        }

        bool useTileJobs = Environment.GetEnvironmentVariable(
            "STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS") == "1";
        if (!useTileJobs)
        {
            Parallel.For(0, numHeads, h =>
            {
                using var scratch = new PrefillFlash64Scratch(headDim);
                for (int nBase = 0; nBase < tokenCount; nBase += Tile)
                {
                    ComputePrefillFlashAttention64Tile(batchQ, cache, layer, tokenCount, startPos,
                        output, qDim, headDim, scale, h, h / headsPerKv, nBase, scratch);
                }
            });
            return;
        }

        var threadScratch = new ThreadLocal<PrefillFlash64Scratch>(
            () => new PrefillFlash64Scratch(headDim), trackAllValues: true);

        try
        {
            Parallel.For(0, numHeads * queryTiles, job =>
            {
                int h = job / queryTiles;
                int nBase = (job - h * queryTiles) * Tile;
                ComputePrefillFlashAttention64Tile(batchQ, cache, layer, tokenCount, startPos,
                    output, qDim, headDim, scale, h, h / headsPerKv, nBase, threadScratch.Value!);
            });
        }
        finally
        {
            foreach (PrefillFlash64Scratch scratch in threadScratch.Values) scratch.Dispose();
            threadScratch.Dispose();
        }
    }

    /// <summary>
    /// Default prefill-attention schedule since the 2x2 below; <c>STINGRAY_PREFILL_ATTN_KV_OUTER=0</c>
    /// restores the previous one. Same arithmetic
    /// as <see cref="ComputePrefillFlashAttention64Tile"/>, different loop order: KV tiles outside,
    /// query tiles inside, so a KV tile's K and V are packed <b>once</b> and reused by every query
    /// tile in the group instead of being repacked per query tile.
    ///
    /// <para><b>Why.</b> The default schedule is <c>for head → for queryTile → for kvTile: pack K;
    /// GEMM</c>, so identical key data is transposed into the pack buffer once per query tile — at
    /// 2048 tokens that is 32 repacks of the same bytes. This is a structural redundancy, not a
    /// constant factor: it is the one thing in this kernel whose cost falls with a reorder rather
    /// than with a faster instruction.</para>
    ///
    /// <para><b>Why it should be bit-exact.</b> Each query row still consumes KV tiles in ascending
    /// order, so its online-softmax accumulator sees exactly the sequence it saw before — the
    /// reorder is a loop interchange, not a reassociation. Two details preserve that. First,
    /// <c>valid</c> is still derived from the query tile's own causal limit, so a group-packed K
    /// tile that extends past a given query tile's reach has those columns zeroed by the existing
    /// clear before P·V, contributing exactly <c>0 × v</c>. Second, a query tile that cannot reach
    /// the current KV tile at all is skipped, which is precisely the iteration the old loop never
    /// ran. <c>TileJobs_MatchHeadJobs_BitExactly</c> is therefore a valid gate for this path.</para>
    ///
    /// <para><b>MEASURED: this helps.</b> SmolLM2-1.7B Q4_K_M, 1550-token prefill, headDim 64,
    /// 12 logical CPUs, 6 interleaved rounds per cell, as a 2×2 against the K-pack transpose
    /// (<c>STINGRAY_CPU_KPACK_SIMD</c>), best-of-6 / median t/s:</para>
    /// <code>
    ///   kpack   kv-outer    best   median      vs baseline (best)
    ///   scalar  off        148.3    143.7      baseline
    ///   scalar  ON         150.6    147.2      +1.6%
    ///   SIMD    off        152.8    148.7      +3.0%
    ///   SIMD    ON         154.2    149.9      +4.0%
    /// </code>
    /// <para>The two are roughly additive, so they attack overlapping but not identical cost: the
    /// SIMD transpose makes each pack cheaper, this reorder performs 7/8 fewer of them. The
    /// baseline is worst on best, median AND worst-case, which is what makes this more than a
    /// directional hint.</para>
    ///
    /// <para><b>A superseded earlier reading is recorded here deliberately.</b> A first 4-round
    /// A/B measured only the SIMD-on row (152.8 vs 154.2 — about +0.9%) and reported it as "no
    /// gain", and an op-count argument was offered to explain the null: the packs are ~8,192
    /// element copies against ~524,000 GEMM MACs, so ~1.5% of the tile's work. That argument is
    /// wrong, and the way it is wrong is worth keeping. It compares OPERATIONS, but the scalar
    /// pack walks a column — one float per KV row, a row-sized stride between touches, the access
    /// pattern that defeats prefetch — so its share of TIME far exceeds its share of ops. An
    /// op-count Amdahl bound is not a time bound for memory-latency-bound work.</para>
    ///
    /// <para><b>Why grouped rather than whole-sequence.</b> Holding running max/sum and the output
    /// accumulator live for every query tile at once costs <c>tokenCount × headDim</c> floats per
    /// thread — about 2 MB per thread at 8192 tokens, times a thread per core. Grouping bounds that
    /// to <c>groupTiles × 64 × headDim</c> (256 KB at the default 8 tiles and headDim 64, which
    /// stays L2-resident) while still amortising the K-pack 8-fold, capturing most of an available
    /// 32-fold saving for a fraction of the footprint.</para>
    /// </summary>
    private static void ComputePrefillFlashAttention64KvOuterHead(
        float* batchQ, PagedKvCache cache, int layer, int tokenCount, int startPos,
        float* output, int qDim, int headDim, float scale, int h, int kvHead,
        PrefillFlash64KvOuterScratch scratch, int groupTiles)
    {
        const int Tile = 64;
        bool bf16 = cache.IsBf16Store;
        int queryTiles = (tokenCount + Tile - 1) / Tile;

        for (int g0 = 0; g0 < queryTiles; g0 += groupTiles)
        {
            int gTiles = Math.Min(groupTiles, queryTiles - g0);
            int qStart = g0 * Tile;
            int qCount = Math.Min(tokenCount - qStart, gTiles * Tile);

            for (int t = 0; t < qCount; t++)
            {
                scratch.RunningMax[t] = float.NegativeInfinity;
                scratch.RunningSum[t] = 0f;
                Buffer.MemoryCopy(batchQ + (long)(qStart + t) * qDim + h * headDim,
                    scratch.QPack + (long)t * headDim, headDim * sizeof(float), headDim * sizeof(float));
            }
            new Span<float>(scratch.Accumulator, qCount * headDim).Clear();

            int groupEnd = Math.Min(startPos + qStart + qCount, cache.Length);
            for (int kvBase = 0; kvBase < groupEnd; kvBase += Tile)
            {
                int kLen = Math.Min(Tile, groupEnd - kvBase);

                // ── Pack K and V ONCE for this KV tile (the whole point of the reorder) ──
                if (bf16)
                {
                    for (int j = 0; j < kLen; j++)
                        scratch.Bf16KeyRows[j] = cache.Bf16KeyAt(layer, kvBase + j) + kvHead * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        float* packedRow = scratch.KPack + d * Tile;
                        int j = 0;
                        for (; j < kLen; j++) packedRow[j] = SimdKernels.Bf16ToF32(scratch.Bf16KeyRows[j][d]);
                        for (; j < Tile; j++) packedRow[j] = 0f;
                    }
                }
                else
                {
                    for (int j = 0; j < kLen; j++)
                        scratch.KeyRows[j] = cache.KeyAt(layer, kvBase + j) + kvHead * headDim;
                    int jFull = 0;
                    if (SimdKernels.KPackSimdEnabled && Avx.IsSupported && (headDim & 7) == 0)
                    {
                        jFull = kLen & ~7;
                        for (int j0 = 0; j0 < jFull; j0 += 8)
                            for (int d0 = 0; d0 < headDim; d0 += 8)
                                SimdKernels.TransposeBlock8x8(
                                    scratch.KeyRows + j0, d0, scratch.KPack + (long)d0 * Tile + j0, Tile);
                    }
                    for (int d = 0; d < headDim; d++)
                    {
                        float* packedRow = scratch.KPack + d * Tile;
                        int j = jFull;
                        for (; j < kLen; j++) packedRow[j] = scratch.KeyRows[j][d];
                        for (; j < Tile; j++) packedRow[j] = 0f;
                    }
                }

                if (bf16)
                    for (int j = 0; j < kLen; j++)
                        SimdKernels.WidenBf16ToF32(cache.Bf16ValueAtHead(layer, kvBase + j, kvHead),
                            scratch.VPack + j * headDim, headDim);
                else
                    for (int j = 0; j < kLen; j++)
                        Buffer.MemoryCopy(cache.ValueAtHead(layer, kvBase + j, kvHead),
                            scratch.VPack + j * headDim, headDim * sizeof(float), headDim * sizeof(float));
                new Span<float>(scratch.VPack + kLen * headDim, (Tile - kLen) * headDim).Clear();

                // ── Every query tile in the group consumes the packed K/V ──
                for (int qt = 0; qt < gTiles; qt++)
                {
                    int nBase = qStart + qt * Tile;
                    int tn = Math.Min(Tile, tokenCount - nBase);
                    if (tn <= 0) break;
                    // Exactly the iterations the per-query-tile loop never ran: this tile's causal
                    // reach ends at or before this KV tile, so it has no valid key here.
                    if (kvBase >= Math.Min(startPos + nBase + tn, cache.Length)) continue;

                    int qOff = nBase - qStart;
                    float* qPack = scratch.QPack + (long)qOff * headDim;
                    float* acc = scratch.Accumulator + (long)qOff * headDim;

                    if (headDim != Tile || s_flash64StridedGemm)
                        SimdKernels.GemmF32_6x2(qPack, scratch.KPack, scratch.Scores,
                            tn, headDim, Tile, headDim, Tile, Tile);
                    else
                        SimdKernels.GemmF32_64x64_6x2(qPack, scratch.KPack, scratch.Scores, tn);

                    for (int t = 0; t < tn; t++)
                    {
                        float* row = scratch.Scores + t * Tile;
                        int valid = Math.Clamp(startPos + nBase + t + 1 - kvBase, 0, kLen);
                        if (valid == 0)
                        {
                            new Span<float>(row, Tile).Clear();
                            continue;
                        }

                        float tileMax = SimdKernels.ScaleAndMaxF32InPlace(row, valid, scale);
                        float oldMax = scratch.RunningMax[qOff + t];
                        float newMax = MathF.Max(oldMax, tileMax);
                        float rescale = float.IsNegativeInfinity(oldMax) ? 0f : MathF.Exp(oldMax - newMax);
                        float tileSum = SimdKernels.ExpMinusMaxSumInPlace(row, valid, newMax);
                        new Span<float>(row + valid, Tile - valid).Clear();
                        scratch.RunningMax[qOff + t] = newMax;
                        scratch.RunningSum[qOff + t] = scratch.RunningSum[qOff + t] * rescale + tileSum;

                        if (rescale != 1f)
                        {
                            var rescaleV = Vector256.Create(rescale);
                            float* accRow = acc + (long)t * headDim;
                            for (int d = 0; d < headDim; d += 8)
                                Avx.Store(accRow + d, Avx.Multiply(Avx.LoadVector256(accRow + d), rescaleV));
                        }
                    }

                    if (headDim != Tile || s_flash64StridedGemm)
                        SimdKernels.GemmF32_6x2(scratch.Scores, scratch.VPack, acc,
                            tn, Tile, headDim, Tile, headDim, headDim, accumulate: true);
                    else
                        SimdKernels.GemmF32_64x64_6x2(scratch.Scores, scratch.VPack, acc, tn, accumulate: true);
                }
            }

            for (int t = 0; t < qCount; t++)
            {
                float* outHead = output + (long)(qStart + t) * qDim + h * headDim;
                float* accRow = scratch.Accumulator + (long)t * headDim;
                var inv = Vector256.Create(1f / scratch.RunningSum[t]);
                for (int d = 0; d < headDim; d += 8)
                    Avx.Store(outHead + d, Avx.Multiply(Avx.LoadVector256(accRow + d), inv));
            }
        }
    }

    private static void ComputePrefillFlashAttention64Tile(
        float* batchQ, PagedKvCache cache, int layer, int tokenCount, int startPos,
        float* output, int qDim, int headDim, float scale, int h, int kvHead, int nBase,
        PrefillFlash64Scratch scratch)
    {
        const int Tile = 64;
        int tn = Math.Min(Tile, tokenCount - nBase);
        bool bf16 = cache.IsBf16Store;

        for (int t = 0; t < tn; t++)
        {
            scratch.RunningMax[t] = float.NegativeInfinity;
            scratch.RunningSum[t] = 0f;
            Buffer.MemoryCopy(batchQ + (long)(nBase + t) * qDim + h * headDim,
                scratch.QPack + t * headDim, headDim * sizeof(float), headDim * sizeof(float));
        }
        new Span<float>(scratch.Accumulator, tn * headDim).Clear();

        int endSeqMax = Math.Min(startPos + nBase + tn, cache.Length);
        for (int kvBase = 0; kvBase < endSeqMax; kvBase += Tile)
        {
            int kLen = Math.Min(Tile, endSeqMax - kvBase);
            // BF16 pages are widened HERE, once per tile, into the same F32 pack the GEMM already
            // consumes 64 times — so the widen amortises 64-fold and every kernel below this point
            // is bit-for-bit the F32 one. (The opposite choice, widening on each streaming read,
            // is right for decode and wrong here; see SimdKernels.WidenBf16ToF32.)
            if (bf16)
            {
                for (int j = 0; j < kLen; j++)
                    scratch.Bf16KeyRows[j] = cache.Bf16KeyAt(layer, kvBase + j) + kvHead * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float* packedRow = scratch.KPack + d * Tile;
                    int j = 0;
                    for (; j < kLen; j++) packedRow[j] = SimdKernels.Bf16ToF32(scratch.Bf16KeyRows[j][d]);
                    for (; j < Tile; j++) packedRow[j] = 0f;
                }
            }
            else
            {
                for (int j = 0; j < kLen; j++)
                    scratch.KeyRows[j] = cache.KeyAt(layer, kvBase + j) + kvHead * headDim;

                // K-pack is a transpose: [key][dim] in the cache becomes [dim][key] for the GEMM.
                // Done scalar it is headDim*kLen single-float copies whose reads walk a column —
                // one float per KV row, a whole row's stride between touches. The 8x8 AVX block
                // form reads each source row as one 32-byte load instead, and is bit-identical
                // because a transpose only moves floats. Full 8-key blocks go through the vector
                // path; the ragged key tail and the zero-fill out to Tile stay scalar.
                int jFull = 0;
                if (SimdKernels.KPackSimdEnabled && Avx.IsSupported && (headDim & 7) == 0)
                {
                    jFull = kLen & ~7;
                    for (int j0 = 0; j0 < jFull; j0 += 8)
                        for (int d0 = 0; d0 < headDim; d0 += 8)
                            SimdKernels.TransposeBlock8x8(
                                scratch.KeyRows + j0, d0, scratch.KPack + (long)d0 * Tile + j0, Tile);
                }
                for (int d = 0; d < headDim; d++)
                {
                    float* packedRow = scratch.KPack + d * Tile;
                    int j = jFull;
                    for (; j < kLen; j++) packedRow[j] = scratch.KeyRows[j][d];
                    for (; j < Tile; j++) packedRow[j] = 0f;
                }
            }
            // Q*Kt. Tile and HeadDim are both the compile-time 64 here, so the strided kernel
            // runs the identical shape and is bit-identical to the hardcoded one
            // (GemmF32StridedParityTests pins that); it is measurably faster all the same, because
            // it hoists the six row base pointers out of the j/k loops instead of recomputing
            // indices. Gated so the two can be interleaved in one binary.
            if (headDim != Tile || s_flash64StridedGemm)
                SimdKernels.GemmF32_6x2(scratch.QPack, scratch.KPack, scratch.Scores,
                    tn, headDim, Tile, headDim, Tile, Tile);
            else
                SimdKernels.GemmF32_64x64_6x2(scratch.QPack, scratch.KPack, scratch.Scores, tn);

            for (int t = 0; t < tn; t++)
            {
                float* row = scratch.Scores + t * Tile;
                int valid = Math.Clamp(startPos + nBase + t + 1 - kvBase, 0, kLen);
                if (valid == 0)
                {
                    new Span<float>(row, Tile).Clear();
                    continue;
                }

                float tileMax = SimdKernels.ScaleAndMaxF32InPlace(row, valid, scale);
                float oldMax = scratch.RunningMax[t];
                float newMax = MathF.Max(oldMax, tileMax);
                float rescale = float.IsNegativeInfinity(oldMax) ? 0f : MathF.Exp(oldMax - newMax);
                float tileSum = SimdKernels.ExpMinusMaxSumInPlace(row, valid, newMax);
                new Span<float>(row + valid, Tile - valid).Clear();
                scratch.RunningMax[t] = newMax;
                scratch.RunningSum[t] = scratch.RunningSum[t] * rescale + tileSum;

                if (rescale != 1f)
                {
                    var rescaleV = Vector256.Create(rescale);
                    float* acc = scratch.Accumulator + t * headDim;
                    for (int d = 0; d < headDim; d += 8)
                        Avx.Store(acc + d, Avx.Multiply(Avx.LoadVector256(acc + d), rescaleV));
                }
            }

            if (bf16)
                for (int j = 0; j < kLen; j++)
                    SimdKernels.WidenBf16ToF32(cache.Bf16ValueAtHead(layer, kvBase + j, kvHead),
                        scratch.VPack + j * headDim, headDim);
            else
                for (int j = 0; j < kLen; j++)
                    Buffer.MemoryCopy(cache.ValueAtHead(layer, kvBase + j, kvHead),
                        scratch.VPack + j * headDim, headDim * sizeof(float), headDim * sizeof(float));
            new Span<float>(scratch.VPack + kLen * headDim, (Tile - kLen) * headDim).Clear();
            // P*V: the transposed shape of the pair above (k = keys, n = head dim).
            if (headDim != Tile || s_flash64StridedGemm)
                SimdKernels.GemmF32_6x2(scratch.Scores, scratch.VPack, scratch.Accumulator,
                    tn, Tile, headDim, Tile, headDim, headDim, accumulate: true);
            else
                SimdKernels.GemmF32_64x64_6x2(
                    scratch.Scores, scratch.VPack, scratch.Accumulator, tn, accumulate: true);
        }

        for (int t = 0; t < tn; t++)
        {
            float* outHead = output + (long)(nBase + t) * qDim + h * headDim;
            float* acc = scratch.Accumulator + t * headDim;
            var inv = Vector256.Create(1f / scratch.RunningSum[t]);
            for (int d = 0; d < headDim; d += 8)
                Avx.Store(outHead + d, Avx.Multiply(Avx.LoadVector256(acc + d), inv));
        }
    }

    /// <summary>
    /// Scratch for <see cref="ComputePrefillFlashAttention64KvOuterHead"/>. Differs from
    /// <see cref="PrefillFlash64Scratch"/> in one way that matters: running max/sum and the output
    /// accumulator are sized for a whole GROUP of query tiles rather than one, because the reorder
    /// keeps them all live while a KV tile is resident. K/V packs and the score tile stay
    /// single-tile — those are what the reorder is amortising, not what it multiplies.
    /// </summary>
    private sealed class PrefillFlash64KvOuterScratch : IDisposable
    {
        private const int Tile = 64;
        public readonly float* Scores = (float*)NativeMemory.AlignedAlloc(Tile * Tile * sizeof(float), 64);
        public readonly float* KPack;
        public readonly float* VPack;
        public readonly float* QPack;
        public readonly float* Accumulator;
        public readonly float* RunningMax;
        public readonly float* RunningSum;
        public readonly float** KeyRows = (float**)NativeMemory.AlignedAlloc((nuint)(Tile * sizeof(nint)), 64);
        public readonly ushort** Bf16KeyRows = (ushort**)NativeMemory.AlignedAlloc((nuint)(Tile * sizeof(nint)), 64);

        public PrefillFlash64KvOuterScratch(int headDim, int maxQueries)
        {
            nuint tileElems = (nuint)(Tile * headDim);
            KPack = (float*)NativeMemory.AlignedAlloc(tileElems * sizeof(float), 64);
            VPack = (float*)NativeMemory.AlignedAlloc(tileElems * sizeof(float), 64);

            nuint groupElems = (nuint)((long)maxQueries * headDim);
            QPack = (float*)NativeMemory.AlignedAlloc(groupElems * sizeof(float), 64);
            Accumulator = (float*)NativeMemory.AlignedAlloc(groupElems * sizeof(float), 64);
            RunningMax = (float*)NativeMemory.AlignedAlloc((nuint)maxQueries * sizeof(float), 64);
            RunningSum = (float*)NativeMemory.AlignedAlloc((nuint)maxQueries * sizeof(float), 64);
        }

        public void Dispose()
        {
            NativeMemory.AlignedFree(Bf16KeyRows);
            NativeMemory.AlignedFree(KeyRows);
            NativeMemory.AlignedFree(RunningSum);
            NativeMemory.AlignedFree(RunningMax);
            NativeMemory.AlignedFree(Accumulator);
            NativeMemory.AlignedFree(QPack);
            NativeMemory.AlignedFree(VPack);
            NativeMemory.AlignedFree(KPack);
            NativeMemory.AlignedFree(Scores);
        }
    }

    private sealed class PrefillFlash64Scratch : IDisposable
    {
        private const int Tile = 64;
        public readonly float* Scores = (float*)NativeMemory.AlignedAlloc(Tile * Tile * sizeof(float), 64);
        public readonly float* Accumulator;
        public readonly float* RunningMax = (float*)NativeMemory.AlignedAlloc(64 * sizeof(float), 64);
        public readonly float* RunningSum = (float*)NativeMemory.AlignedAlloc(64 * sizeof(float), 64);
        public readonly float* QPack;
        public readonly float* KPack;
        public readonly float* VPack;
        public readonly float** KeyRows = (float**)NativeMemory.AlignedAlloc((nuint)(64 * sizeof(nint)), 64);
        /// <summary>BF16-store counterpart of <see cref="KeyRows"/>. Only one of the two is ever
        /// populated for a given cache; both are allocated because the scratch is pooled per thread
        /// and 512 bytes is not worth a conditional allocation.</summary>
        public readonly ushort** Bf16KeyRows = (ushort**)NativeMemory.AlignedAlloc((nuint)(64 * sizeof(nint)), 64);

        public PrefillFlash64Scratch(int headDim)
        {
            nuint elements = (nuint)(Tile * headDim);
            Accumulator = (float*)NativeMemory.AlignedAlloc(elements * sizeof(float), 64);
            QPack = (float*)NativeMemory.AlignedAlloc(elements * sizeof(float), 64);
            KPack = (float*)NativeMemory.AlignedAlloc(elements * sizeof(float), 64);
            VPack = (float*)NativeMemory.AlignedAlloc(elements * sizeof(float), 64);
        }

        public void Dispose()
        {
            NativeMemory.AlignedFree(Bf16KeyRows);
            NativeMemory.AlignedFree(KeyRows);
            NativeMemory.AlignedFree(VPack);
            NativeMemory.AlignedFree(KPack);
            NativeMemory.AlignedFree(QPack);
            NativeMemory.AlignedFree(RunningSum);
            NativeMemory.AlignedFree(RunningMax);
            NativeMemory.AlignedFree(Accumulator);
            NativeMemory.AlignedFree(Scores);
        }
    }

    /// <summary>
    /// Weighted-V microkernel for prefill attention. Holds eight tokens' eight-float output chunks
    /// in YMM registers across the ascending KV loop. Every lane receives the same FMA sequence as
    /// the former memory-accumulator loop, so chunked and unchunked prefill remain bit-identical.
    /// </summary>
    private static void AccumulatePrefillValuesRegister8(float** valueRows, float* scores, int stride,
        float* output, int nBase, int tokenCount, int qDim, int headOffset, int headDim,
        int startPos, int cacheLength)
    {
        for (int tBase = 0; tBase < tokenCount; tBase += 8)
        {
            int active = Math.Min(8, tokenCount - tBase);
            int firstEnd = Math.Min(startPos + nBase + tBase + 1, cacheLength);
            int lastEnd = Math.Min(firstEnd + active - 1, cacheLength);

            for (int d = 0; d < headDim; d += 8)
            {
                var a0 = Vector256<float>.Zero;
                var a1 = Vector256<float>.Zero;
                var a2 = Vector256<float>.Zero;
                var a3 = Vector256<float>.Zero;
                var a4 = Vector256<float>.Zero;
                var a5 = Vector256<float>.Zero;
                var a6 = Vector256<float>.Zero;
                var a7 = Vector256<float>.Zero;

                if (active == 8)
                {
                    for (int i = 0; i < firstEnd; i++)
                    {
                        var v = Avx.LoadVector256(valueRows[i] + d);
                        a0 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 0) * stride + i]), v, a0);
                        a1 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 1) * stride + i]), v, a1);
                        a2 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 2) * stride + i]), v, a2);
                        a3 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 3) * stride + i]), v, a3);
                        a4 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 4) * stride + i]), v, a4);
                        a5 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 5) * stride + i]), v, a5);
                        a6 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 6) * stride + i]), v, a6);
                        a7 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 7) * stride + i]), v, a7);
                    }
                }
                else
                {
                    for (int i = 0; i < firstEnd; i++)
                    {
                        var v = Avx.LoadVector256(valueRows[i] + d);
                        a0 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 0) * stride + i]), v, a0);
                        if (active > 1) a1 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 1) * stride + i]), v, a1);
                        if (active > 2) a2 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 2) * stride + i]), v, a2);
                        if (active > 3) a3 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 3) * stride + i]), v, a3);
                        if (active > 4) a4 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 4) * stride + i]), v, a4);
                        if (active > 5) a5 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 5) * stride + i]), v, a5);
                        if (active > 6) a6 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 6) * stride + i]), v, a6);
                    }
                }

                // A full causal group differs only in its final seven positions. These FMAs remain
                // ascending for every accumulator; this is loop interchange, not reassociation.
                for (int i = firstEnd; i < lastEnd; i++)
                {
                    var v = Avx.LoadVector256(valueRows[i] + d);
                    int firstActive = i - firstEnd + 1;
                    if (firstActive <= 1 && active > 1) a1 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 1) * stride + i]), v, a1);
                    if (firstActive <= 2 && active > 2) a2 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 2) * stride + i]), v, a2);
                    if (firstActive <= 3 && active > 3) a3 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 3) * stride + i]), v, a3);
                    if (firstActive <= 4 && active > 4) a4 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 4) * stride + i]), v, a4);
                    if (firstActive <= 5 && active > 5) a5 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 5) * stride + i]), v, a5);
                    if (firstActive <= 6 && active > 6) a6 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 6) * stride + i]), v, a6);
                    if (firstActive <= 7 && active > 7) a7 = Fma.MultiplyAdd(Vector256.Create(scores[(long)(tBase + 7) * stride + i]), v, a7);
                }

                Avx.Store(output + (long)(nBase + tBase + 0) * qDim + headOffset + d, a0);
                if (active > 1) Avx.Store(output + (long)(nBase + tBase + 1) * qDim + headOffset + d, a1);
                if (active > 2) Avx.Store(output + (long)(nBase + tBase + 2) * qDim + headOffset + d, a2);
                if (active > 3) Avx.Store(output + (long)(nBase + tBase + 3) * qDim + headOffset + d, a3);
                if (active > 4) Avx.Store(output + (long)(nBase + tBase + 4) * qDim + headOffset + d, a4);
                if (active > 5) Avx.Store(output + (long)(nBase + tBase + 5) * qDim + headOffset + d, a5);
                if (active > 6) Avx.Store(output + (long)(nBase + tBase + 6) * qDim + headOffset + d, a6);
                if (active > 7) Avx.Store(output + (long)(nBase + tBase + 7) * qDim + headOffset + d, a7);
            }
        }
    }

    // ================================================================
    //  Attention
    // ================================================================

    private void Attention(PagedKvCache cache, int layer, int position)
        => Attention(cache, layer, layer, position, _headDim, windowSize: -1, _numKvHeads);

    /// <summary>
    /// Multi-head attention with optional per-layer head dim, KV-source aliasing, and
    /// sliding-window bound. <paramref name="readLayer"/> is the layer whose K/V pages
    /// to read (== <paramref name="ownLayer"/> for non-shared layers; the source layer
    /// when KV is aliased). <paramref name="windowSize"/> &gt; 0 restricts the score and
    /// V-aggregation loops to the last <paramref name="windowSize"/> positions.
    /// <paramref name="kvHeads"/> is the active layer's KV head count (Gemma 4 12B:
    /// 8 GQA on SWA layers, 1 MQA on the k_eq_v global layers) — it can differ from the
    /// model-level <see cref="_numKvHeads"/>, so the head→KV-group ratio is computed here.
    /// </summary>
    private void Attention(PagedKvCache cache, int readLayer, int ownLayer, int position,
        int hd, int windowSize, int kvHeads)
    {
        // After SnapKV eviction (issue #51), the absolute position keeps
        // growing while the cache only stores `cache.Length` slots — `position`
        // would overshoot. The prefill loop increments cache.Length before
        // calling Attention (so position+1 == cache.Length); the decode loop
        // increments after (so position+1 == cache.Length+1). Clamping to
        // cache.Length+1 keeps the old answer for both prefill and the
        // un-evicted decode case while bounding the read to the actually
        // stored slots once eviction has shrunk the cache.
        _ = ownLayer;
        int endSeq = Math.Min(position + 1, cache.Length + 1);
        int startSeq = windowSize > 0 ? Math.Max(0, endSeq - windowSize) : 0;
        // Gemma 4 uses self.scaling = 1.0 (no pre-attention scaling); other archs
        // use 1/sqrt(head_dim). See llama.cpp src/models/gemma4.cpp:11
        //   hparams.f_attention_scale = 1.0f
        float scale = _layerHeadDim is not null ? 1.0f : 1.0f / MathF.Sqrt(hd);
        // Head→KV-group ratio for the ACTIVE layer (kvHeads, not the model-level
        // _numKvHeads): Gemma 4 12B global layers are MQA (kvHeads=1 → all _numHeads map
        // to KV head 0), SWA layers GQA (kvHeads=8). For non-per-layer models kvHeads ==
        // _numKvHeads so hpkg == _headsPerKvGroup.
        int ctxLen = _ctxLen; int hpkg = _numHeads / kvHeads;
        // The per-layer K/V stride now lives in PagedKvCache (see its layerHeadDim parameter), so
        // both the row-major K reads below and the transposed V reads agree on it. This method
        // used to compute a slotStride here and discard it, which left the V region striding at
        // the model-level head_dim while K strided at the layer's — the Gemma 4 KV-head bug.
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;
        int rl = readLayer; int hdLocal = hd; int startLocal = startSeq;

        int scoreLenAll = endSeq - startLocal;
        int numHeadsLocal = _numHeads;

        // ── Score pass, parallelised over POSITION TILES rather than heads ──
        // Parallelising over heads makes head h read bytes [h*hd, h*hd+hd) of every KV row, i.e. a
        // stride equal to the whole row (numKvHeads*headDim floats — 8 KB on this model). Strides
        // beyond a page are not prefetched, so every read exposed full memory latency; decode was
        // achieving ~22 GB/s against a measured 36.8 GB/s ceiling. Walking positions instead lets
        // each KV row be read contiguously while all heads consume it, and the query vectors
        // (numHeads*headDim floats) are small enough to stay resident across the whole tile.
        //
        // Bit-identical: each score is the same dot of the same operands, and only the order in
        // which independent (head, position) pairs are computed changes.
        const int PosTile = 64;
        int posTiles = (scoreLenAll + PosTile - 1) / PosTile;
        // BF16-store caches hold 2-byte pages; the dtype is fixed for the cache's lifetime, so this
        // is hoisted entirely out of the position and head loops rather than tested per element.
        bool bf16 = cache.IsBf16Store;
        if (posTiles > 1)
        {
            Parallel.For(0, posTiles, ti =>
            {
                int i0 = ti * PosTile;
                int i1 = Math.Min(i0 + PosTile, scoreLenAll);
                if (bf16)
                {
                    for (int i = i0; i < i1; i++)
                    {
                        ushort* kRow = cache.Bf16KeyAt(rl, startLocal + i);
                        for (int hh = 0; hh < numHeadsLocal; hh++)
                            scores[(long)hh * ctxLen + i] =
                                SimdKernels.DotF32Bf16(q + hh * hdLocal, kRow + (hh / hpkg) * hdLocal, hdLocal) * scale;
                    }
                    return;
                }
                for (int i = i0; i < i1; i++)
                {
                    float* kRow = cache.KeyAt(rl, startLocal + i);
                    for (int hh = 0; hh < numHeadsLocal; hh++)
                        scores[(long)hh * ctxLen + i] =
                            SimdKernels.DotF32(q + hh * hdLocal, kRow + (hh / hpkg) * hdLocal, hdLocal) * scale;
                }
            });
        }
        else if (bf16)
        {
            for (int i = 0; i < scoreLenAll; i++)
            {
                ushort* kRow = cache.Bf16KeyAt(rl, startLocal + i);
                for (int hh = 0; hh < numHeadsLocal; hh++)
                    scores[(long)hh * ctxLen + i] =
                        SimdKernels.DotF32Bf16(q + hh * hdLocal, kRow + (hh / hpkg) * hdLocal, hdLocal) * scale;
            }
        }
        else
        {
            for (int i = 0; i < scoreLenAll; i++)
            {
                float* kRow = cache.KeyAt(rl, startLocal + i);
                for (int hh = 0; hh < numHeadsLocal; hh++)
                    scores[(long)hh * ctxLen + i] =
                        SimdKernels.DotF32(q + hh * hdLocal, kRow + (hh / hpkg) * hdLocal, hdLocal) * scale;
            }
        }

        // Softmax and the weighted-V sum stay parallel over heads: the V accumulation is a
        // per-head reduction over ascending i, so splitting it by position would need per-thread
        // partials and would change the accumulation order (and the result).
        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* outHead = attnOut + h * hdLocal;
            float* headScores = scores + (long)h * ctxLen;

            int scoreLen = scoreLenAll;

            SimdKernels.SoftmaxInPlace(headScores, scoreLen);

            for (int d = 0; d < hdLocal; d++) outHead[d] = 0;

            if (bf16)
            {
                for (int i = 0; i < scoreLen; i++)
                    SimdKernels.AccumulateScaledBf16(
                        outHead, cache.Bf16ValueAtHead(rl, startLocal + i, kvHead), headScores[i], hdLocal);
                return;
            }

            for (int i = 0; i < scoreLen; i++)
            {
                int t = startLocal + i;
                float* vVec = cache.ValueAtHead(rl, t, kvHead);
                float w = headScores[i];
                if (Fma.IsSupported && hdLocal >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hdLocal; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hdLocal; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hdLocal; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

    // ================================================================
    //  TurboQuant Attention
    // ================================================================

    private void TqAttention(int layer, int position)
    {
        var tq = _tqKvCache!;
        // After SnapKV (issue #60) eviction the absolute position keeps
        // growing while the TQ cache only stores `tq.Length` slots. The
        // Forward decode path's Append runs before IncrementPosition so the
        // new K/V is at slot tq.Length (and visible via Fp32KeyAt(position=
        // tq.Length)), hence the `+1` — mirrors PagedKvCache.Attention.
        // Pre-eviction position+1 == tq.Length so the clamp is a no-op.
        int seqLen = Math.Min(position + 1, tq.Length + 1);
        int tqLen = tq.GetTqLength(layer);
        int fp32Start = tqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int ctxLen = _ctxLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        int tqBlkSz = tq.TqBlockSize;
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;
        var rotated = _rotatedQuery; var decomp = _decompBuf;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * ctxLen;
            float* headRotated = rotated + h * hd;
            float* headDecomp = decomp + h * hd;

            // Rotate the query into the compressed-domain basis (Lloyd-Max:
            // per-head sign-flip + WHT; KVarN: plain WHT — issue #180).
            tq.RotateQuery(layer, kvHead,
                new ReadOnlySpan<float>(qHead, hd),
                new Span<float>(headRotated, hd));

            // K-scoring over the compressed region. Lloyd-Max (issue #34):
            // tile-walks full 32-position FastScan tiles through an i8-LUT
            // pshufb kernel and falls back to per-block DequantDot on the <32
            // staging tail. KVarN (issue #180): whole 128-token tiles via the
            // fused KVarNCompressor.KeyScores (no staging tail exists).
            tq.ComputeKScores(layer, kvHead, headRotated, scale, headScores);

            // Phase 1b: FP32 window positions
            for (int t = fp32Start; t < seqLen; t++)
            {
                float* kVec = tq.Fp32KeyAt(layer, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            // V-aggregation over the compressed region: tiles accumulate in the
            // rotated domain with ONE deferred inverse WHT per head (Lloyd-Max
            // adds a sign-flip; KVarN uses UnrotateOutput), then the FP32-window
            // loop below accumulates the recent positions on top in the
            // original (un-rotated) domain — the domain contract both codecs share.
            tq.ComputeVAggregation(layer, kvHead, headScores, outHead);

            for (int t = fp32Start; t < seqLen; t++)
            {
                float* vVec = tq.Fp32ValueAt(layer, t) + kvHead * hd;
                float w = headScores[t];
                if (Fma.IsSupported && hd >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hd; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

    // ================================================================
    //  Dense FFN (non-MoE)
    // ================================================================

    private void DenseFfn(int layer)
    {
        SimdKernels.MatVecDual(_ffnGate, _wGate[layer].DataPtr, _ffnUp, _wUp[layer].DataPtr,
            _normBuf, _intermDim, _embDim, _wGate[layer].DType, _wUp[layer].DType);
        if (_hp.FfnActivation == FfnActivation.GeluApprox)
            SimdKernels.GeluTanhMul(_ffnGate, _ffnUp, _ffnGate, _intermDim);
        else
            SimdKernels.SiLuMul(_ffnGate, _ffnUp, _intermDim);
        FusedMatVec(_hidden, _wDown[layer], _ffnGate, _embDim, _intermDim);
    }

    // ================================================================
    //  MoE FFN (Mixture of Experts)
    // ================================================================

    private void MoeFfn(int layer)
    {
        int numExperts = _hp.NumExperts;
        int numActive = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;

        // Step 1: Router — compute expert logits and select top-k
        FusedMatVec(_routerLogits, _wGateInp![layer], _normBuf, numExperts, _embDim);

        // Gating: sigmoid for Llama-4, softmax for others
        if (_hp.UseSigmoidGating)
            SimdKernels.SigmoidInPlace(_routerLogits, numExperts);
        else
            SimdKernels.SoftmaxInPlace(_routerLogits, numExperts);

        // Find top-k experts (for k=1, just argmax)
        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_routerLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);

        if (_traceRouters && (_traceRouterPos < 0 || _traceRouterPos == _currentPos))
        {
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var sb = new System.Text.StringBuilder(512);
            sb.Append("[router pos=").Append(_currentPos).Append(" L").Append(layer).Append(']');
            float wsum = 0;
            for (int i = 0; i < numActive; i++)
            {
                sb.Append(' ').Append(selectedExperts[i]).Append('=')
                  .Append(expertWeights[i].ToString("F4", inv));
                wsum += expertWeights[i];
            }
            sb.Append(" sum=").Append(wsum.ToString("F4", inv));
            Console.Error.WriteLine(sb.ToString());
        }

        // Step 2: Shared expert (runs on every token if present)
        // Shared expert uses the same dim as routed experts (ExpertIntermediateDim)
        if (_hp.HasSharedExpert)
        {
            FusedMatVec(_expertGate, _wGateShexp![layer], _normBuf, expertDim, _embDim);
            FusedMatVec(_expertUp, _wUpShexp![layer], _normBuf, expertDim, _embDim);
            SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
            FusedMatVec(_sharedOut, _wDownShexp![layer], _expertGate, _embDim, expertDim);
        }

        // Step 3: Selected expert(s) — sparse execution
        // Zero the output accumulator
        new Span<float>(_hidden, _embDim).Clear();

        for (int k = 0; k < numActive; k++)
        {
            int expertIdx = selectedExperts[k];
            float weight = expertWeights[k];

            // Expert weights are packed: all experts concatenated in one tensor.
            // Each expert's gate/up is [expertDim, embDim], down is [embDim, expertDim].
            // Expert slice offset in packed tensor: expertIdx * expertDim * (bytes per row)
            ExpertMatVec(_expertGate, _wGateExps![layer], expertIdx, expertDim, _embDim, _normBuf);
            ExpertMatVec(_expertUp, _wUpExps![layer], expertIdx, expertDim, _embDim, _normBuf);

            if (_hp.UseSigmoidGating)
            {
                // Llama-4: apply sigmoid weight before FFN (scale gate/up ≡ scaling input)
                SimdKernels.ScaleInPlace(_expertGate, weight, expertDim);
                SimdKernels.ScaleInPlace(_expertUp, weight, expertDim);
                weight = 1.0f;
            }

            SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
            ExpertMatVecDown(_hidden, _wDownExps![layer], expertIdx, _embDim, expertDim, _expertGate, weight);
        }

        // Step 4: Add shared expert output
        if (_hp.HasSharedExpert)
            SimdKernels.AddInPlace(_hidden, _sharedOut, _embDim);
    }

    /// <summary>
    /// Master switch for the batched MoE prefill FFN. Set <c>STINGRAY_MOE_BATCHED_PREFILL=0</c>
    /// to force MoE prompts back onto the per-token sequential trunk. Settable so parity tests can
    /// run both arms in one process (the sequential arm is the oracle, and re-launching to get it
    /// would make the comparison depend on process state rather than on this one flag).
    /// </summary>
    /// <summary>
    /// Force batched prefill for per-layer-head-dim models (Gemma 4). Default OFF, and it must
    /// stay off until the batched core grows sliding-window attention, per-layer KV head counts
    /// and KV-layer sharing — see the gate in <c>PrefillWithPerPositionLogits</c>. Exists so the
    /// remaining work can be exercised and measured, not as a supported configuration.
    /// </summary>
    private static readonly bool s_perLayerHeadDimPrefillForced =
        Environment.GetEnvironmentVariable("STINGRAY_PER_LAYER_HD_PREFILL") == "1";

    /// <summary>
    /// Whether prefill flash-64 uses <see cref="SimdKernels.GemmF32_6x2"/> (strided) instead of
    /// <see cref="SimdKernels.GemmF32_64x64_6x2"/> (shape-hardcoded). The two are bit-identical at
    /// this shape, so this is a pure speed switch with no numerics question attached — it exists
    /// only so both arms can be interleaved in one binary rather than compared across rebuilds.
    /// <c>STINGRAY_FLASH64_STRIDED_GEMM=0</c> restores the hardcoded kernel.
    /// </summary>
    /// <summary>
    /// KV-outer prefill-attention reorder. <b>On by default</b>; <c>STINGRAY_PREFILL_ATTN_KV_OUTER=0</c>
    /// restores the per-query-tile schedule. Measured at +1.6% alone and +4.0% combined with the
    /// SIMD K-pack transpose — see the 2×2 table on
    /// <see cref="ComputePrefillFlashAttention64KvOuterHead"/>. It is bit-exact with the old
    /// schedule (<c>Flash64KvOuterTests</c>), so the default carries no numerical risk; the cost is
    /// scratch, ~256 KB per thread instead of ~16 KB, because a group of query tiles stays live
    /// while a KV tile is resident.
    ///
    /// <para>Settable rather than a readonly env snapshot so a test can flip it inside one process
    /// and diff the two schedules against each other. Reading it only from the environment would
    /// have made the natural gate useless: the reorder short-circuits before the tile-jobs branch,
    /// so an env-configured run of the existing schedule-comparison test would put BOTH arms on
    /// this path and compare it with itself — a confident pass proving nothing.</para>
    /// </summary>
    /// <summary>
    /// Admits head dimensions 128/256 to the Flash-64 prefill path. <b>Off by default</b> — the
    /// widths are implemented but held back pending the parity decision documented at the gate.
    /// Settable rather than env-only so the decision can be measured at all: the comparison needs
    /// flash-on and flash-off logits from within one process, which an env snapshot read once at
    /// type-init cannot provide.
    /// </summary>
    internal static bool Flash64WideHeadDimsEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_PREFILL_ATTN_WIDE_HEADS") == "1";

    internal static bool Flash64KvOuterEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_PREFILL_ATTN_KV_OUTER") != "0";

    /// <summary>
    /// Query tiles held live per KV pack in the reordered path. Trades scratch footprint for
    /// K-pack amortisation: 8 tiles is 512 queries, ~256 KB of accumulator+Q at headDim 64.
    /// </summary>
    private static readonly int s_flash64KvOuterGroupTiles =
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_PREFILL_ATTN_KV_OUTER_TILES"),
            System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture, out int g) && g > 0 ? g : 8;

    private static readonly bool s_flash64StridedGemm =
        Environment.GetEnvironmentVariable("STINGRAY_FLASH64_STRIDED_GEMM") != "0";

    public static bool MoeBatchedPrefillEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_MOE_BATCHED_PREFILL") != "0";

    /// <summary>
    /// Whether this model may take the batched MoE prefill path instead of falling back to the
    /// per-token sequential trunk.
    ///
    /// <para>The excluded cases are all things the batched cores do not model at all, for dense
    /// models either — post-attention/post-FFW norms, per-layer output scale and PLE are applied
    /// only on <c>RunTrunk</c>. Admitting a MoE model that has them would produce a silent
    /// numerics divergence between chunked and unchunked prefill of the same prompt, so they stay
    /// on the sequential path. The router trace is excluded because it prints
    /// <c>_currentPos</c>, which the batched path does not advance per token; a trace that lied
    /// about position would be worse than no batching.</para>
    /// </summary>
    private bool MoeBatchedPrefillSupported =>
        MoeBatchedPrefillEnabled
        && _hp.IsMoE
        && _wGateInp is not null && _wGateExps is not null
        && _wUpExps is not null && _wDownExps is not null
        && _tqKvCache is null
        && !_traceRouters && !_traceNorms
        && _postAttnNorm is null && _postFfwNorm is null
        && _layerOutputScale is null && !_hp.HasPerLayerTokenEmbd;

    /// <summary>
    /// Grow the batched-MoE scratch to hold <paramref name="n"/> token rows. Buffers are kept
    /// across layers and chunks (the reuse distance is one layer) and released in
    /// <see cref="Dispose"/>.
    /// </summary>
    private void EnsureMoeBatchScratch(int n)
    {
        if (n <= _moeBatchCap) return;
        FreeMoeBatchScratch();

        int numExperts = _hp.NumExperts;
        int na = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;
        long pairs = (long)n * na;

        _moeBatchRouter   = (float*)NativeMemory.Alloc((nuint)((long)n * numExperts * sizeof(float)));
        _moeBatchSel      = (int*)  NativeMemory.Alloc((nuint)(pairs * sizeof(int)));
        _moeBatchWts      = (float*)NativeMemory.Alloc((nuint)(pairs * sizeof(float)));
        _moeExpStart      = (int*)  NativeMemory.Alloc((nuint)((numExperts + 1) * sizeof(int)));
        _moeExpCursor     = (int*)  NativeMemory.Alloc((nuint)(numExperts * sizeof(int)));
        _moeExpTokI       = (int*)  NativeMemory.Alloc((nuint)(pairs * sizeof(int)));
        _moeExpTokK       = (int*)  NativeMemory.Alloc((nuint)(pairs * sizeof(int)));
        _moeBatchGathered = (float*)NativeMemory.Alloc((nuint)((long)n * _embDim * sizeof(float)));
        _moeBatchGate     = (float*)NativeMemory.Alloc((nuint)((long)n * expertDim * sizeof(float)));
        _moeBatchUp       = (float*)NativeMemory.Alloc((nuint)((long)n * expertDim * sizeof(float)));
        _moeBatchDown     = (float*)NativeMemory.Alloc((nuint)(pairs * _embDim * sizeof(float)));
        _moeBatchCap = n;
    }

    private void FreeMoeBatchScratch()
    {
        if (_moeBatchCap == 0) return;
        NativeMemory.Free(_moeBatchRouter);   _moeBatchRouter = null;
        NativeMemory.Free(_moeBatchSel);      _moeBatchSel = null;
        NativeMemory.Free(_moeBatchWts);      _moeBatchWts = null;
        NativeMemory.Free(_moeExpStart);      _moeExpStart = null;
        NativeMemory.Free(_moeExpCursor);     _moeExpCursor = null;
        NativeMemory.Free(_moeExpTokI);       _moeExpTokI = null;
        NativeMemory.Free(_moeExpTokK);       _moeExpTokK = null;
        NativeMemory.Free(_moeBatchGathered); _moeBatchGathered = null;
        NativeMemory.Free(_moeBatchGate);     _moeBatchGate = null;
        NativeMemory.Free(_moeBatchUp);       _moeBatchUp = null;
        NativeMemory.Free(_moeBatchDown);     _moeBatchDown = null;
        _moeBatchCap = 0;
    }

    /// <summary>
    /// Batched MoE FFN for one prefill layer: the MoE twin of the dense
    /// gate/up GEMM → SiLU → down GEMM sequence in <see cref="PrefillCore"/>.
    ///
    /// <para><paramref name="batchNorm"/> holds the <paramref name="n"/> pre-FFN-normed rows and
    /// <paramref name="batchOut"/> receives the FFN output (fully overwritten; the caller adds the
    /// residual). The two must not alias — unlike the dense path, which writes the down projection
    /// straight back over its normed input, every expert re-reads <paramref name="batchNorm"/>.</para>
    ///
    /// <para>Structure, and why it is not just "widen the GEMMs": routing is per token, so the
    /// tokens sharing an expert are an arbitrary subset. This (1) routes every token — the router
    /// itself is dense and stays per token in F32, deliberately, see below; (2) buckets the
    /// (token, slot) pairs by selected expert into CSR order; (3) gathers each expert's tokens
    /// into one contiguous batch and runs three ordinary batched GEMMs over it, so the expert's
    /// weight rows are streamed once for the whole bucket rather than once per token; (4) reduces
    /// the unweighted down partials per token in top-k slot order. Step 2 is the part with no
    /// analogue in the dense path; step 4's ordering is load-bearing, see its comment.</para>
    ///
    /// <para>The router stays on the exact per-token F32 <see cref="FusedMatVec"/> the sequential
    /// path uses. Batching it would be nearly free in cost terms, but top-k selection is discrete:
    /// int8 activation quantisation could flip a marginal expert choice, and then the batched and
    /// sequential paths would not merely round differently, they would run different experts.
    /// The router is ~0.3% of this FFN's MACs, so there is nothing to win by risking that.</para>
    ///
    /// <para>The expert GEMMs do take the int8 batched path (<c>allowQ8: true</c>) — the same one
    /// the dense batched prefill has used since the Q8-prefill ship, and admissible for the same
    /// reason: the rows are positions within one prompt. That is the only source of divergence
    /// from the sequential F32 trunk — the same class the dense batched path already carries.
    /// With <c>Q8PrefillEnabled</c> off this path is bit-identical to sequential, which
    /// MoeBatchedPrefillParityTests pins.</para>
    /// </summary>
    private void MoeFfnBatched(int layer, float* batchNorm, float* batchOut, int n)
    {
        int numExperts = _hp.NumExperts;
        int na = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;

        EnsureMoeBatchScratch(n);

        // ── 1. Route every token (per-token F32, identical to the sequential path) ──────────
        for (int t = 0; t < n; t++)
        {
            float* logits = _moeBatchRouter + (long)t * numExperts;
            FusedMatVec(logits, _wGateInp![layer], batchNorm + (long)t * _embDim, numExperts, _embDim);

            if (_hp.UseSigmoidGating)
                SimdKernels.SigmoidInPlace(logits, numExperts);
            else
                SimdKernels.SoftmaxInPlace(logits, numExperts);

            SelectTopK(logits, numExperts, na,
                new Span<int>(_moeBatchSel + (long)t * na, na),
                new Span<float>(_moeBatchWts + (long)t * na, na),
                normalize: _hp.NormalizeMoeTopKWeights);
        }

        // ── 2. Bucket the (token, slot) pairs by expert, CSR-style ─────────────────────────
        int* expStart = _moeExpStart;
        int* cursor = _moeExpCursor;
        long pairs = (long)n * na;
        for (int e = 0; e <= numExperts; e++) expStart[e] = 0;
        for (long s = 0; s < pairs; s++) expStart[_moeBatchSel[s] + 1]++;
        for (int e = 0; e < numExperts; e++) expStart[e + 1] += expStart[e];
        for (int e = 0; e < numExperts; e++) cursor[e] = expStart[e];
        for (int t = 0; t < n; t++)
            for (int k = 0; k < na; k++)
            {
                long s = (long)t * na + k;
                int p = cursor[_moeBatchSel[s]]++;
                _moeExpTokI[p] = t;
                _moeExpTokK[p] = k;
            }

        // ── 3. One batch of GEMMs per used expert ──────────────────────────────────────────
        ref readonly TensorRef gateExps = ref _wGateExps![layer];
        ref readonly TensorRef upExps = ref _wUpExps![layer];
        ref readonly TensorRef downExps = ref _wDownExps![layer];
        int bprGate = RowBytes(gateExps.DType, _embDim);
        int bprUp = RowBytes(upExps.DType, _embDim);
        int bprDown = RowBytes(downExps.DType, expertDim);

        for (int e = 0; e < numExperts; e++)
        {
            int p0 = expStart[e], p1 = expStart[e + 1];
            int cnt = p1 - p0;
            if (cnt == 0) continue;

            for (int i = 0; i < cnt; i++)
                Copy(_moeBatchGathered + (long)i * _embDim,
                     batchNorm + (long)_moeExpTokI[p0 + i] * _embDim, _embDim);

            SimdKernels.MatMulBatched(_moeBatchGate,
                gateExps.DataPtr + (long)e * expertDim * bprGate, _moeBatchGathered,
                cnt, expertDim, _embDim, gateExps.DType, allowQ8: true);
            SimdKernels.MatMulBatched(_moeBatchUp,
                upExps.DataPtr + (long)e * expertDim * bprUp, _moeBatchGathered,
                cnt, expertDim, _embDim, upExps.DType, allowQ8: true);

            // Llama-4 sigmoid gating scales the FFN input rather than its output, exactly as
            // MoeFfn does; the reduce below then uses a weight of 1 for those models.
            if (_hp.UseSigmoidGating)
                for (int i = 0; i < cnt; i++)
                {
                    float w = _moeBatchWts[(long)_moeExpTokI[p0 + i] * na + _moeExpTokK[p0 + i]];
                    SimdKernels.ScaleInPlace(_moeBatchGate + (long)i * expertDim, w, expertDim);
                    SimdKernels.ScaleInPlace(_moeBatchUp + (long)i * expertDim, w, expertDim);
                }

            // The bucket's rows are contiguous, so one SiLuMul covers the whole batch.
            SimdKernels.SiLuMul(_moeBatchGate, _moeBatchUp, cnt * expertDim);

            // Down projection into a scratch batch, then scattered UNWEIGHTED into
            // (token, slot) order — the weighting happens in phase 4.
            SimdKernels.MatMulBatched(_moeBatchGathered,
                downExps.DataPtr + (long)e * _embDim * bprDown, _moeBatchGate,
                cnt, _embDim, expertDim, downExps.DType, allowQ8: true);

            for (int i = 0; i < cnt; i++)
                Copy(_moeBatchDown + ((long)_moeExpTokI[p0 + i] * na + _moeExpTokK[p0 + i]) * _embDim,
                     _moeBatchGathered + (long)i * _embDim, _embDim);
        }

        // ── 4. Reduce per token, in TOP-K SLOT ORDER ──────────────────────────────────────
        // Not expert order, which is what the CSR loop above naturally produces. FP32 addition
        // is not associative, and reducing 8 expert contributions in a different order than
        // MoeFfn's `for k in 0..numActive` loop is not a last-bit difference: measured on OLMoE
        // it moved the final logits by up to 0.20 with every kernel otherwise identical, enough
        // to change the sampled token. Storing unweighted partials per (token, slot) and
        // reducing them here costs one extra pass over N*k*embDim floats and buys back exact
        // agreement with the sequential trunk. (This is also what the CUDA hybrid's
        // BatchedRoutedExpertsCpu does, for the same reason.)
        for (int t = 0; t < n; t++)
        {
            float* dst = batchOut + (long)t * _embDim;
            new Span<float>(dst, _embDim).Clear();
            for (int k = 0; k < na; k++)
                SimdKernels.WeightedAddInPlace(dst,
                    _moeBatchDown + ((long)t * na + k) * _embDim,
                    _hp.UseSigmoidGating ? 1f : _moeBatchWts[(long)t * na + k], _embDim);
        }

        // ── 5. Shared expert: dense over every token, so an ordinary batched FFN ───────────
        if (_hp.HasSharedExpert)
        {
            MatMulBatchedCached(_moeBatchGate, in _wGateShexp![layer], batchNorm, n, expertDim, _embDim);
            MatMulBatchedCached(_moeBatchUp, in _wUpShexp![layer], batchNorm, n, expertDim, _embDim);
            SimdKernels.SiLuMul(_moeBatchGate, _moeBatchUp, n * expertDim);
            MatMulBatchedCached(_moeBatchDown, in _wDownShexp![layer], _moeBatchGate, n, _embDim, expertDim);
            for (int t = 0; t < n; t++)
                SimdKernels.AddInPlace(batchOut + (long)t * _embDim,
                    _moeBatchDown + (long)t * _embDim, _embDim);
        }
    }

    /// <summary>Bytes one weight row of <paramref name="cols"/> elements occupies in this dtype.</summary>
    private static int RowBytes(DType dtype, int cols) =>
        (cols / DTypeInfo.BlockSize(dtype)) * DTypeInfo.BytesPerBlock(dtype);

    /// <summary>
    /// MatVec for a single expert slice from a packed expert tensor.
    /// The packed tensor has shape [numExperts * rows, cols]. Expert i's slice
    /// starts at row offset (i * rows).
    /// </summary>
    private void ExpertMatVec(float* output, in TensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(output, expertData, input, rows, cols, packedTensor.DType);
    }

    /// <summary>
    /// MatVec for expert down projection, with weighted accumulation into output.
    /// output += weight * (expertDown[expertIdx] × input)
    /// </summary>
    private void ExpertMatVecDown(float* output, in TensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input, float weight)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;

        SimdKernels.MatVec(_moeDownTemp, expertData, input, rows, cols, packedTensor.DType);

        SimdKernels.WeightedAddInPlace(output, _moeDownTemp, weight, rows);
    }

    private static void SelectTopK(float* logits, int n, int k,
        Span<int> indices, Span<float> weights, bool normalize)
    {
        // Simple selection for small k (typically 1 or 2)
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                bool alreadySelected = false;
                for (int j = 0; j < ki; j++)
                    if (indices[j] == i) { alreadySelected = true; break; }
                if (!alreadySelected && logits[i] > bestVal)
                { bestVal = logits[i]; bestIdx = i; }
            }
            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }

        // Renormalize selected weights to sum to 1 (Qwen3-MoE / Mixtral convention).
        // OLMoE skips this — its router uses raw post-softmax probabilities, so
        // unused mass on non-selected experts intentionally shrinks the MoE block's
        // contribution to the residual.
        if (normalize && k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += weights[i];
            if (sum > 0)
                for (int i = 0; i < k; i++) weights[i] /= sum;
        }
    }

    // ================================================================
    //  Embedding lookup (single-row dequant)
    // ================================================================

    private void EmbedToken(int token) => EmbedTokenInto(token, _hidden);

    // ================================================================
    //  Gemma 4 Per-Layer-Embedding (PLE)
    // ================================================================

    // Token-major layout: per_layer_token_embd shape (PleAll=10752, vocab=262144) stores
    // one row of length PleAll per token (GGUF dim[0] is row width). Gather + dequant
    // the row, then per-layer normalise + add projection + scale.
    private void BuildPerLayerProjections(int token)
    {
        int stackedDim = _hp.NumLayers * _pleWidth;

        int bytesPerRow = (stackedDim / DTypeInfo.BlockSize(_pleTokenEmbed.DType))
                        * DTypeInfo.BytesPerBlock(_pleTokenEmbed.DType);
        byte* rowPtr = _pleTokenEmbed.DataPtr + (long)token * bytesPerRow;
        if (_pleTokenEmbed.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, stackedDim)
                .CopyTo(new Span<float>(_pleRowBuf, stackedDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, _pleRowBuf, stackedDim, _pleTokenEmbed.DType);
        }

        // Gemma scales every embedding table by sqrt(its hidden dim). The PLE table's
        // hidden dim is PleWidth (256 → 16×), matching the trunk's sqrt(EmbeddingDim)
        // scale on token_embd.
        float pleScale = MathF.Sqrt(_pleWidth);
        SimdKernels.ScaleInPlace(_pleRowBuf, pleScale, stackedDim);

        SimdKernels.MatVec(_projPerLayer, (byte*)_perLayerModelProj,
            _hidden, stackedDim, _embDim, DType.Float32);

        float embScale = 1.0f / MathF.Sqrt(_embDim);
        SimdKernels.ScaleInPlace(_projPerLayer, embScale, stackedDim);

        float invSqrt2 = 1.0f / MathF.Sqrt(2.0f);
        var projNormW = GetNormWeight(_perLayerProjNormTensor);
        for (int L = 0; L < _hp.NumLayers; L++)
        {
            float* slice = _projPerLayer + (long)L * _pleWidth;
            FastRmsNorm(slice, slice, projNormW, _pleWidth, _hp.RmsNormEps);
            SimdKernels.AddInPlace(slice, _pleRowBuf + (long)L * _pleWidth, _pleWidth);
            SimdKernels.ScaleInPlace(slice, invSqrt2, _pleWidth);
        }

    }

    private void ApplyPerLayerEmbedding(int layer)
    {
        float* slice = _projPerLayer + (long)layer * _pleWidth;
        FusedMatVec(_pleX, _pleInpGate![layer], _hidden, _pleWidth, _embDim);
        SimdKernels.GeluTanhMul(_pleX, slice, _pleX, _pleWidth);
        FusedMatVec(_pleY, _plePostProj![layer], _pleX, _embDim, _pleWidth);
        var postW = GetNormWeight(_plePostNorm![layer]);
        FastRmsNorm(_pleY, _pleY, postW, _embDim, _hp.RmsNormEps);
        SimdKernels.AddInPlace(_hidden, _pleY, _embDim);
    }

    private void EmbedTokenInto(int token, float* dest)
    {
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(_embTensor.DType))
                        * DTypeInfo.BytesPerBlock(_embTensor.DType);
        byte* rowPtr = _embTensor.DataPtr + (long)token * bytesPerRow;
        if (_embTensor.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, _embDim)
                .CopyTo(new Span<float>(dest, _embDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, dest, _embDim, _embTensor.DType);
        }
    }

    // ================================================================
    //  Fused quantized MatVec (no intermediate F32 weight buffer)
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FusedMatVec(float* output, in TensorRef tensor, float* input, int rows, int cols)
    {
        SimdKernels.MatVec(output, tensor.DataPtr, input, rows, cols, tensor.DType);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FastRmsNorm(float* output, float* input, float* weight, int size, float eps)
    {
        if (_useWideNorms) SimdKernels.RmsNormWide(output, input, weight, size, eps);
        else               SimdKernels.RmsNorm    (output, input, weight, size, eps);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FastPureRmsNorm(float* output, float* input, int size, float eps)
    {
        if (_useWideNorms) SimdKernels.PureRmsNormWide(output, input, size, eps);
        else               SimdKernels.PureRmsNorm    (output, input, size, eps);
    }

    // ================================================================
    //  Norm weight cache (tiny F32 weights, cached permanently)
    // ================================================================

    private float* GetNormWeight(in TensorRef tensor)
    {
        if (_normCache.TryGetValue(tensor.Name, out var cached))
            return (float*)cached;

        var data = _model.GetTensorData(tensor.Info);
        int count = (int)tensor.Info.ElementCount;
        var buf = Alloc(count);

        if (tensor.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), tensor.DType, count);

        // GGUF converter for gemma family already bakes the HF "(1 + w)" RMSNorm
        // convention; we multiply by stored `w` directly (mirrors llama.cpp build_norm
        // in src/llama-graph.cpp). Verified vs actual GGUF: attn_norm ~8, attn_q_norm
        // ~0.98 — already final multipliers.

        _normCache[tensor.Name] = (nint)buf;
        return buf;
    }

    // ================================================================
    //  Tensor resolution
    // ================================================================

    private TensorRef ResolveTensor(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new TensorRef(name, info, info.DType, _model.GetTensorDataPtr(info));
    }

    private float LoadScalarF32(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);
        float[] buf = new float[1];
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, 1).CopyTo(buf);
        else
            Dequantize.ToFloat32(data, buf.AsSpan(), info.DType, 1);
        return buf[0];
    }

    private float* LoadBias(string name, int count)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing bias tensor: {name}");
        var data = _model.GetTensorData(info);
        var buf = Alloc(count);
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), info.DType, count);
        return buf;
    }

    private readonly unsafe struct TensorRef
    {
        public readonly string Name;
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public TensorRef(string name, GgufTensorInfo info, DType dtype, byte* dataPtr)
        {
            Name = name; Info = info; DType = dtype; DataPtr = dataPtr;
        }
    }

    // ================================================================
    //  Utilities
    // ================================================================

    /// <summary>
    /// Apply RMSNorm independently to each head-sized chunk.
    /// weight has [headDim] elements and is shared across all heads.
    /// </summary>
    private static void PerHeadRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight, headDim, eps);
    }

    private static void PerChannelRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight + h * headDim, headDim, eps);
    }

    private void ApplyQkNorm(float* q, float* k, int layer)
    {
        if (_perChannelQkNorm)
        {
            PerChannelRmsNorm(q, _qNorm[layer], _numHeads,   _headDim, _hp.RmsNormEps);
            PerChannelRmsNorm(k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
        }
        else
        {
            PerHeadRmsNorm(q, _qNorm[layer], _numHeads,   _headDim, _hp.RmsNormEps);
            PerHeadRmsNorm(k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
        }
    }

    /// <summary>
    /// Per-layer-head-dim QK-norm. <paramref name="k"/> may be null on KV-share layers
    /// where the K projection didn't run (the source layer already normed its own K).
    /// </summary>
    private void ApplyQkNormLayer(float* q, float* k, int layer, int layerHd, int kvHeads)
    {
        if (_perChannelQkNorm)
        {
            PerChannelRmsNorm(q, _qNorm[layer], _numHeads, layerHd, _hp.RmsNormEps);
            if (k != null)
                PerChannelRmsNorm(k, _kNorm[layer], kvHeads, layerHd, _hp.RmsNormEps);
        }
        else
        {
            PerHeadRmsNorm(q, _qNorm[layer], _numHeads, layerHd, _hp.RmsNormEps);
            if (k != null)
                PerHeadRmsNorm(k, _kNorm[layer], kvHeads, layerHd, _hp.RmsNormEps);
        }
    }

    private static void PerHeadPureRmsNorm(float* data, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.PureRmsNorm(data + h * headDim, data + h * headDim, headDim, eps);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)(count * sizeof(float)));

    /// <summary>
    /// Widen one token's K or V row from a compact per-layer head packing (head <c>h</c> at
    /// <c>h * headDim</c>) to the KV cache's own head stride (head <c>h</c> at
    /// <c>h * cacheHeadStride</c>), which is fixed model-wide at <c>_maxHeadDim</c>.
    /// <para>The destination must already be zeroed; the gaps between heads are never written, so
    /// re-zeroing per token would be pure waste — every call writes exactly the same head slots.</para>
    /// </summary>
    private static void ScatterToCacheStride(float* dst, float* src, int numHeads,
        int headDim, int cacheHeadStride)
    {
        if (headDim == cacheHeadStride)
        {
            Copy(dst, src, numHeads * headDim);
            return;
        }
        for (int h = 0; h < numHeads; h++)
            Copy(dst + (long)h * cacheHeadStride, src + (long)h * headDim, headDim);
    }

    private static void Copy(float* dst, float* src, int size) =>
        new ReadOnlySpan<float>(src, size).CopyTo(new Span<float>(dst, size));

    // ================================================================
    //  Continuous Batching API
    // ================================================================

    /// <summary>
    /// Creates a new empty <see cref="PagedKvCache"/> compatible with this model's layer/head dimensions.
    /// Used by <see cref="ContinuousBatchingEngine"/> to allocate per-sequence caches.
    /// </summary>
    public PagedKvCache CreateCache() =>
        new PagedKvCache(_hp.NumLayers, _hp.NumKvHeads, _maxHeadDim,
            bf16Store: PagedKvCache.Bf16StoreRequested,
            autoBf16: PagedKvCache.Bf16AutoRequested,
            layerHeadDim: _layerHeadDim);

    // ── IBatchedForwardPass (issue #190) ────────────────────────────────────────
    // The engine drives this forward pass through the backend-agnostic interface, holding
    // caches as opaque ISequenceKvCache handles. For the CPU path the handle IS the concrete
    // PagedKvCache the methods above already take, so these explicit implementations just
    // unwrap it. SnapKvEnabled / KvBytesPerToken / PrefillDequantCacheActive are public and
    // satisfy the interface implicitly.
    ISequenceKvCache IBatchedForwardPass.CreateCache() => CreateCache();

    int IPrefixCacheableBatchedForwardPass.PrefixCacheBlockSize => PagedKvCache.PageSize;

    ISequenceKvCache IPrefixCacheableBatchedForwardPass.CapturePrefix(ISequenceKvCache cache, int prefixLength) =>
        ((PagedKvCache)cache).ForkSharedPrefix(prefixLength);

    ISequenceKvCache IPrefixCacheableBatchedForwardPass.ForkPrefix(ISequenceKvCache prefix) =>
        ((PagedKvCache)prefix).ForkSharedPrefix(((PagedKvCache)prefix).Length);

    ReadOnlySpan<float> IBatchedForwardPass.PrefillWithCache(
        IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos)
        => PrefillWithCache(tokens, (PagedKvCache)cache, startPos);

    float[]?[] IBatchedForwardPass.PrefillPackedMulti(
        ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
        => PrefillPackedMulti(chunks, startPos, AsPaged(caches), wantLogits);

    float[][] IBatchedForwardPass.BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        => BatchForwardMulti(tokens, positions, AsPaged(caches));

    private static PagedKvCache[] AsPaged(ISequenceKvCache[] caches)
    {
        var r = new PagedKvCache[caches.Length];
        for (int i = 0; i < caches.Length; i++)
            r[i] = (PagedKvCache)caches[i];
        return r;
    }

    /// <summary>
    /// Forward pass for a single token using the provided explicit cache (no TurboQuant).
    /// Used by <see cref="PrefillWithCache"/> for single-token prompts and MoE sequential prefill.
    /// </summary>
    private ReadOnlySpan<float> ForwardCore(int token, int pos, PagedKvCache cache)
    {
        // Scratch sized from _ctxLen, but the KV cache is not: PagedKvCache defaults to 8192
        // blocks (131,072 positions), so it keeps accepting appends long after `pos` has run off
        // the end of the ctxLen-sized buffers. Attention writes scores[h * _ctxLen + t] for
        // t < pos + 1, and RoPE reads _ropeCosTable + pos * _ropeHalfDim; both are unchecked
        // native accesses, so overrunning corrupts memory rather than failing. Callers are
        // expected to stop at MaxSeqLen — this makes the invariant unbypassable instead of
        // trusting each one, and turns silent corruption into a diagnosable throw.
        if ((uint)pos >= (uint)_ctxLen)
        {
            throw new ArgumentOutOfRangeException(nameof(pos), pos,
                $"Position exceeds the active context length ({_ctxLen}). Generation must stop at " +
                $"MaxSeqLen; continuing would write past the attention-score and RoPE scratch buffers.");
        }

        EmbedToken(token);
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            Copy(_residual, _hidden, _embDim);
            var normW = GetNormWeight(_attnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, normW, _embDim, _hp.RmsNormEps);
            FusedMatVec(_q, _wq[layer], _normBuf, _numHeads * _headDim, _embDim);
            FusedMatVec(_k, _wk[layer], _normBuf, _numKvHeads * _headDim, _embDim);
            FusedMatVec(_v, _wv[layer], _normBuf, _numKvHeads * _headDim, _embDim);
            if (_hasAttnBias)
            {
                SimdKernels.AddInPlace(_q, _bq[layer], _numHeads * _headDim);
                SimdKernels.AddInPlace(_k, _bk[layer], _numKvHeads * _headDim);
                SimdKernels.AddInPlace(_v, _bv[layer], _numKvHeads * _headDim);
            }
            {
                bool useRoPE = _hp.NoRopeLayerStep == 0
                    || (layer + 1) % _hp.NoRopeLayerStep != 0;
                if (_hasQkNorm && !_hp.UseL2QkNorm)
                {
                    ApplyQkNorm(_q, _k, layer);
                }
                if (useRoPE)
                {
                    ApplyRope(_q, pos, _numHeads);
                    ApplyRope(_k, pos, _numKvHeads);
                }
                if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                {
                    PerHeadPureRmsNorm(_q, _numHeads, _headDim, _hp.RmsNormEps);
                    PerHeadPureRmsNorm(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
                }
            }
            cache.Append(layer,
                new ReadOnlySpan<float>(_k, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_v, _numKvHeads * _headDim));
            Attention(cache, layer, pos);
            FusedMatVec(_hidden, _wo[layer], _attnOut, _embDim, _numHeads * _headDim);
            if (_hasAttnOutputBias)
                SimdKernels.AddInPlace(_hidden, _bo[layer], _embDim);
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);
            Copy(_residual, _hidden, _embDim);
            var ffnNormW = GetNormWeight(_ffnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, ffnNormW, _embDim, _hp.RmsNormEps);
            if (_hp.IsMoE)
                MoeFfn(layer);
            else
                DenseFfn(layer);
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);
        }
        cache.IncrementPosition();
        var outNormW = GetNormWeight(_outputNorm);
        SimdKernels.RmsNorm(_hidden, _hidden, outNormW, _embDim, _hp.RmsNormEps);
        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);
        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    /// <summary>
    /// Prefill prompt tokens into an explicitly provided KV cache instead of the engine's primary cache.
    /// Used by <see cref="ContinuousBatchingEngine"/> to prefill per-sequence caches concurrently.
    /// Not supported when TurboQuant KV cache is enabled.
    /// </summary>
    /// <param name="tokens">Prompt token IDs to process.</param>
    /// <param name="cache">The KV cache to write into.</param>
    /// <param name="startPos">Starting position in the cache (default 0).</param>
    /// <returns>Logits for the last token.</returns>
    public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, PagedKvCache cache, int startPos = 0)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("PrefillWithCache is not supported when TurboQuant KV cache is enabled.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on PrefillWithCache.");
        int N = tokens.Count;
        if (N == 0) throw new ArgumentException("Token list is empty", nameof(tokens));
        // Keep the externally supplied-cache route coherent with PrefillDispatch. Continuous
        // batching calls this method, so leaving the all-control safeguard only on Prefill would
        // make server-admitted structural probes take the numerically unsafe Q8 activation path.
        if (N == 1 || IsAllControlTokenPrompt(tokens) || IsSingleDistinctTokenPrompt(tokens)
            || (_hp.IsMoE && !MoeBatchedPrefillSupported))
            return PrefillWithCacheSequential(tokens, cache, startPos);
        return PrefillCore(tokens, cache, startPos);
    }

    /// <summary>
    /// Exact, token-at-a-time external-cache admission. Used for individually ineligible
    /// requests and for the whole packed group when one member is an all-control structural
    /// probe: packing must not make an ordinary neighbour's numerical route timing-dependent.
    /// </summary>
    private ReadOnlySpan<float> PrefillWithCacheSequential(
        IReadOnlyList<int> tokens, PagedKvCache cache, int startPos)
    {
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = ForwardCore(tokens[i], startPos + i, cache);
        return logits;
    }

    /// <summary>
    /// Batched decode step for N sequences simultaneously: one token per sequence, each with its own
    /// KV cache at the given position.
    ///
    /// <para><b>On what batching actually buys here.</b> This used to claim it "amortizes weight
    /// reads N× across concurrent users". Measured on CPU (docs/session-native-inference-runtime-plan.md
    /// §3.4.11): going from 1 to 4 sequences cuts per-token trunk time by 12.7%, and four
    /// concurrent sessions aggregate 1.18x a single session — not 4x. The weight reuse is real
    /// (<see cref="SimdKernels.BatchedMatVecTierEnabled"/> routes the batch through
    /// <c>MatVec4In</c>, which reads and dequantizes each weight row once for four inputs) but the
    /// per-input FMA work is still N×, and that is where the time goes: FFN alone is ~70% of decode
    /// trunk time and improves only 11% at N=4. CPU decode at these sizes is not weight-bandwidth
    /// bound. Attention is ~3% of decode, so per-sequence attention is not the limiter either.</para>
    /// Not supported when TurboQuant KV cache is enabled or for MoE models.
    /// </summary>
    /// <param name="tokens">Next token for each sequence (length N).</param>
    /// <param name="positions">Current decode position for each sequence (= cache.Length before this call).</param>
    /// <param name="caches">Per-sequence KV cache (length N).</param>
    /// <returns>Logits array for each sequence (length N × VocabSize).</returns>
    public float[][] BatchForwardMulti(int[] tokens, int[] positions, PagedKvCache[] caches)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("BatchForwardMulti is not supported when TurboQuant KV cache is enabled.");
        if (_hp.IsMoE)
            throw new NotSupportedException("BatchForwardMulti is not supported for MoE models; use individual ForwardCore calls.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on BatchForwardMulti.");
        int N = tokens.Length;
        if (N == 0) return Array.Empty<float[]>();
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        bool profDecodeB = DecodeProfileTimers.Enabled;
        // One decoded token per SEQUENCE, so the profiler's per-token averages mean the same thing
        // here as on the sequential path rather than counting a whole batch as a single token.
        if (profDecodeB) for (int pt = 0; pt < N; pt++) DecodeProfileTimers.CountToken();
        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim);
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            try
            {
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Stage timing (STINGRAY_PROFILE_DECODE=1). BatchForwardMulti had none,
                    // so the multi-sequence decode path -- the one whose flat scaling the session
                    // runtime plan is trying to explain -- could not be profiled at all, only
                    // speculated about. Same categories as ForwardCore so the two decode paths
                    // compare bucket for bucket.
                    long bStage = profDecodeB ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    var normW = GetNormWeight(_attnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }
                    if (profDecodeB)
                    {
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.RmsNorm,
                            System.Diagnostics.Stopwatch.GetTimestamp() - bStage);
                        bStage = System.Diagnostics.Stopwatch.GetTimestamp();
                    }
                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);
                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }
                    if (profDecodeB)
                    {
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.QkvProj,
                            System.Diagnostics.Stopwatch.GetTimestamp() - bStage);
                    }
                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;
                    // Per-sequence: RoPE, KV append to individual cache, causal attention.
                    // RoPE/norm/append and the attention itself are timed separately: the open
                    // question is specifically whether per-sequence ATTENTION is the irreducible
                    // residual, and lumping the cache bookkeeping in with it would not answer that.
                    long bRope = 0, bAttn = 0;
                    for (int n = 0; n < N; n++)
                    {
                        long bSeq = profDecodeB ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;
                        int pos = positions[n];
                        // Soft-reset this layer's position so the Append lands at pos
                        caches[n].TruncateTo(pos);
                        if (useRoPE)
                        {
                            ApplyRope(qn, pos, _numHeads);
                            ApplyRope(kn, pos, _numKvHeads);
                        }
                        if (_hasQkNorm)
                        {
                            if (_hp.UseL2QkNorm)
                            {
                                PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                            }
                            else
                            {
                                ApplyQkNorm(qn, kn, layer);
                            }
                        }
                        caches[n].Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        caches[n].IncrementPosition(); // _length = pos+1
                        if (profDecodeB)
                        {
                            long now = System.Diagnostics.Stopwatch.GetTimestamp();
                            bRope += now - bSeq;
                            bSeq = now;
                        }
                        Copy(_q, qn, qDim);
                        Attention(caches[n], layer, pos);
                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                        if (profDecodeB) bAttn += System.Diagnostics.Stopwatch.GetTimestamp() - bSeq;
                    }
                    if (profDecodeB)
                    {
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.RoPE, bRope);
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.Attention, bAttn);
                        bStage = System.Diagnostics.Stopwatch.GetTimestamp();
                    }
                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);
                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                    if (profDecodeB)
                    {
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.OutProj,
                            System.Diagnostics.Stopwatch.GetTimestamp() - bStage);
                        bStage = System.Diagnostics.Stopwatch.GetTimestamp();
                    }
                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }
                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);
                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);
                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                    if (profDecodeB)
                        DecodeProfileTimers.Add(DecodeProfileTimers.Category.Ffn,
                            System.Diagnostics.Stopwatch.GetTimestamp() - bStage);
                }
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }
            var outNormW = GetNormWeight(_outputNorm);
            var result = new float[N][];
            for (int n = 0; n < N; n++)
            {
                float* h = batchHidden + (long)n * _embDim;
                SimdKernels.RmsNorm(h, h, outNormW, _embDim, _hp.RmsNormEps);
                FusedMatVec(_logits, _outputWeight, h, _hp.VocabSize, _embDim);
                result[n] = new float[_hp.VocabSize];
                new ReadOnlySpan<float>(_logits, _hp.VocabSize).CopyTo(result[n]);
            }
            return result;
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// Packed multi-sequence prefill (issue #183 Gap 2): processes one chunk of prompt
    /// tokens from each of S sequences in a single forward pass. All chunks are
    /// concatenated into one packed batch so every GEMM amortizes weight reads across
    /// the combined token count — the multi-prompt analogue of what
    /// <see cref="BatchForwardMulti"/> does for decode. Attention stays per-token
    /// against each sequence's own cache (varlen / cu_seqlens-style: no cross-sequence
    /// attention, no padding).
    ///
    /// SnapKV eviction is never applied here — chunked admission feeds this with
    /// startPos &gt; 0 segments where SnapKV scoring doesn't run; callers that want
    /// SnapKV must use whole-prompt <see cref="PrefillWithCache"/> instead (the engine
    /// gates on <see cref="SnapKvEnabled"/>).
    /// </summary>
    /// <param name="chunks">Per-sequence token chunk (each non-empty).</param>
    /// <param name="startPos">Per-sequence cache position at which its chunk begins.</param>
    /// <param name="caches">Per-sequence KV cache.</param>
    /// <param name="wantLogits">
    /// Per-sequence: compute logits for the chunk's last token (true for a sequence's
    /// final chunk; intermediate chunks skip the vocab projection).
    /// </param>
    /// <returns>Per-sequence logits array, null where <paramref name="wantLogits"/> was false.</returns>
    public float[]?[] PrefillPackedMulti(
        ReadOnlyMemory<int>[] chunks, int[] startPos, PagedKvCache[] caches, bool[] wantLogits)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("PrefillPackedMulti is not supported when TurboQuant KV cache is enabled.");
        if (_hp.IsMoE)
            throw new NotSupportedException("PrefillPackedMulti is not supported for MoE models.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on PrefillPackedMulti.");

        int S = chunks.Length;
        if (S == 0) return Array.Empty<float[]?>();
        if (startPos.Length != S || caches.Length != S || wantLogits.Length != S)
            throw new ArgumentException("chunks/startPos/caches/wantLogits lengths must match.");

        // Packed admission is normally the highest-throughput prefill route. It must not,
        // however, reintroduce Q8 activation quantisation for an all-control structural probe
        // which Prefill and PrefillWithCache deliberately keep on the F32 route. Such prompts
        // are rare and short; falling back for the whole affected packed batch avoids silently
        // changing numerical behaviour based solely on whether another request arrived nearby.
        for (int s = 0; s < S; s++)
        {
            if (!IsAllControlTokenPrompt(chunks[s].Span) && !IsSingleDistinctTokenPrompt(chunks[s].Span)) continue;
            var fallback = new float[]?[S];
            for (int i = 0; i < S; i++)
            {
                ReadOnlySpan<float> logits = PrefillWithCacheSequential(
                    chunks[i].ToArray(), caches[i], startPos[i]);
                if (wantLogits[i]) fallback[i] = logits.ToArray();
            }
            return fallback;
        }

        // Packed offsets: sequence s owns packed rows [off[s], off[s+1]).
        var off = new int[S + 1];
        for (int s = 0; s < S; s++)
        {
            if (chunks[s].IsEmpty)
                throw new ArgumentException($"Chunk for sequence {s} is empty.", nameof(chunks));
            off[s + 1] = off[s] + chunks[s].Length;
        }
        int N = off[S];

        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            for (int s = 0; s < S; s++)
            {
                var span = chunks[s].Span;
                for (int i = 0; i < span.Length; i++)
                    EmbedTokenInto(span[i], batchHidden + (long)(off[s] + i) * _embDim);
            }

            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            try
            {
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Rewind each cache so this layer's appends land at startPos[s]
                    // (Append advances the shared position counter every layer; same
                    // per-layer soft reset PrefillCore does for its single cache).
                    for (int s = 0; s < S; s++)
                        caches[s].TruncateTo(startPos[s]);

                    var normW = GetNormWeight(_attnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    MatMulBatchedCached(batchQ, in _wq[layer], batchNorm, N, qDim, _embDim);
                    MatMulBatchedCached(batchK, in _wk[layer], batchNorm, N, kvDim, _embDim);
                    MatMulBatchedCached(batchV, in _wv[layer], batchNorm, N, kvDim, _embDim);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }

                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    // Per-token: RoPE at the token's own absolute position, KV append
                    // into the token's own cache, causal attention over that cache only.
                    for (int s = 0; s < S; s++)
                    {
                        for (int i = 0; i < chunks[s].Length; i++)
                        {
                            int n = off[s] + i;
                            int pos = startPos[s] + i;
                            float* qn = batchQ + (long)n * qDim;
                            float* kn = batchK + (long)n * kvDim;
                            float* vn = batchV + (long)n * kvDim;

                            if (_hasQkNorm && !_hp.UseL2QkNorm)
                            {
                                ApplyQkNorm(qn, kn, layer);
                            }
                            if (useRoPE)
                            {
                                ApplyRope(qn, pos, _numHeads);
                                ApplyRope(kn, pos, _numKvHeads);
                            }
                            if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                            {
                                PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                            }

                            caches[s].Append(layer,
                                new ReadOnlySpan<float>(kn, kvDim),
                                new ReadOnlySpan<float>(vn, kvDim));
                            caches[s].IncrementPosition();

                            Copy(_q, qn, qDim);
                            Attention(caches[s], layer, pos);
                            Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                        }
                    }

                    MatMulBatchedCached(batchNorm, in _wo[layer], batchAttnOut, N, _embDim, qDim);
                    if (_hasAttnOutputBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }

                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }
                    MatMulBatchedCached(batchFfnGate, in _wGate[layer], batchNorm, N, _intermDim, _embDim);
                    MatMulBatchedCached(batchFfnUp, in _wUp[layer], batchNorm, N, _intermDim, _embDim);
                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);
                    MatMulBatchedCached(batchNorm, in _wDown[layer], batchFfnGate, N, _embDim, _intermDim);
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }

                // Leave every cache at its post-chunk length for subsequent decode/chunks.
                for (int s = 0; s < S; s++)
                    caches[s].TruncateTo(startPos[s] + chunks[s].Length);
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }

            // Final norm + vocab projection only for sequences whose chunk completes
            // their prompt — intermediate chunks never need logits.
            var outNormW = GetNormWeight(_outputNorm);
            var result = new float[]?[S];
            for (int s = 0; s < S; s++)
            {
                if (!wantLogits[s]) continue;
                float* lastHidden = batchHidden + (long)(off[s + 1] - 1) * _embDim;
                SimdKernels.RmsNorm(lastHidden, lastHidden, outNormW, _embDim, _hp.RmsNormEps);
                FusedMatVec(_logits, _outputWeight, lastHidden, _hp.VocabSize, _embDim);
                var arr = new float[_hp.VocabSize];
                new ReadOnlySpan<float>(_logits, _hp.VocabSize).CopyTo(arr);
                result[s] = arr;
            }
            return result;
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }


    public void Dispose()
    {
        NativeMemory.Free(_hidden);
        NativeMemory.Free(_residual);
        NativeMemory.Free(_normBuf);
        NativeMemory.Free(_q);
        NativeMemory.Free(_k);
        NativeMemory.Free(_v);
        NativeMemory.Free(_attnOut);
        NativeMemory.Free(_ffnGate);
        NativeMemory.Free(_ffnUp);
        NativeMemory.Free(_logits);
        NativeMemory.Free(_attnScores);
        NativeMemory.Free(_ropeCosTable);
        NativeMemory.Free(_ropeSinTable);
        if (_ropeCosTableSwa != null) NativeMemory.Free(_ropeCosTableSwa);
        if (_ropeSinTableSwa != null) NativeMemory.Free(_ropeSinTableSwa);
        _taps?.Dispose();

        foreach (var ptr in _normCache.Values)
            NativeMemory.Free((void*)ptr);
        _normCache.Clear();

        foreach (var ptr in _q4kx8Cache.Values)
            NativeMemory.Free((void*)ptr);
        _q4kx8Cache.Clear();
        _q4kx8CacheUsedBytes = 0;

        foreach (var ptr in _dequantWeightCache.Values)
            NativeMemory.Free((void*)ptr);
        _dequantWeightCache.Clear();

        if (_hasAttnBias)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                if (_bq[i] != null) NativeMemory.Free(_bq[i]);
                if (_bk[i] != null) NativeMemory.Free(_bk[i]);
                if (_bv[i] != null) NativeMemory.Free(_bv[i]);
                if (_bo[i] != null) NativeMemory.Free(_bo[i]);
            }
        }

        if (_hasQkNorm && !_hp.UseL2QkNorm)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                NativeMemory.Free(_qNorm[i]);
                NativeMemory.Free(_kNorm[i]);
            }
        }

        if (_hp.IsMoE)
        {
            NativeMemory.Free(_routerLogits);
            NativeMemory.Free(_sharedOut);
            NativeMemory.Free(_expertGate);
            NativeMemory.Free(_expertUp);
            NativeMemory.Free(_moeDownTemp);
            FreeMoeBatchScratch();
        }

        if (_hp.HasPerLayerTokenEmbd)
        {
            NativeMemory.Free(_perLayerModelProj);
            NativeMemory.Free(_pleRowBuf);
            NativeMemory.Free(_projPerLayer);
            NativeMemory.Free(_pleX);
            NativeMemory.Free(_pleY);
        }

        _kvCache.Dispose();
    }
}
