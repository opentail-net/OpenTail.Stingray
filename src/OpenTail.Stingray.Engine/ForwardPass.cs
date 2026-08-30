
namespace OpenTail.Stingray.Engine;

/// <summary>
/// Optimized CPU forward pass for a dense LLaMA-family transformer.
/// Uses AVX2 SIMD, fused dequant-matvec, and multi-threading.
/// </summary>
public sealed unsafe partial class ForwardPass : IForwardPass, IBatchedForwardPass, IPrefixCacheableBatchedForwardPass
{
    // Widened from GgufModel to the tensor-source seam: ForwardPass uses only FindTensor,
    // GetTensorData and GetTensorDataPtr, so a non-GGUF source can feed this unmodified loop.
    private readonly IModelTensorSource _model;
    private readonly ModelHyperparams _hp;
    private readonly PagedKvCache _kvCache;
    private readonly int _ctxLen; // scratch buffer sizing (attnScores, TurboQuant)
    private bool _disposed;

    /// <summary>
    /// When true, <see cref="RunTrunk"/> skips the final vocabulary projection (<c>FusedMatVec</c>).
    /// Used by specialized pipelines (e.g. Fish Speech) that slice only a small sub-vocabulary.
    /// </summary>
    public bool SkipOutputProjection { get; set; }

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
    // Parallel-residual (gptneox) scratch: holds attn_out while DenseFfn/MoeFfn overwrite
    // _hidden with ffn_out, since both must be summed with the SAME inpL afterward. Null
    // (unused) for every architecture with the ordinary sequential residual.
    private readonly float* _parAttnOut;
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

    // Apertus xIELU (non-gated FFN): non-null only when hp.XieluAlphaN is non-null. The
    // "is this layer non-gated" signal at the hot path is _wGate[layer].DataPtr is null, not a
    // separate flag — see DenseFfn.
    private readonly float[]? _xieluAlphaN, _xieluAlphaP, _xieluBeta, _xieluEps;
    // JAIS-2: ReLU-squared non-gated FFN activation. See ModelHyperparams.UsesReluSquared.
    private readonly bool _usesReluSquared;

    // Precomputed tensor metadata for hot-path access
    private readonly TensorRef _embTensor;
    // GPT-2's learned absolute position embedding table (`position_embd.weight`), added to the
    // token embedding once per token before the trunk starts. Null for every RoPE architecture.
    private readonly TensorRef? _posEmbdTensor;
    private readonly float* _posEmbdScratch;
    private readonly TensorRef[] _attnNorm;
    private readonly TensorRef[] _wq, _wk, _wv, _wo;
    private readonly TensorRef[] _ffnNorm;
    private readonly TensorRef[] _wGate, _wUp, _wDown;
    private readonly TensorRef _outputNorm;
    private TensorRef _outputWeight;

    /// <summary>
    /// Swaps the raw data pointer for the output projection head without reallocating or modifying the tensor metadata.
    /// Used by multi-head models such as QwenTTS acoustic code predictor.
    /// </summary>
    public void SetOutputWeightDataPtr(byte* dataPtr)
    {
        _outputWeight = new TensorRef(_outputWeight.Name, _outputWeight.Info, _outputWeight.DType, dataPtr);
    }

    // Optional attention biases (Qwen models)
    private readonly bool _hasAttnBias;
    private readonly bool _hasAttnOutputBias;
    private readonly float*[] _bq, _bk, _bv, _bo;

    // GPT-NeoX/Pythia: LayerNorm bias (attn/ffn/output norm) and FFN up/down bias. Null
    // arrays when the architecture doesn't carry them (i.e. always, except gptneox today).
    private readonly bool _hasNormBias;
    // Command-R (cohere2): true LayerNorm math with NO bias tensor at all — _hasNormBias stays
    // false (no bias arrays to allocate/load) but FastNorm still needs to take the LayerNorm
    // path, not RMSNorm. See ModelHyperparams.UsesLayerNorm.
    private readonly bool _usesLayerNorm;
    // OLMo v1: no learned scale OR bias on any norm at all. See ModelHyperparams.UsesUnweightedNorm.
    private readonly bool _usesUnweightedNorm;
    private readonly float*[]? _bAttnNorm;
    private readonly float*[]? _bFfnNorm;
    private readonly float* _bOutputNorm;
    private readonly bool _hasFfnBias;
    private readonly float*[]? _bFfnUp;
    private readonly float*[]? _bFfnDown;

    // Partial-RoPE width: leading _ropeDim channels of each head are rotated, the rest pass
    // through. Equals _headDim for every architecture except gptneox (16 of 64 for Pythia).
    private readonly int _ropeDim;

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

    // MLA (Multi-head Latent Attention, DeepSeek-V2/V3/R1). Q stays a plain per-head projection
    // (wq) for "lite" checkpoints (q_lora_rank==0, the only variant implemented so far -- see
    // ResolveTensor's deepseek2 branch). K/V are produced from a compressed 512+64-dim latent
    // instead of separate wk/wv projections: wKvAMqa compresses the residual down, kvANorm RMS-
    // normalizes the 512-dim latent part, wKvB decompresses it back up to full per-head K-nope
    // and V (this engine only implements the legacy unsplit wkv_b tensor layout -- see
    // MlaComputeKv's doc comment for why the split wk_b/wv_b "absorption" layout isn't handled).
    // Internal per-head layout is [rope(ropeDim), nope(headDim-ropeDim)] -- ROPE FIRST, unlike
    // ggml's [nope, rope] -- specifically so the existing partial-RoPE mechanism (_ropeDim <
    // layerHd rotates only the LEADING _ropeDim channels) can rotate the MLA rope component with
    // zero new RoPE code; attention's dot product is order-invariant as long as Q and K use the
    // same permutation, which they do here.
    private readonly bool _isMla;
    private readonly TensorRef[]? _wKvAMqa;  // [embDim, kvLoraRank + ropeDim] per layer
    private readonly TensorRef[]? _kvANorm;  // [kvLoraRank] per layer (RMSNorm weight)
    private readonly TensorRef[]? _wKvB;     // legacy unsplit [kvLoraRank, numHeads*(nopeDim+vDim)] per layer
    private readonly int _mlaKvLoraRank;
    private readonly int _mlaNopeDim;        // headDim - ropeDim (K's non-RoPE component width)
    private readonly int _mlaVDim;           // per-head V width (attention.value_length)
    private readonly float* _mlaKvCmprPe;    // scratch [kvLoraRank + ropeDim]: compressed latent + shared rope key
    private readonly float* _mlaDecompressed; // scratch [numHeads * (nopeDim + vDim)]: decompressed k_nope+v, per head
    private readonly float* _mlaAttnOutCompact; // scratch [numHeads * vDim]: attention output with the zero-pad tail dropped, ready for _wo

    /// <summary>
    /// Whether layer <paramref name="layer"/> uses the MoE FFN. DeepSeek-V2/V3's leading
    /// LeadingDenseBlockCount layers (blk.0 only, for DeepSeek-V2-Lite) are plain dense FFN even
    /// though hp.IsMoE is true for the model overall -- every OTHER MoE architecture this engine
    /// supports has LeadingDenseBlockCount==0, so this collapses to the model-level hp.IsMoE flag
    /// for them (every layer MoE-or-not is decided once, not per layer).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool IsMoeLayer(int layer) => _hp.IsMoE && layer >= _hp.LeadingDenseBlockCount;

    /// <summary>
    /// Active fine-tuned LoRA adapter dynamically applied during this forward pass.
    /// When null, the forward pass runs base model weights with zero overhead.
    /// </summary>
    public OpenTail.Stingray.Core.Lora.LoraAdapter? ActiveLora { get; set; }

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

    // Diagnostic: MLA (deepseek2) layer-0 intermediate sums, to diff against llama.cpp's
    // llama-eval-callback ground truth (env: STINGRAY_MLA_TRACE=1). Temporary, for the
    // 2026-08-21 ground-truth-diffing session -- see docs/bugstofix.md.
    private static readonly bool s_mlaTrace =
        Environment.GetEnvironmentVariable("STINGRAY_MLA_TRACE") == "1";
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
        if (hp.XieluAlphaN is { } xn)
        {
            _xieluAlphaN = new float[hp.NumLayers]; _xieluAlphaP = new float[hp.NumLayers];
            _xieluBeta = new float[hp.NumLayers]; _xieluEps = new float[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _xieluAlphaN[i] = xn[i];
                _xieluAlphaP[i] = hp.XieluAlphaP![i];
                _xieluBeta[i] = hp.XieluBeta![i];
                _xieluEps[i] = hp.XieluEps![i];
            }
        }
        _usesReluSquared = hp.UsesReluSquared;

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
        if (hp.UseParallelResidual) _parAttnOut = Alloc(_embDim);
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
            // MLA (DeepSeek-V2/V3): YaRN-scaled table instead of the plain one. freqFactors
            // (Gemma 4's per-pair rope_freqs.weight) never coexists with MLA, so no conflict.
            if (hp.KvLoraRank > 0 && hp.RopeYarnFactor > 1f)
            {
                // llama-context.cpp deliberately pre-divides cparams.yarn_attn_factor by
                // (1 + 0.1*log(factor)) ("cancel this factor" -- discussions/7416, PR #17945)
                // specifically so it cancels back out inside deepseek2.cpp's attn_factor_org
                // recovery (attn_factor * (1 + 0.1*log(factor))). Passing 1f here (skipping the
                // pre-division) left an extra (1+0.1*log(factor)) factor baked into the RoPE
                // table's magnitude scaling that shouldn't be there. With RopeYarnLogMul != 0
                // (true for every deepseek2 GGUF we've seen), deepseek2.cpp's own DEEPSEEK2
                // special case (llama-context.cpp:211) forces that formula's numerator/
                // denominator to match, so the pre-division reduces to this closed form.
                float attnFactorForTable = hp.RopeYarnLogMul != 0f
                    ? 1f / (1f + 0.1f * MathF.Log(hp.RopeYarnFactor))
                    : 1f;
                SimdKernels.BuildYarnRopeTable(_ropeCosTable, _ropeSinTable, ctxLen, maxRopeDim, hp.RopeTheta,
                    hp.RopeYarnOrigCtxLen, freqScale: 1f / hp.RopeYarnFactor, extFactor: 1f, attnFactor: attnFactorForTable);
            }
            else
            {
                SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable, ctxLen, maxRopeDim, hp.RopeTheta, globalFreqFactors);
            }
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

        if (_model.FindTensor("position_embd.weight") is { } posEmbdInfo)
        {
            _posEmbdTensor = new TensorRef("position_embd.weight", posEmbdInfo, posEmbdInfo.DType,
                _model.GetTensorDataPtr(posEmbdInfo));
            _posEmbdScratch = Alloc(_embDim);
        }

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

        _hasNormBias = hp.HasNormBias;
        _usesLayerNorm = hp.UsesLayerNorm;
        _usesUnweightedNorm = hp.UsesUnweightedNorm;
        if (_hasNormBias)
        {
            _bAttnNorm = new float*[L];
            _bFfnNorm = new float*[L];
            _bOutputNorm = LoadBias("output_norm.bias", _embDim);
        }

        _hasFfnBias = hp.HasFfnBias;
        if (_hasFfnBias)
        {
            _bFfnUp = new float*[L];
            _bFfnDown = new float*[L];
        }

        // GPT-NeoX/Pythia partial RoPE (rope.dimension_count < headDim): rotate only the
        // leading _ropeDim channels of each head, dims [_ropeDim, _headDim) pass through
        // unchanged. hp.RopeDim already sizes the cos/sin table (see maxRopeDim above); most
        // architectures leave it at headDim, so this is a no-op for them.
        _ropeDim = hp.RopeDim > 0 ? hp.RopeDim : _headDim;

        _hasQkNorm = hp.HasQkNorm;
        _perChannelQkNorm = hp.IsPerChannelQkNorm;
        _qNorm = new float*[L]; _kNorm = new float*[L];

        _isMla = hp.KvLoraRank > 0;
        if (_isMla)
        {
            _mlaKvLoraRank = hp.KvLoraRank;
            _mlaNopeDim = _headDim - _ropeDim;
            _mlaVDim = hp.MlaVHeadDim > 0 ? hp.MlaVHeadDim : _mlaNopeDim;
            if (_mlaNopeDim < 1)
                throw new NotSupportedException(
                    $"MLA head_dim ({_headDim}) must exceed rope_dim ({_ropeDim}) to leave room for the non-RoPE component.");
            _wKvAMqa = new TensorRef[L];
            _kvANorm = new TensorRef[L];
            _wKvB = new TensorRef[L];
            _mlaKvCmprPe = Alloc(_mlaKvLoraRank + _ropeDim);
            _mlaDecompressed = Alloc(_numHeads * (_mlaNopeDim + _mlaVDim));
            _mlaAttnOutCompact = Alloc(_numHeads * _mlaVDim);
        }

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

            // OLMo2 has NO attn_norm tensor at all — attention reads the raw residual directly,
            // with only a POST-attention norm (_postAttnNorm below) applied to the sublayer's
            // output before the residual add. Left at its default (DataPtr null) when absent;
            // RunTrunk/PrefillCore check that sentinel the same way _wGate[i].DataPtr is null
            // already gates Apertus/GPT-NeoX's non-gated FFN.
            _attnNorm[i] = _model.FindTensor($"blk.{i}.attn_norm.weight") is not null
                ? ResolveTensor($"blk.{i}.attn_norm.weight")
                : default;
            _wo[i] = ResolveTensor($"blk.{i}.attn_output.weight");
            // Falcon-7B has no ffn_norm tensor at all — attention and FFN read the SAME
            // LayerNorm output (src/models/falcon.cpp: "use the attn norm, not the result").
            // Reusing _attnNorm[i]'s TensorRef recomputes an identical LayerNorm a second time
            // (cheap, and deterministic — bit-identical to reusing a cached activation) rather
            // than requiring a second code path; falls through to the ordinary per-layer
            // ffn_norm tensor for every other architecture. For OLMo2, _attnNorm[i] is ALSO
            // absent (DataPtr null, see above), so this correctly degrades to "no FFN pre-norm
            // either" rather than accidentally reusing a tensor that doesn't exist.
            _ffnNorm[i] = _model.FindTensor($"blk.{i}.ffn_norm.weight") is not null
                ? ResolveTensor($"blk.{i}.ffn_norm.weight")
                : _attnNorm[i];

            // GPT-NeoX/Pythia ships one fused blk.N.attn_qkv.weight (shape [embDim, embDim +
            // 2*kvEmbDim]) rather than separate attn_q/attn_k/attn_v tensors. llama.cpp's
            // converter (conversion/gptneox.py) concatenates Q rows, then K rows, then V rows
            // — a plain contiguous split, NOT interleaved per head — so this only needs a
            // byte-offset TensorRef into the same backing tensor per projection, with no
            // actual data copy or repacking.
            if (_model.FindTensor($"blk.{i}.attn_qkv.weight") is { } qkvInfo)
            {
                byte* qkvBase = _model.GetTensorDataPtr(qkvInfo);
                int bytesPerRow = (_embDim / DTypeInfo.BlockSize(qkvInfo.DType))
                                * DTypeInfo.BytesPerBlock(qkvInfo.DType);
                int qDim = _numHeads * layerHd;
                int kvDim = _numKvHeads * layerHd;

                // Each slice needs its OWN Info with its own row count, not the fused tensor's
                // Info verbatim — PrefaultWeights sizes its read range from Info.ByteSize,
                // oblivious to any row-offset baked into DataPtr. Reusing the full-width Info
                // for _wk/_wv would compute a prefault range starting partway through the
                // tensor and reading a further FULL fused-width past that point — harmless only
                // because it happens to land on other valid mmap'd tensor data for every layer
                // except potentially the last one in the file (see the identical, but actually
                // triggered, defect this exact pattern caused for GLM4's fused ffn_up).
                GgufTensorInfo SliceRows(int rows)
                {
                    var dims = (long[])qkvInfo.Dimensions.Clone();
                    dims[^1] = rows;
                    return qkvInfo with { Dimensions = dims };
                }
                _wq[i] = new TensorRef($"blk.{i}.attn_q.weight", SliceRows(qDim), qkvInfo.DType, qkvBase);
                _wk[i] = new TensorRef($"blk.{i}.attn_k.weight", SliceRows(kvDim), qkvInfo.DType, qkvBase + (long)qDim * bytesPerRow);
                _wv[i] = new TensorRef($"blk.{i}.attn_v.weight", SliceRows(kvDim), qkvInfo.DType, qkvBase + (long)(qDim + kvDim) * bytesPerRow);
            }
            else if (_isMla)
            {
                // "Lite" MLA checkpoints (DeepSeek-V2-Lite) have q_lora_rank==0: Q is a plain
                // per-head projection, same tensor name and layout as any other architecture's
                // wq. Full-size DeepSeek-V2/V3/R1 (q_lora_rank>0, separate wq_a/wq_b + a q-side
                // RMSNorm) are NOT handled -- ResolveTensor will throw "Missing tensor:
                // blk.N.attn_q.weight" on such a checkpoint rather than silently mis-loading it.
                _wq[i] = ResolveTensor($"blk.{i}.attn_q.weight");
                _wKvAMqa![i] = ResolveTensor($"blk.{i}.attn_kv_a_mqa.weight");
                _kvANorm![i] = ResolveTensor($"blk.{i}.attn_kv_a_norm.weight");
                // Only the legacy unsplit wkv_b tensor is handled (see MlaComputeKv's doc
                // comment). A GGUF shipping the split wk_b/wv_b "absorption" layout instead has
                // no attn_kv_b tensor at all, so this throws "Missing tensor" rather than loading
                // nothing and producing silently-wrong attention.
                _wKvB![i] = ResolveTensor($"blk.{i}.attn_kv_b.weight");
            }
            else
            {
                _wq[i] = ResolveTensor($"blk.{i}.attn_q.weight");
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
            }

            // DeepSeek-V2/V3: the first LeadingDenseBlockCount layers are plain dense FFN
            // (blk.0 here — leading_dense_block_count=1), MoE only kicks in from that layer
            // on. hp.IsMoE alone is a MODEL-level flag ("this architecture has MoE layers
            // somewhere"); every other MoE architecture this engine supports (Qwen3-MoE,
            // OLMoE, Mixtral, ...) has NO leading dense layers, so LeadingDenseBlockCount is 0
            // and this collapses to the original "IsMoE ⇒ every layer" check for them.
            if (hp.IsMoE && i >= hp.LeadingDenseBlockCount)
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
                // Apertus has no ffn_gate tensor at all (non-gated FFN, xIELU activation) —
                // _wGate[i] is left at its default (DataPtr null), which DenseFfn/PrefillCore
                // both check to pick the non-gated path. Every other dense architecture declares
                // ffn_gate unconditionally, so this lookup is not itself optional for them.
                //
                // GLM4 is a THIRD case: no ffn_gate tensor either, but NOT non-gated — ffn_up is
                // fused gate+up into one tensor at double width (blk.N.ffn_up.weight, row count
                // 2*intermDim), the same contiguous-row-offset packing GPT-NeoX's fused attn_qkv
                // uses. Confirmed against ggml_vec_swiglu_f32 (the compute kernel
                // ggml_swiglu(cur) — no separate "b" tensor — actually runs): the FIRST half of
                // rows is the gate input (SiLU applied) and the SECOND half is the up input
                // (multiplied directly, no activation) — y = SiLU(rows[0:n]) * rows[n:2n]. Split
                // by byte offset into two independent TensorRefs pointing into the same backing
                // tensor, with no data copy, then fall through to the ordinary SiLU-gated
                // MatVecDual/SiLuMul dispatch completely unchanged.
                var ffnGateInfo = _model.FindTensor($"blk.{i}.ffn_gate.weight");
                var ffnUpInfo = _model.FindTensor($"blk.{i}.ffn_up.weight");
                if (ffnGateInfo is null && ffnUpInfo is { } fusedUpInfo
                    && fusedUpInfo.Dimensions[^1] == 2L * _intermDim)
                {
                    byte* fusedBase = _model.GetTensorDataPtr(fusedUpInfo);
                    int fusedBytesPerRow = (_embDim / DTypeInfo.BlockSize(fusedUpInfo.DType))
                                        * DTypeInfo.BytesPerBlock(fusedUpInfo.DType);
                    // Each half needs its OWN Info with the halved row count, not the fused
                    // tensor's Info verbatim — PrefaultWeights sizes its read range from
                    // Info.ByteSize, oblivious to any row-offset baked into DataPtr. Reusing the
                    // full-width Info for the second half (_wUp) computed a prefault range that
                    // started halfway through the tensor and read a further FULL fused-width
                    // past that point — for the last layer in the file, that ran off the end of
                    // the mmap entirely (measured: AccessViolationException in
                    // MmapPrefault.StrideRead on a real GLM4 checkpoint, not a per-layer-count
                    // rounding issue — every layer's _wUp region was oversized, only the last
                    // layer's overrun had nowhere left to land safely).
                    var halfDims = (long[])fusedUpInfo.Dimensions.Clone();
                    halfDims[^1] = _intermDim;
                    var halfInfo = fusedUpInfo with { Dimensions = halfDims };
                    _wGate[i] = new TensorRef($"blk.{i}.ffn_gate.weight", halfInfo, halfInfo.DType, fusedBase);
                    _wUp[i] = new TensorRef($"blk.{i}.ffn_up.weight", halfInfo, halfInfo.DType,
                        fusedBase + (long)_intermDim * fusedBytesPerRow);
                }
                else
                {
                    if (ffnGateInfo is not null)
                        _wGate[i] = ResolveTensor($"blk.{i}.ffn_gate.weight");
                    _wUp[i] = ResolveTensor($"blk.{i}.ffn_up.weight");
                }
                _wDown[i] = ResolveTensor($"blk.{i}.ffn_down.weight");
            }

            if (_hasAttnBias)
            {
                if (_model.FindTensor($"blk.{i}.attn_qkv.bias") is not null)
                {
                    int qDim = _numHeads * layerHd;
                    int kvDim = _numKvHeads * layerHd;
                    // Load the fused bias into one scratch buffer, then copy each slice into
                    // its OWN allocation. _bq[i]/_bk[i]/_bv[i] must never alias a shared
                    // buffer: Dispose() frees all three independently (every other bias array
                    // in this file is its own allocation), and NativeMemory.Free on a pointer
                    // that isn't a block's own allocation start corrupts the native heap.
                    float* qkvBiasScratch = LoadBias($"blk.{i}.attn_qkv.bias", qDim + kvDim + kvDim);
                    _bq[i] = Alloc(qDim);
                    _bk[i] = Alloc(kvDim);
                    _bv[i] = Alloc(kvDim);
                    Copy(_bq[i], qkvBiasScratch, qDim);
                    Copy(_bk[i], qkvBiasScratch + qDim, kvDim);
                    Copy(_bv[i], qkvBiasScratch + qDim + kvDim, kvDim);
                    NativeMemory.Free(qkvBiasScratch);
                }
                else
                {
                    _bq[i] = LoadBias($"blk.{i}.attn_q.bias", _numHeads * layerHd);
                    if (!kvShared)
                    {
                        _bk[i] = LoadBias($"blk.{i}.attn_k.bias", _numKvHeads * layerHd);
                        _bv[i] = LoadBias($"blk.{i}.attn_v.bias", _numKvHeads * layerHd);
                    }
                }
                // Output-projection bias is optional (Qwen2 omits it; left null when absent).
                if (_hasAttnOutputBias)
                    _bo[i] = LoadBias($"blk.{i}.attn_output.bias", _embDim);
            }

            if (_hasNormBias)
            {
                _bAttnNorm![i] = LoadBias($"blk.{i}.attn_norm.bias", _embDim);
                // Falcon: no ffn_norm.bias tensor either — mirrors the weight fallback above.
                _bFfnNorm![i] = _model.FindTensor($"blk.{i}.ffn_norm.bias") is not null
                    ? LoadBias($"blk.{i}.ffn_norm.bias", _embDim)
                    : _bAttnNorm[i];
            }
            if (_hasFfnBias)
            {
                _bFfnUp![i] = LoadBias($"blk.{i}.ffn_up.bias", _intermDim);
                _bFfnDown![i] = LoadBias($"blk.{i}.ffn_down.bias", _embDim);
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

        // OLMo v1 has no output_norm tensor at all (UsesUnweightedNorm) — left at its default
        // (DataPtr null); RunTrunk's final-norm step calls SimdKernels.PureLayerNorm directly
        // instead of dereferencing a weight that doesn't exist.
        _outputNorm = model.FindTensor("output_norm.weight") is not null
            ? ResolveTensor("output_norm.weight")
            : default;
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

            if (IsMoeLayer(i))
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
    /// Real implementation of <see cref="IForwardPass.LastHidden"/> for the CPU backend: exposes
    /// the persistent post-final-norm hidden-state buffer (<c>_hidden</c>, `[embDim]`) that
    /// <c>ForwardPass.Decode.cs</c>'s single-token <see cref="Forward(int, int)"/> and the
    /// prefill path already compute and overwrite in place on every call -- this is a real,
    /// pre-existing buffer, not new state; the property just makes it externally readable.
    /// Real, documented caveat inherited from the buffer's own single-slot semantics (same
    /// contract the interface doc already states): valid only until the NEXT `Forward`/
    /// `Prefill`/`ForwardEmbedding` call overwrites `_hidden` -- callers needing a stable
    /// snapshot must copy it immediately. Enables real cross-model hidden-state bridging (e.g.
    /// a second, smaller transformer conditioned on this model's last hidden state) without any
    /// other Engine change -- see docs/audio-review-progress.md's QwenTTS Talker/Code Predictor
    /// entries for the real motivating case this was added for.
    /// </summary>
    public ReadOnlySpan<float> LastHidden => new(_hidden, _embDim);

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
    /// <summary>
    /// Resets the underlying KV cache, allowing zero-allocation reuse of this ForwardPass instance across generation sessions.
    /// </summary>
    public void ResetKvCache() => _kvCache.Reset();

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
        //
        // Per-layer head-dim models (Gemma 4) ALWAYS fall back to sequential Forward here, and not
        // because of unimplemented strides — issue #351 plumbed per-layer qDim/kvDim through the
        // batched blocks (buffers sized from _maxHeadDim, RoPE/Q-K-norm via ApplyRopeLayer,
        // PrefillCoreAttention deriving headDim per layer). Gemma4 needs real features PrefillCore
        // doesn't implement at all: per-layer KV head count (MQA/GQA mix), KV-layer sharing across
        // layers (_layerKvSrc), attention_k_eq_v, a per-head V norm before the cache write, and —
        // the actual blocker — sliding-window attention, which PrefillCoreAttention has no
        // windowSize parameter for. SnapKV eviction also isn't covered (SnapKvSelector assumes one
        // model-wide head dim), so per-layer models with SnapKV active take this path too.
        //
        // Note for onAllPositionLogits callers: this fallback never calls MatMulBatched, so it
        // cannot exercise Q8PrefillEnabled — a caller diagnosing that path specifically should
        // confirm the model isn't MoE / doesn't have per-layer head dims first.
        //
        // STINGRAY_PER_LAYER_HD_PREFILL=1 used to force the batched path anyway, to make the
        // remaining work measurable. Measured (2026-08-07): forcing it doesn't just produce wrong
        // output, it produces an AccessViolationException — the batched path indexes KV at the
        // model-wide head dim (512) on layers that actually carry 256, walking off the buffer end.
        // A path that corrupts memory can't be timed, so the flag now fails fast with an
        // explanation instead of forcing a route that was never bounds-safe.
        //
        // Earlier framings of this routing decision (superseded, kept for the per-layer plumbing
        // history): docs/reference/forwardpass-investigation-log.md
        // #gemma-4-per-layer-head-dim-batched-prefill--superseded-framings
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
        // Post-attention/post-FFW norm (OLMo2, Gemma 4 dense — see MoeBatchedPrefillSupported's
        // doc comment for the MoE case above) is applied only on the sequential RunTrunk path;
        // PrefillCore's batched loop has no equivalent step. Gemma 4 never reaches here at all
        // (perLayerHdUnsupported already routes it away), so this specifically covers OLMo2 and
        // any future dense post-norm architecture.
        bool postNormUnsupported = _postAttnNorm is not null || _postFfwNorm is not null;
        // Sliding-window attention (Command-R/cohere2): PrefillCoreAttention has no windowSize
        // parameter at all — it was never taught SWA masking, because Gemma 4 (the only prior
        // SWA architecture) is always routed away by perLayerHdUnsupported above before reaching
        // this check. cohere2 has SWA WITHOUT per-layer head dims, so it would otherwise slip
        // through and silently attend to the full context on every layer instead of the intended
        // window. Routing it to the sequential path reuses RunTrunk's Attention(), which already
        // threads windowSize correctly (proven by every Gemma 4 receipt).
        bool swaUnsupported = _isSwaLayer is not null && _layerHeadDim is null;
        // OLMo v1 (UsesUnweightedNorm): PrefillCore's batched norm steps only know how to skip a
        // null-DataPtr norm tensor entirely (OLMo2's convention) or apply a weighted one — never
        // taught the third case, "normalize anyway with no learned parameters at all". Routing to
        // the sequential path reuses RunTrunk's fix instead of teaching PrefillCore a third norm
        // mode for a single architecture.
        bool unweightedNormUnsupported = _usesUnweightedNorm;
        if (moeUnsupported || perLayerHdUnsupported || postNormUnsupported || swaUnsupported
            || unweightedNormUnsupported)
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
    /// <param name="allowBlas">
    /// Forwarded to <see cref="SimdKernels.MatMulBatched"/> (and gates the dequant-cache BLAS
    /// path above it). Defaults to true; <see cref="PrefillPackedMulti"/> passes false because its
    /// rows span multiple INDEPENDENT sessions' prompts, not positions within one prompt -- see
    /// that parameter's doc for why batch-composition-dependent kernel choice is unsafe there.
    /// </param>
    private void MatMulBatchedCached(float* output, in TensorRef w, float* input,
        int N, int rows, int cols, bool allowBlas = true)
    {
        // Repacked 8-row Q4_K path (perf-loop iteration 42): measured 2.6x over the row-major
        // _8In at the trunk's Q4_K shape, and separately measured faster end-to-end than the
        // issue #189 dequant-cache/BLAS path below (147 vs 74 t/s prefill, SmolLM2-1.7B-Q4_K_M,
        // CPU baseline) once that path started auto-engaging merely because libopenblas.dll was
        // present on disk. Checked first for that reason: BLAS availability is not evidence BLAS
        // is faster than this kernel for the same tensor, so it must not win the race by ordering
        // alone. See docs/cpu-performance-baseline.md.
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

        if (allowBlas && _dequantCacheEnabled && w.DType != DType.Float32 && N >= SimdKernels.MinBatchForBlas)
        {
            float* wf32 = GetDequantWeightF32(in w, rows, cols);
            if (wf32 != null)
            {
                SimdKernels.MatMulBatchedF32(output, wf32, input, N, rows, cols);
                return;
            }
        }

        SimdKernels.MatMulBatched(output, w.DataPtr, input, N, rows, cols, w.DType, allowQ8: true, allowBlas: allowBlas);
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
        float* input, int N, int rows, int cols, bool allowBlas = true)
    {
        bool useCache1 = allowBlas && _dequantCacheEnabled && w1.DType != DType.Float32 && N >= SimdKernels.MinBatchForBlas;
        bool useCache2 = allowBlas && _dequantCacheEnabled && w2.DType != DType.Float32 && N >= SimdKernels.MinBatchForBlas;

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
            MatMulBatchedCached(output1, in w1, input, N, rows, cols, allowBlas);
            MatMulBatchedCached(output2, in w2, input, N, rows, cols, allowBlas);
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
        MatMulBatchedCached(output1, in w1, input, N, rows, cols, allowBlas);
        MatMulBatchedCached(output2, in w2, input, N, rows, cols, allowBlas);
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

}

