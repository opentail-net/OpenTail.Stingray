namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
    // ernie4_5 (dense path) -- ADMITTED 2026-09-01. Was diagnosed as blocked purely on the
    // tokenizer axis (same shape as minicpm/internlm2/baichuan: tokenizer.ggml.model=llama with
    // tokenizer.ggml.scores present and no merges, already fixed by
    // GgufTokenizer.SpmMergePiecesByScore), and the forward pass needed zero new code (dense,
    // non-MoE branch is a plain RMSNorm pre-norm/SiLU-gated-FFN/GQA/full-RoPE trunk, identical
    // shape to exaone -- confirmed against examples/llama.cpp/llama.cpp/src/models/
    // ernie4-5.cpp). First greedy-parity attempt found a REAL THIRD tokenizer bug, though: a
    // literal newline mid-prompt diverged from the reference at exactly one token (engine
    // produced UnknownTokenId=0, reference produced 23 = "<0x0A>"). Root cause: GgufTokenizer.
    // EncodeSpm's "no direct vocab entry after merging" branch emitted a single UnknownTokenId
    // for the whole unmatched piece, instead of real llama.cpp's per-UTF8-BYTE SentencePiece
    // byte-fallback lookup (llm_tokenizer_spm_session::resegment's "output any symbols that did
    // not form tokens as bytes" branch -> llama_vocab::byte_to_token, tries "<0xXX>" uppercase
    // hex first, then the raw single-byte string, only falling to UNK if neither exists) -- a
    // real, general SPM gap distinct from the two xverse-motivated fixes (SpmMergePiecesByScore
    // made merging work at all; this is the still-unmatched-after-merging TAIL case neither of
    // those touched). Fixed via GgufTokenizer.AppendSpmByteFallback, covered by
    // SpmMergeByScoreTests.EncodeSpm_PieceWithNoDirectVocabEntry_UsesByteFallbackToken_NotUnk.
    //
    // Checkpoint: baidu/ERNIE-4.5-0.3B-PT (Apache-2.0, genuinely permissive -- bucket-1), via
    // bartowski/baidu_ERNIE-4.5-0.3B-PT-GGUF, Q8_0 (386 MB). Transient local download, never
    // vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-09-01, tools/llama.cpp llama-tokenize/llama-server, CPU
    // backend), AFTER the byte-fallback fix: templated prompt "<|begin_of_sentence|>User: The
    // capital of France is\nAssistant: " tokenizes byte-for-byte identically to llama-tokenize
    // (20 tokens incl. BOS). Greedy continuation FULL 8-of-8-token exact match through EOS:
    // engine and llama-server both produce [700, 9689, 315, 10298, 357, 11855, 93937, 2] ("The
    // capital of France is Paris." + </s>).
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake (the tokenizer fix itself IS covered, see above).
    public static readonly ArchitectureDescriptor Ernie45 = new()
    {
        Id = "ernie4_5",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // paddleocr — PaddleOCR-VL 1.6 text decoder (ERNIE-4.5-0.3B-shaped, llama.cpp src/models/paddleocr.cpp:
    // the qwen2vl graph with LLAMA_ROPE_TYPE_MROPE, sections [16,24,24,0]). ADMITTED 2026-09-27.
    // Two fixes: NEOX rotation (M-RoPE rotates NeoX-style; it had fallen through to interleaved) and
    // 2D M-RoPE positions for image tokens (ForwardPass.AddMRopeImage, mirrors mtmd MTMD_POS_TYPE_MROPE).
    // Evidence (paddleocr-vl-1.6.gguf, CPU): wikitext [512,1024) PPL 39947 vs llama-perplexity 40027
    // (an OCR model, hence the size); OCR of a rendered "Invoice 4217 / Total: 38.50 EUR" image is
    // token-for-token identical to llama-server --mmproj (16 tokens + EOS). The vision encoder is pinned
    // by LlamaMtmdVisionParityTests.PaddleOcr_Rainbow448_MatchesLlamaMtmdDebug.
    public static readonly ArchitectureDescriptor Paddleocr = new()
    {
        Id = "paddleocr",
        Traits = new() { ReadsRopeDimensionSections = true },
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Vision: PaddleOCR-VL",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    /// <summary>
    /// Command-R: sliding-window attention alternates every swaPeriod layers with the LAST layer of each block global
    /// (llama.cpp set_swa_pattern(period), dense_first=false); RoPE only on SWA layers; true (bias-less) LayerNorm;
    /// parallel residual; logit_scale applied as a direct multiply (cohere2.cpp), unlike Granite's reciprocal.
    /// </summary>
    private static ModelHyperparams Cohere2Semantics(ModelArchitectureSemanticsContext ctx)
    {
        var hp = ctx.Baseline;
        int numLayers = hp.NumLayers;
        IReadOnlyList<bool>? isSwa = null;
        if (numLayers > 0)
        {
            int swaPeriod = ctx.Int("attention.sliding_window_pattern", 4);
            var swa = new bool[numLayers];
            for (int i = 0; i < numLayers; i++)
                swa[i] = swaPeriod == 0 || (i % swaPeriod < swaPeriod - 1);
            isSwa = swa;
        }
        float rawLogitScale = ctx.Float("logit_scale");
        return hp with
        {
            SlidingWindowSize = numLayers > 0 ? ctx.Int("attention.sliding_window") : hp.SlidingWindowSize,
            IsSwaLayer = isSwa ?? hp.IsSwaLayer,
            UseParallelResidual = true,
            UsesLayerNorm = true,
            RopeOnlySwaLayers = true,
            LogitScale = rawLogitScale != 0f ? rawLogitScale : hp.LogitScale,
        };
    }

    // glm4 (non-multimodal/text-only) — admitted 2026-08-09 on a 14-of-24-token exact prefix
    // (then a documented 0.0214-logit near-tie, the deepest-position/tightest-margin near-tie
    // accepted this session). Much smaller in scope than earlier estimated: the
    // "conditional/multi-section RoPE" this plan originally flagged only applies to the
    // multimodal (use_mrope) branch — a text-only checkpoint takes ordinary ggml_rope_ext,
    // already fully supported. The sandwich-norm pattern (pre-norm AND post-norm on both
    // attention and FFN) is Gemma 4's own shape, already generalized to plain tensor-presence
    // detection while building the OLMo2 receipt. The one genuinely new mechanism: ffn_up is a
    // single FUSED tensor at double width (no separate ffn_gate) — confirmed against
    // ggml_vec_swiglu_f32's actual math (first half = gate/SiLU, second half = up/multiplied)
    // — split by byte offset into independent TensorRefs, the same pattern GPT-NeoX's fused
    // attn_qkv already established. See Glm4GreedyParityTests / examples/llama.cpp/
    // llama.cpp/src/models/glm4.cpp and docs/done/01-gguf-model-coverage-plan.md for the receipt,
    // including two real defects found and fixed while building it: a fused-tensor-slice
    // prefault-sizing bug that actually crashed with AccessViolationException (and was fixed
    // retroactively for the pre-existing GPT-NeoX/Falcon fused-QKV split too, since it's the
    // identical latent defect there, just not yet triggered), and a wholly missing partial-RoPE
    // implementation for the "normal"/interleaved (non-NEOX) rotation convention
    // (SimdKernels.ApplyRoPECachedPartial, new).
    public static readonly ArchitectureDescriptor Glm4 = new()
    {
        Id = "glm4",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // glm4moe (GLM-4.5-Air) — admitted 2026-10-04: Goldens/glm4moe.golden.json teacher-forced against llama-server on GLM-4.5-Air-Q2_K,
    // all 54 positions of a short and a 190-token prompt match (30 confident, 0 near-tie differences); second-half wikitext PPL
    // (-c 512) 3.3709 vs llama-perplexity 3.4260 +/- 0.41. 128 experts top-8, sigmoid gating with selection bias, one shared expert,
    // one leading dense layer. Only the Q2_K quantisation has been run.
    public static readonly ArchitectureDescriptor Glm4moe = new()
    {
        Id = "glm4moe",
        Traits = new() { DefaultExpertGatingFunc = 2 },
        UsesNeoxRope = true,
        // glm4-moe.cpp names the pre-FFN norm post_attention_norm: it is the FFN norm here, not a Gemma/OLMo2-style post-attention-output norm.
        ApplyModelSemantics = ctx => ctx.Baseline with { PostAttnNormIsFfnNorm = true, HasPostAttnNorm = false },
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "GLM-4.5-Air (`glm4moe`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
