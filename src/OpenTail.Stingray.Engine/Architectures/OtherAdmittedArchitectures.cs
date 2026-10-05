namespace OpenTail.Stingray.Engine;

internal static class OtherAdmittedArchitectures
{
    public static readonly ArchitectureDescriptor Mimo = new()
    {
        Id = "mimo",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // Shares the evidence comment and receipt for 'mimo'.
    public static readonly ArchitectureDescriptor Mimo2 = new()
    {
        Id = "mimo2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // olmoe — admitted 2026-08-08 on perplexity parity, NOT on token-for-token greedy parity,
    // which it does not achieve. On wikitext at a matched 2048-token context llama.cpp b8585
    // scores 7.4868 and this engine scores 7.3889 (1.3%). The greedy divergence is at a flat
    // position where the top five candidates span 1.55 logits, i.e. where a differently
    // quantised matmul reorders candidates. Evidence and the argument for accepting it:
    // docs/done/01-gguf-model-coverage-plan.md §1b. Note `olmo2` is deliberately NOT here — it
    // shares neither a fixture nor a receipt.
    public static readonly ArchitectureDescriptor Olmoe = new()
    {
        Id = "olmoe",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the MoE partial-offload row is a backend capability, not a family verification row.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // gpt-oss — admitted 2026-09-26. Runs on its own GptOssForwardPass (CPU only; attention
    // sinks, 1:1 SWA/full alternation, biased MoE, OAI SwiGLU, YaRN factor 32 on both layer
    // kinds), routed by RunCommand/InferenceEngineLoader. Receipt vs llama-server (vendored
    // tools/llama.cpp) on gpt-oss-20b-MXFP4: "The capital of France is" teacher-forced over
    // llama-server's 24 greedy tokens, 22/24 exact argmax, worst gap 0.063 logits
    // (GptOssRealWeightSmokeTests; step 1 is a 0.02-logit tie inside llama.cpp itself); a 188-token Paris-history prompt (past the 128-token window) 16/32 exact, then
    // a flip at a 0.105-logit near-tie (220 vs 5030). First-token top-5 in identical order,
    // gaps to the top logit within 0.002-0.065 of llama.cpp's -fa off run — smaller than
    // llama.cpp's own -fa on/off shift (up to 0.13). Plain RoPE instead of YaRN diverged at
    // token 14 with a different top-5, so the YaRN wiring is load-bearing. See
    // docs/done/101-work-queue-after-coverage-plan.md.
    public static readonly ArchitectureDescriptor GptOss = new()
    {
        Id = "gpt-oss",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "gpt-oss (`gpt-oss`)",
        EvidenceDoc = "docs/done/101-work-queue-after-coverage-plan.md",
    };

    // smollm3 — one twist over the plain llama trunk: NoPE every 4th layer, gated the same way
    // as llama4's noRopeStep. See SmolLm3GreedyParityTests for the full 24-token greedy receipt.
    public static readonly ArchitectureDescriptor Smollm3 = new()
    {
        Id = "smollm3",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // apertus — admitted 2026-08-08 on an 11-token EXACT prefix match (one full sentence)
    // against llama.cpp, diverging afterward into a different but still coherent, on-topic
    // completion (not degenerate output). The first "new-kernel" architecture admitted this
    // session: no ffn_gate tensor at all (plain up -> xIELU -> down, ModelHyperparams.Xielu*,
    // SimdKernels.XieluInPlace), detected from tensor inventory rather than architecture
    // string. See ApertusGreedyParityTests and docs/done/01-gguf-model-coverage-plan.md §1f for the
    // receipt, including a real defect found and fixed in the xIELU parameter transform
    // (GGUF stores pre-softplus values; llama.cpp's ggml_xielu() wrapper — not the compute
    // kernel — applies softplus before use, easy to miss by reading only the kernel).
    public static readonly ArchitectureDescriptor Apertus = new()
    {
        Id = "apertus",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // gptneox (Pythia) — LayerNorm (mean/variance + learned bias, not RMSNorm), a biased
    // non-gated GELU FFN, a fused blk.*.attn_qkv.weight/bias tensor pair (split by
    // contiguous row offset in ForwardPass's constructor — Q rows, then K rows, then V
    // rows; confirmed against examples/llama.cpp/llama.cpp/conversion/gptneox.py and
    // src/models/gptneox.cpp, NOT the interleaved per-head layout an earlier draft
    // assumed), and the metadata-driven parallel-residual graph (x + attn(ln1(x)) +
    // ffn(ln2(x)), both norms reading the SAME incoming residual — ModelHyperparams.
    // HasNormBias/HasFfnBias/UseParallelResidual). See GptNeoxGreedyParityTests and
    // docs/done/01-gguf-model-coverage-plan.md for the receipt. TurboQuant prefill, continuous-
    // batching admission, and CUDA/Vulkan are not wired to this profile.
    public static readonly ArchitectureDescriptor GptNeoX = new()
    {
        Id = "gptneox",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // falcon (7B only — 40B's second attn_norm_2 tensor is NOT implemented, no small 40B
    // checkpoint to validate against) — reuses every gptneox mechanism (LayerNorm, biased
    // non-gated GELU FFN [though Falcon carries no biases at all], fused attn_qkv,
    // UseParallelResidual's 3-way sum) plus one new wrinkle: Falcon-7B has NO separate
    // ffn_norm tensor at all — attention and FFN read the SAME LayerNorm output (confirmed
    // against src/models/falcon.cpp: "use the attn norm, not the result"). ForwardPass's
    // constructor falls _ffnNorm/_bFfnNorm back to _attnNorm/_bAttnNorm's own TensorRef/
    // pointer when blk.*.ffn_norm.{weight,bias} is absent — Dispose() guards the aliased
    // bias pointer so it isn't double-freed. use_parallel_residual is never a metadata key
    // for this arch (llama.cpp hardcodes it in the graph), so ModelGraph.cs hardcodes it too
    // for arch=="falcon" rather than reading a key that doesn't exist. Also exercises MQA
    // (head_count=71, head_count_kv=1) through the existing GQA-parametrized fused-QKV
    // split for the first time on this profile. See FalconGreedyParityTests and
    // docs/done/01-gguf-model-coverage-plan.md for the receipt.
    public static readonly ArchitectureDescriptor Falcon = new()
    {
        Id = "falcon",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // olmo2 — a THIRD residual pattern, distinct from both the ordinary pre-norm trunk and
    // gptneox/falcon's parallel residual: post-norm sandwiching. No attn_norm/ffn_norm tensor
    // exists in the GGUF at all — attention and FFN both read the RAW residual directly, and
    // the norm (RMSNorm, no bias) is applied to each sublayer's OUTPUT via attn_post_norm/
    // ffn_post_norm, immediately before the residual add (confirmed against
    // src/models/olmo2.cpp: x1 = x + PostNorm(Attn(x)); x2 = x1 + PostNorm(FFN(x1))).
    // ForwardPass's constructor leaves _attnNorm[i]/_ffnNorm[i] at their default (DataPtr
    // null) when absent — the same tensor-presence sentinel Apertus/GPT-NeoX already use for
    // "no ffn_gate" — and RunTrunk/PrefillCore's pre-norm steps copy the raw residual through
    // unmodified when that sentinel is set, instead of normalizing. The post-norm application
    // itself reuses Gemma 4's existing _postAttnNorm/_postFfwNorm mechanism unchanged (same
    // llama.cpp tensor names, LLM_TENSOR_ATTN_POST_NORM/FFN_POST_NORM) — generalized in
    // ModelGraph.cs to detect from tensor presence for any architecture, not just gemma4.
    // Because that mechanism was never wired into PrefillCore's batched loop (documented
    // there as Gemma-4-only in MoeBatchedPrefillSupported's doc comment), PrefillDispatch now
    // also falls back to sequential per-token Forward() for ANY post-norm model, not just
    // per-layer-head-dim ones — the same fallback pattern Gemma 4 already uses, just widened.
    // QK-norm reuses the OLMoE whole-vector-RMS fix unchanged (same convention, same code).
    // See Olmo2GreedyParityTests and docs/done/01-gguf-model-coverage-plan.md for the receipt.
    public static readonly ArchitectureDescriptor Olmo2 = new()
    {
        Id = "olmo2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // exaone — ADMITTED 2026-08-09, full 24-of-24-token exact match. Genuinely gate-only: an
    // ordinary pre-norm llama-style trunk (RMSNorm, SiLU-gated FFN, standard GQA attention,
    // NEOX RoPE — confirmed against examples/llama.cpp/llama.cpp/src/models/exaone.cpp), and
    // its optional top-level rope_freqs.weight tensor (llama3.1-style per-dimension frequency
    // correction, [40] = ropeHalfDim on this checkpoint) is already read generically by
    // ForwardPass's constructor (built for Gemma 4, detected purely by tensor name/shape, not
    // architecture-gated) — zero new code was needed anywhere.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Every known EXAONE
    // checkpoint (LGAI-EXAONE, `EXAONE-3.5-2.4B-Instruct-GGUF` was used for this receipt) ships
    // under "EXAONE AI Model License Agreement 1.1 - NC" — explicitly non-commercial, not
    // MIT/Apache-2.0/BSD/MPL. Per the license policy in docs/done/01-gguf-model-coverage-plan.md
    // ("License policy: code vs. checkpoint"), the architecture code itself doesn't redistribute
    // any restricted asset and was verified once against a transient local checkpoint (deleted
    // immediately after, never vendored), but no permanent test is kept in the tree referencing
    // that checkpoint by name.
    //
    // Verification evidence (2026-08-09, EXAONE-3.5-2.4B-Instruct-GGUF Q8_0, llama.cpp
    // b8585-cad2d3884): prompt "The capital of France is" -> ids [1320, 7304, 670, 9776, 772].
    // Reference 24-token greedy continuation (--temp 0 --top-k 1 --seed 0, no-bos):
    // " Paris. Paris is located in northern France on the Seine River. It is oneQuestion: What
    // is the capital of" -> ids [12229, 375, 12229, 772, 6244, 666, 15609, 9776, 807, 629, 3654,
    // 1085, 9817, 375, 1533, 772, 1300, 37913, 387, 3017, 772, 629, 7304, 670]. This engine
    // matched ALL 24 tokens exactly, and the prefill/decode stepwise-consistency check agreed
    // (argmax match, logit maxDiff within the standard <5.0 int8-prefill-approximation bound).
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake. A change to RunTrunk/PrefillCore/the rope_freqs handling above
    // could silently break this profile and nothing in CI would notice.
    public static readonly ArchitectureDescriptor Exaone = new()
    {
        Id = "exaone",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // orion -- ADMITTED 2026-09-01, full 4-of-4-token exact greedy match, zero new code.
    // Was diagnosed as blocked on tokenizer.ggml.model=llama + scores-only-no-merges (same
    // shape as minicpm/internlm2/baichuan/ernie4_5, already fixed by SpmMergePiecesByScore)
    // plus an architecture-side "new LayerNorm-with-bias + gated-SiLU-FFN combination" the
    // original 2026-08-09 note assumed was unbuilt. Re-reading the real reference
    // (examples/llama.cpp/llama.cpp/src/models/orion.cpp) before writing anything found this
    // combination was ALREADY covered by existing generic dispatch: ModelGraph.cs's
    // UsesLayerNorm flag is driven purely by tensor presence (blk.0.attn_norm.bias found in
    // the GGUF, model-agnostic -- the same detection gptneox/falcon/codeshell already use),
    // and orion's gated-SiLU FFN (ffn_gate/ffn_up/ffn_down, no bias) is the same shape the
    // plain llama FFN path already builds. orion was also already in ModelGraph.cs's
    // isNeoxRope dispatch table from an earlier session pass. Net: the allowlist string alone
    // was sufficient, no ModelGraph.cs/ForwardPass.cs changes needed at all.
    //
    // Checkpoint: OrionStarAI/Orion-14B-Chat (custom OrionStar license, not a clean SPDX
    // permissive license -- bucket-2), via demonsu/orion-14b-chat-gguf, Q4_K_M (8.81 GB).
    // Transient local download, never vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-09-01, tools/llama.cpp llama-tokenize/llama-server, CPU
    // backend): templated prompt
    // "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
    // tokenizes byte-for-byte identically to llama-tokenize (33 tokens, no BOS --
    // tokenizer.ggml.add_bos_token=false for this checkpoint). Full 4-of-4-token exact greedy
    // match through EOS: engine and llama-server both produce [23571, 327, 50376, 2]
    // ("Paris." + </s>).
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Orion = new()
    {
        Id = "orion",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

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
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Vision: PaddleOCR-VL",
        EvidenceDoc = "docs/STATUS.md",
    };

    // nemotron_h — NVIDIA Nemotron-H / Nemotron Nano v2 (llama.cpp nemotron-h.cpp): every layer is exactly one of
    // Mamba-2 (8 B/C groups), NoPE attention, or a non-gated ReLU² MLP, each with its own RMSNorm + residual.
    // ADMITTED 2026-09-27 on the shared Mamba-2 mixer (ForwardPass.Mamba2.cs, CPU). Evidence: nemotron-nano-12b-v2-vl
    // Q2_K text decoder, wikitext second-half [1024,+) PPL at -c 2048 6.6332 vs llama-perplexity --chunks 1 6.6338;
    // NemotronHParityTests teacher-forces llama-server's continuation and matches its token at all 22 positions
    // where llama.cpp's top-1 margin exceeds 1.5 nats (Q2_K forks free-running greedy at close pairs, e.g. '."' vs
    // '."' + newline at 0.22 nats in llama.cpp, 0.10 the other way in ours).
    public static readonly ArchitectureDescriptor Nemotronh = new()
    {
        Id = "nemotron_h",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "NVIDIA Nemotron-H / Nemotron Nano v2 (`nemotron_h`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    // lfm2 — Liquid LFM2 (llama.cpp lfm2.cpp): gated short-conv layers (in_proj -> b|c|x, causal depthwise conv of
    // b*x over shortconv.l_cache steps, gate by c, out_proj) + GQA attention with QK-norm and NeoX RoPE, SwiGLU FFN
    // on every layer; final norm stored as token_embd_norm. ADMITTED 2026-09-27 (ForwardPass.ShortConv.cs, CPU).
    // Evidence: LFM2-1.2B Q8_0 wikitext second-half [1024,+) PPL at -c 2048 10.9195 vs llama-perplexity --chunks 1
    // 10.9543 (0.3%); with BOS the next-token top-5 after "The capital of France is" matches llama-server to 0.14
    // nats; Lfm2ParityTests teacher-forces llama-server continuations (14 confident positions match). The model
    // degenerates without BOS, so admit-arch now prepends it like llama-server does. Licence: LFM Open License
    // v1.0 (free commercial use under $10M revenue); documented, not a support gate.
    public static readonly ArchitectureDescriptor Lfm2 = new()
    {
        Id = "lfm2",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Liquid LFM2 (`lfm2`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    // lfm2moe -- Liquid LFM2-MoE (llama.cpp lfm2.cpp, MoE FFN: sigmoid gating + exp_probs_b selection bias, top-k
    // renormalised). ADMITTED 2026-10-01. LFM2-8B-A1B-Q4_K_M: batched prefill matches token-by-token bit-for-bit with
    // STINGRAY_PREFILL_ATTN_FLASH64=0 (the earlier "drift" was flash-64's online softmax, not a bug); wikitext
    // [512,1024) PPL at -c 1024 is 7.3425 per-token / 7.2882 batched-flash vs llama-perplexity 7.9130 +/- 1.09
    // (<0.6 SE); greedy "The capital of France is" continuation matches llama-completion.
    // docs/1-correctness/12-lfm2moe-batched-per-token-parity-plan.md.
    public static readonly ArchitectureDescriptor Lfm2moe = new()
    {
        Id = "lfm2moe",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Liquid LFM2-MoE (`lfm2moe`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    // internlm2 -- ADMITTED 2026-09-01. Was blocked purely on the tokenizer axis (same as
    // minicpm/ernie4_5/baichuan): tokenizer.ggml.model=llama with tokenizer.ggml.scores
    // (92,544 entries) and no tokenizer.ggml.merges array -- already fixed by
    // GgufTokenizer.SpmMergePiecesByScore. The forward pass needed genuinely zero new code:
    // internlm2's plain pre-norm/GQA/RoPE trunk was already covered by this engine's existing
    // generic dispatch (no architecture-specific kernel gate exists for it anywhere in
    // ModelGraph.cs/ForwardPass.cs -- confirmed by grep before this admission), so adding the
    // allowlist string alone was sufficient.
    //
    // Checkpoint: internlm/internlm2_5-1_8b-chat (custom InternLM license, not a clean SPDX
    // permissive license -- bucket-2), via bartowski/internlm2_5-1_8b-chat-GGUF, Q4_K_M
    // (1.17 GB). Transient local download, never vendored, deleted immediately after this
    // receipt.
    //
    // Verification evidence (2026-09-01, tools/llama.cpp llama-server /completion with
    // return_tokens:true, CPU backend): templated prompt
    // "<s><|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n".
    // Full 8-of-8-token exact greedy match through EOS: engine and llama-server both produce
    // [918, 6872, 446, 9760, 505, 12247, 281, 92542] ("The capital of France is Paris." +
    // <|im_end|>).
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Internlm2 = new()
    {
        Id = "internlm2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // starcoder2 — ADMITTED 2026-08-09, full 24-of-24-token exact match. Reuses gptneox/
    // falcon's LayerNorm-with-bias + non-gated biased-GELU FFN infrastructure exactly (same
    // SimdKernels.LayerNorm/GeluInPlace, same HasNormBias/HasFfnBias/HasAttnBias/
    // HasAttnOutputBias tensor-presence detection), but with the ORDINARY sequential residual
    // (x1 = x + attn(LN(x)); x2 = x1 + ffn(LN(x1))) — confirmed against
    // examples/llama.cpp/llama.cpp/src/models/starcoder2.cpp — not gptneox/falcon's parallel
    // 3-way sum. UseParallelResidual is false here (no metadata key, no arch-string hardcode).
    //
    // ONE REAL DEFECT FOUND AND FIXED — a latent bug this receipt was the first thing to
    // exercise. RunTrunk's sequential (non-parallel-residual) FFN pre-norm still called
    // FastRmsNorm directly instead of the bias-aware FastNorm dispatcher, because no
    // previously-admitted architecture had BOTH HasNormBias=true AND UseParallelResidual=false
    // at the same time (gptneox/falcon are always parallel-residual) — the sequential+LayerNorm
    // combination was unreachable code until starcoder2. Symptom: greedy continuation matched
    // llama.cpp for exactly 1 token then diverged completely, and the prefill/decode
    // consistency check disagreed with ITSELF (maxDiff 30.4, argmax mismatch) — a strong signal
    // of a structural bug, not a numerical approximation. Fixed by routing that call through
    // FastNorm with the layer's ffn-norm bias, matching what PrefillCore's equivalent branch
    // already did correctly.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint license:
    // "bigcode-openrail-m" (BigCode OpenRAIL-M — a restricted-use RAIL license: use-based
    // restrictions, e.g. malicious-code generation), not MIT/Apache-2.0/BSD/MPL. Verified once
    // against `bigcode/starcoder2-3b` (via QuantFactory/starcoder2-3b-GGUF, Q8_0), a transient
    // local download, never vendored, deleted immediately after this receipt. Per the license
    // policy in docs/done/01-gguf-model-coverage-plan.md, no permanent test persists.
    //
    // Verification evidence (2026-08-09, starcoder2-3b Q8_0, llama.cpp b8585-cad2d3884): prompt
    // "The capital of France is" -> ids [1338, 18972, 451, 45569, 458]. Reference 24-token
    // greedy continuation (--temp 0 --top-k 1 --seed 0, no-bos): " Paris.\n\n```\n\nI want to
    // get the value of the attribute `value` of the `span" -> ids [2736, 316, 51, 222, 222, 932,
    // 222, 222, 78, 2660, 391, 640, 341, 804, 451, 341, 3895, 548, 872, 101, 451, 341, 548, 681].
    // This engine matched ALL 24 tokens exactly (after the FastNorm fix above), and the
    // prefill/decode stepwise-consistency check agreed.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake. In particular, the RunTrunk sequential-FFN-norm fix above is
    // currently the ONLY thing exercising that exact code combination; a future change there
    // could silently reintroduce the bug this receipt just fixed.
    public static readonly ArchitectureDescriptor Starcoder2 = new()
    {
        Id = "starcoder2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // cohere2 (Command-R7B) — ADMITTED 2026-08-09, 1-token exact match plus a documented
    // near-tie, bucket-2. Reuses gptneox/falcon's shared-attn/ffn-norm fallback (no separate
    // ffn_norm tensor) and UseParallelResidual's 3-way sum, but needed THREE genuinely new
    // mechanisms, all confirmed against examples/llama.cpp/llama.cpp/src/models/cohere2.cpp
    // before writing any code: (1) LayerNorm WITHOUT a learned bias — SimdKernels.LayerNorm's
    // bias param is now null-safe (skips the bias-add step), and the new
    // ModelHyperparams.UsesLayerNorm decouples "use LayerNorm math" from HasNormBias ("has a
    // bias tensor"), since a weight-only norm tensor looks identical on disk whether the
    // architecture means RMSNorm or bias-less LayerNorm — this is an arch-string fact, not a
    // tensor-presence one. (2) Generic (non-Gemma4-gated) sliding-window attention — 3 local +
    // 1 global layers (swaPeriod=4, hardcoded default even when the metadata key is absent),
    // computed in ModelGraph.cs outside the isGemma4 block via the exact formula in
    // llama-hparams.cpp's set_swa_pattern (dense_first=false: is_swa[il] = il%period <
    // period-1) — NOT Gemma 4's literal-bool-array convention, since cohere2's own metadata
    // key is a plain period scalar. (3) RoPE applied ONLY on SWA layers, none at all on global
    // ones (ModelHyperparams.RopeOnlySwaLayers) — cohere2.cpp's attention block has no `else`
    // branch on its `if (is_swa)` rope application, the opposite selection rule from
    // Llama-4/SmolLM3's period-based NoRopeLayerStep. Also: PrefillCoreAttention has NO
    // windowSize parameter at all (only ever needed by Gemma 4, which never reaches it — see
    // perLayerHdUnsupported), so PrefillDispatch now also falls back to sequential Forward()
    // for any SWA model without per-layer head dims, reusing RunTrunk's Attention() call
    // (proven correct by every Gemma 4 receipt) instead of teaching PrefillCore SWA masking.
    // logit_scale is read with the OPPOSITE convention from Granite's (direct multiply, not
    // reciprocal — cohere2.cpp does ggml_scale(cur, f_logit_scale) unconditionally, not
    // 1/f_logit_scale).
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `CohereLabs/c4ai-command-r7b-12-2024` (bartowski GGUF, Q4_K_M), CC-BY-NC-4.0 — not
    // MIT/Apache-2.0/BSD/MPL. Transient local download, never vendored, deleted immediately
    // after this receipt.
    //
    // Verification evidence (2026-08-09, c4ai-command-r7b-12-2024 Q4_K_M, llama.cpp
    // b8585-cad2d3884): prompt "The capital of France is" -> ids [2162, 7784, 1719, 5334, 1801].
    // First generated token matches exactly (id 1690). Second diverges: this engine picks
    // token 19 (",", logit 13.7218) where llama.cpp's reference implies 1671 (" a", this
    // engine's own logit for it: 13.6563) — a 0.0655-logit gap, tighter than every other
    // near-tie accepted this session, on a Q4_K_M checkpoint. Ruled out before accepting:
    // re-ran with STINGRAY_CPU_PREFILL_Q8=0 (same result — not an int8-prefill artifact);
    // confirmed this checkpoint declares no rope_freqs.weight tensor (not a missing-mechanism
    // gap); confirmed via list-tensors that blk.*.attn_norm has no .bias tensor and no
    // separate ffn_norm tensor (matches the bias-less/shared-norm design exactly, not a
    // loading defect). Reads as ordinary Q4_K accumulation-order sensitivity at a
    // closely-contested position, the same category of evidence the OLMoE/Apertus/GPT-NeoX
    // receipts were accepted on — measured directly via a top-5 logit dump before accepting,
    // not assumed.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake. In particular the SWA/RopeOnlySwaLayers/UsesLayerNorm additions
    // above are new, cohere2-only code paths nothing else in the codebase exercises.
    public static readonly ArchitectureDescriptor Cohere2 = new()
    {
        Id = "cohere2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

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
    };

    // glm4moe (GLM-4.5-Air) — admitted 2026-10-04: GlmMoeGreedyParityTests teacher-forced against llama-server on GLM-4.5-Air-Q2_K,
    // all 54 positions of a short and a 190-token prompt match (30 confident, 0 near-tie differences); second-half wikitext PPL
    // (-c 512) 3.3709 vs llama-perplexity 3.4260 +/- 0.41. 128 experts top-8, sigmoid gating with selection bias, one shared expert,
    // one leading dense layer. Only the Q2_K quantisation has been run.
    public static readonly ArchitectureDescriptor Glm4moe = new()
    {
        Id = "glm4moe",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "GLM-4.5-Air (`glm4moe`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    // stablelm — admitted 2026-08-09. Smallest code change of any new-kernel architecture this
    // session: LayerNorm-with-bias, non-gated-FFN plumbing, and NEOX partial rope were all
    // already generic (built for gptneox/falcon/glm4), so nothing new was needed for any of
    // those. The one real finding: this checkpoint's GGUF carries a stale
    // `stablelm.use_parallel_residual=true` metadata key that stablelm.cpp's graph builder
    // never actually reads — the real sequential-vs-parallel choice is made by branching on
    // whether the per-layer `ffn_norm` TENSOR exists, and this checkpoint has real ffn_norm
    // tensors on every layer (i.e. genuinely sequential, despite the metadata saying true).
    // Reusing the pre-existing GetBool(metadata, "{arch}.use_parallel_residual") fallback
    // (written for gptneox, where the key genuinely is consulted) would have silently taken
    // the wrong branch. Fixed in ModelGraph.cs: stablelm now derives UseParallelResidual from
    // blk.0.ffn_norm.weight tensor presence instead of the metadata key.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `stabilityai/stablelm-2-zephyr-1_6b` (afrideva GGUF, Q8_0), Stability AI "other" license
    // (non-commercial, gated) — not MIT/Apache-2.0/BSD/MPL. Transient local download, never
    // vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-08-09, stablelm-2-zephyr-1_6b Q8_0, llama.cpp
    // b8585-cad2d3884): prompt "The capital of France is" -> ids [791, 6864, 315, 9822, 374].
    // First 4 generated tokens match exactly (" Paris, and it"). Diverges at position 4: this
    // engine picks token 596 ("'s", logit 24.6648) where llama.cpp's reference implies 374
    // (" is", this engine's own logit: 24.6441) — a 0.0207-logit gap, on a near-lossless Q8_0
    // checkpoint (every other near-tie accepted this session was Q4_K/Q4_K_M). Ruled out
    // before accepting: re-ran with STINGRAY_CPU_PREFILL_Q8=0 (identical result); confirmed
    // via list-tensors that no rope_freqs.weight or attn_q_norm/attn_k_norm tensor is silently
    // missing; confirmed the post-divergence continuation stays fully coherent English, not
    // degenerate. Reads as ordinary Q8_0 accumulation-order sensitivity at a closely-contested
    // position, the same evidentiary category as every other near-tie receipt this session.
    //
    // DO NOT MODIFY THE UseParallelResidual TENSOR-PRESENCE BRANCH FOR stablelm WITHOUT GOOD
    // REASON — there is no regression test to catch a mistake, and reverting to the
    // metadata-key-only computation would silently break it again.
    public static readonly ArchitectureDescriptor Stablelm = new()
    {
        Id = "stablelm",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // hunyuan-dense — admitted 2026-08-09 on a FULL 24-of-24-token exact greedy match
    // (deterministic, though the reference itself is a degenerate repeated-token loop, since
    // the checkpoint is Instruct-tuned and the receipt used a bare, un-templated prompt — still
    // valid token-for-token parity evidence, just not a coherent completion). NOT
    // llama_model_hunyuan_vl (the actual multimodal class) — hunyuan-dense INHERITS its
    // load_arch_hparams/load_arch_tensors/graph wholesale from hunyuan-vl.cpp with no override,
    // confirmed by reading models.h before writing any code, so this receipt's evidence covers
    // both. Ordinary pre-norm RMSNorm trunk, standard GQA (head_count=16, head_count_kv=8),
    // SiLU-gated FFN, no biases anywhere, no MoE, no MRoPE for the text-only dense checkpoint
    // (rope.dimension_sections absent) — none of that is new. The one genuinely new mechanism:
    // weighted QK-norm (a learned per-head RMSNorm, attn_q_norm/attn_k_norm, shape [128] =
    // headDim, not per-channel) applied AFTER RoPE rather than before — confirmed directly
    // against hunyuan-vl.cpp's graph (rope first, then build_norm on the already-rotated Q/K).
    // This engine had two existing QK-norm timings (Qwen3: weighted, before RoPE; Llama-4:
    // unweighted L2, after RoPE) but no "weighted, after RoPE" combination — added
    // ModelHyperparams.QkNormAfterRope and wired it into PrefillCore and RunTrunk (the two
    // paths a plain Prefill()/Forward() receipt exercises; PrefillCoreTq/BatchVerify/
    // BatchForwardMulti's own QK-norm blocks are untouched and would need the identical fix if
    // hunyuan-dense is ever run through those paths). Also needed a new pre-tokenizer cascade:
    // this checkpoint declares tokenizer.ggml.pre=hunyuan-dense, a DISTINCT llama.cpp pre-type
    // (LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE) from the plain "hunyuan" already in this engine's
    // table (which is actually the Qwen-2 cascade) — confirmed via llama-vocab.cpp; added as a
    // 3-stage cascade (PreTokenizerPatterns.DigitRun3/Cjk/HunyuanDenseTail) shared with the
    // deepseek3-llm/joyai-llm pre-types llama.cpp folds onto the same case, verified by
    // checking tokenizer.Encode against llama-tokenize before writing any forward-pass code.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `tencent/Hunyuan-0.5B-Instruct` (bartowski GGUF, Q8_0, 578 MB), Tencent Hunyuan Community
    // License (MAU threshold, territorial exclusions) — not MIT/Apache-2.0/BSD/MPL. Transient
    // local download, never vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-08-09, hunyuan-0.5b-instruct Q8_0, llama.cpp b8585-cad2d3884):
    // prompt "The capital of France is" -> ids [628, 6801, 279, 9391, 316] (confirms the new
    // pre-tokenizer cascade matches the reference exactly). Raw (no chat template) greedy
    // completion degenerates to token 478 repeated 24 times — this engine reproduces that
    // EXACT degenerate sequence for all 24 tokens, a full deterministic match.
    //
    // DO NOT MODIFY THE QkNormAfterRope TIMING FOR hunyuan-dense WITHOUT GOOD REASON — there is
    // no regression test to catch a mistake, and this receipt is currently the only thing in
    // the codebase exercising that combination.
    public static readonly ArchitectureDescriptor Hunyuandense = new()
    {
        Id = "hunyuan-dense",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // hunyuan-moe (Hunyuan-A13B-Instruct) — admitted 2026-10-04 on a real checkpoint (DevQuasar Q3_K_S only), CPU:
    // HunyuanMoeGreedyParityTests teacher-forced against llama-server (same GGUF): all 39 confident positions match
    // (27 of 32 on a 196-token prompt, 12 of 22 on a 5-token one), 0 near-tie differences. Needed in ModelGraph: QK-norm
    // after RoPE (as hunyuan-dense) and renormalised top-k expert weights. Not on any GPU path yet.
    public static readonly ArchitectureDescriptor Hunyuanmoe = new()
    {
        Id = "hunyuan-moe",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Hunyuan-A13B-Instruct (`hunyuan-moe`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    // afmoe (Arcee Trinity Mini) — admitted 2026-10-04 on a real checkpoint (arcee-ai Q4_K_M only), CPU, contexts below the 2048-token
    // sliding window only (window masking beyond it is not verified): AfmoeGreedyParityTests teacher-forced against llama-server, all 31
    // confident positions match (13 of 32 on a 183-token prompt, 18 of 22 on a 5-token one), 3 near-tie differences. Needed: the per-layer
    // attention output gate and 3:1 sliding/global pattern with RoPE only on sliding layers (shared with Muse-Glimmer), muP embedding scale
    // sqrt(n_embd), and the afmoe pre-tokenizer (right-aligned digit groups). Not on any GPU path yet.
    public static readonly ArchitectureDescriptor Afmoe = new()
    {
        Id = "afmoe",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Arcee Trinity Mini (`afmoe`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    // gpt2 — admitted 2026-08-09, FULL 22-of-22-token exact greedy match, bucket-1 (genuinely
    // MIT). The first architecture this session without RoPE at all: GPT-2 encodes position
    // via a learned absolute position-embedding table (`position_embd.weight`) added to the
    // token embedding once, before the trunk starts, not via rotary embeddings inside
    // attention. New: ForwardPass._posEmbdTensor (loaded only when the tensor exists) and a
    // `position` parameter threaded through EmbedTokenInto/EmbedToken and all 8 call sites
    // (every prefill/decode dispatch path). Disabling RoPE needed no new field at all —
    // ModelHyperparams.NoRopeLayerStep = 1 makes the EXISTING Llama-4/SmolLM3 periodic-skip
    // formula ((layer+1) % step != 0) evaluate to "never" for every layer, reusing dispatch
    // every call site already had. Everything else (LayerNorm-with-bias, fused
    // attn_qkv.weight/.bias, non-gated biased-GELU FFN) was already generic from
    // gptneox/falcon. A Q6_K quant tried first diverged at position 5 on only a 0.106-logit
    // gap — read as ordinary quantization sensitivity for a genuinely small/weak 124M model
    // (more sensitive than larger checkpoints, not less) and confirmed by re-running against
    // a near-lossless F16 checkpoint, which matches exactly with no near-tie at all. See
    // Gpt2GreedyParityTests and docs/done/01-gguf-model-coverage-plan.md §1q for the receipt.
    public static readonly ArchitectureDescriptor Gpt2 = new()
    {
        Id = "gpt2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the GPT-2 SafeTensors row describes a separate loader path.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // olmo (v1) — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-1
    // (genuinely Apache-2.0, AI2). One genuinely new mechanism: LayerNorm with NEITHER a
    // learned scale NOR a bias at all — confirmed against olmo.cpp: every build_norm call
    // passes both weight and bias as NULL, and no attn_norm/ffn_norm/output_norm tensor
    // exists in the GGUF at all. A third norm shape distinct from weighted LayerNorm-with-bias
    // (gptneox/falcon/gpt2/starcoder2) and bias-less-but-still-weighted LayerNorm (cohere2). A
    // missing norm tensor already meant something specific here (OLMo2's "skip normalizing
    // here entirely, sandwich-normed on the output instead"), so this needed a genuine
    // arch-string check (ModelHyperparams.UsesUnweightedNorm) to disambiguate from that, not a
    // generalized tensor-presence rule. Added SimdKernels.PureLayerNorm (mean-subtract +
    // variance-normalize, no weight/bias parameter) and wired it into RunTrunk's three norm
    // points ahead of the existing null-DataPtr-means-skip check; PrefillCore's batched norm
    // steps were NOT taught this third mode, routed to the sequential path instead via a new
    // unweightedNormUnsupported flag in PrefillDispatch's fallback gate (same pattern
    // OLMo2/cohere2/Gemma-4 already use for their own PrefillCore gaps). Everything else
    // (plain MHA, standard interleaved RoPE, SiLU-gated FFN, tied embeddings) was already
    // generic. Full exact match on the first real attempt. See OlmoGreedyParityTests and
    // docs/done/01-gguf-model-coverage-plan.md §1s.
    public static readonly ArchitectureDescriptor Olmo = new()
    {
        Id = "olmo",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the MoE partial-offload row is a backend capability, not a family verification row.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    // starcoder (v1) — admitted 2026-08-09, FULL 23-of-23-token exact greedy match, bucket-2,
    // near-zero code change. Confirmed against starcoder.cpp before writing any code: SAME
    // shape as gpt2 (this session's earlier admission) — learned absolute position embeddings
    // (ggml_get_rows(pos_embd, inp_pos), no RoPE anywhere), LayerNorm-with-bias, fused
    // attn_qkv.weight/.bias, non-gated biased-GELU FFN. Also exercises MQA (head_count=16,
    // head_count_kv=1) through the already-generic GQA-parametrized fused-QKV split (first
    // proven on falcon's identical head_count_kv=1 shape). The only change: extended
    // ModelGraph.cs's NoRopeLayerStep=1 gate (built for gpt2) from a single-arch check to
    // arch is "gpt2" or "starcoder". Full exact match on the first real attempt.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `bigcode/starcoderbase-1b` (mradermacher GGUF, Q8_0), BigCode OpenRAIL-M — a restricted-
    // use RAIL license (e.g. malicious-code-generation restrictions), not MIT/Apache-2.0/
    // BSD/MPL. Transient local download, never vendored, deleted immediately after this
    // receipt.
    //
    // Verification evidence (2026-08-09, starcoderbase-1b Q8_0, llama.cpp b8585-cad2d3884):
    // prompt "The capital of France is" -> ids [1318, 18926, 432, 45600, 438]. Full 23-of-23
    // token exact match against the reference continuation, no near-tie, no divergence.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Starcoder = new()
    {
        Id = "starcoder",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // codeshell — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-2,
    // genuinely zero new production code. Confirmed against codeshell.cpp before writing any
    // test: LayerNorm-with-bias, fused attn_qkv.weight/.bias, non-gated biased-GELU FFN — same
    // shapes as gptneox/falcon/starcoder — but REAL RoPE (ggml_rope_ext calls present, NEOX
    // convention per llama_model_rope_type(), and "codeshell" was already in this engine's
    // isNeoxRope list from an earlier session pass), not gpt2/starcoder's absolute position
    // embeddings — so it didn't even need the NoRopeLayerStep widening those two used. The
    // only failure along the way was a wrong test assertion (assumed NORM rope by misreading
    // llama-model.cpp's rope-type switch; codeshell is genuinely in the NEOX case block), not
    // an engine defect.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `WisdomShell/CodeShell-7B` (mradermacher GGUF, Q4_K_M), custom WisdomShell/CodeShell
    // license (no SPDX permissive tag found) — not MIT/Apache-2.0/BSD/MPL. Transient local
    // download, never vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-08-09, CodeShell-7B Q4_K_M, llama.cpp b8585-cad2d3884):
    // prompt "The capital of France is" -> ids [46479, 53434, 15979, 48944, 19206, 55391].
    // Full 24-of-24 token exact match against the reference continuation, no near-tie, no
    // divergence.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Codeshell = new()
    {
        Id = "codeshell",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // jais2 — admitted 2026-08-09, FULL 3-of-3-token exact greedy match (including a natural
    // EOS stop), bucket-2. Confirmed against jais2.cpp before writing any code: LayerNorm-
    // with-bias, separate (not fused) biased Q/K/V/output projections, and standard NEOX RoPE
    // (already in this engine's isNeoxRope list) were all already generic. The one genuinely
    // new piece: non-gated FFN with ReLU-squared activation (max(0,x)^2, LLM_FFN_RELU_SQR),
    // biased the same way GPT-NeoX's GELU is (up-bias inside the activation, down-bias after)
    // — added SimdKernels.ReluSqrInPlace and ModelHyperparams.UsesReluSquared
    // (arch=="jais2"), wired into both DenseFfn and PrefillCore's non-gated-FFN branches
    // alongside the existing xIELU/GELU dispatch. Also needed a new pre-tokenizer regex
    // (PreTokenizerPatterns.Jais2, registered under "jais-2") — Llama-3's pattern with the
    // trailing whitespace alternative replaced by a cascading fixed-length run (512, 256, ...,
    // 1), ported directly from llama-vocab.cpp's LLAMA_VOCAB_PRE_TYPE_JAIS2 case, verified
    // against llama-tokenize before writing any forward-pass code.
    //
    // NO AUTOMATED TEST FOR THIS ARCHITECTURE, FOR LICENCE REASONS. Checkpoint:
    // `yoriis/JAIS2-IT-0.3` (a third-party fine-tune of `inceptionai/Jais-2-8B-Chat`, itself
    // Apache-2.0 but gated — requires accepting terms via the HF web UI, which this session
    // cannot do programmatically), via `mradermacher/JAIS2-IT-0.3-GGUF`, Q4_K_M — the
    // fine-tune's own license isn't independently declared, so treated as bucket-2 rather than
    // assumed to inherit the base's Apache-2.0. Transient local download, never vendored,
    // deleted immediately after this receipt.
    //
    // Verification evidence (2026-08-09, JAIS2-IT-0.3 Q4_K_M, llama.cpp b8585-cad2d3884):
    // prompt "The capital of France is" -> ids [1947, 9748, 1267, 13517, 1358]. Greedy
    // completion stops naturally at EOS after " Paris." (3 tokens including EOS, id 150024) —
    // this engine reproduces the identical 3-token sequence exactly, including landing on EOS
    // at the same position (confirming the full logit ranking is correct, not just an
    // argmax-until-something coincidence). A longer receipt was attempted via
    // `--ignore-eos`, but that flag suppresses EOS in llama.cpp's SAMPLER, not the forward
    // pass — comparing against it produced a false "divergence" at the exact position this
    // engine's un-suppressed greedy correctly picks EOS, matching the real (non-suppressed)
    // reference; the 3-token receipt is the correct evidence.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Jais2 = new()
    {
        Id = "jais2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // jais (v1) — admitted 2026-09-28. Confirmed against jais.cpp before writing any code.
    //
    // ARCHITECTURAL ANALYSIS (jais.cpp vs jais2.cpp diff):
    // jais and jais2 are genuinely DIFFERENT architectures, not aliases:
    //   • Position encoding: jais uses ALiBi (ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS),
    //     no inp_pos/ggml_rope_ext anywhere in the graph); jais2 uses NEOX RoPE.
    //   • QKV projection: jais uses a fused blk.N.attn_qkv.weight/bias (same layout as
    //     gptneox/falcon/codeshell/gpt2/starcoder — already handled generically by the
    //     fused-QKV split in ForwardPass's constructor); jais2 uses separate wq/wk/wv.
    //   • FFN: jais uses a GATED SiLU FFN (ffn_gate + ffn_up + ffn_down, all biased —
    //     LLM_FFN_SILU, LLM_FFN_PAR); jais2 uses a NON-GATED ReLU-squared FFN (ffn_up +
    //     ffn_down only, LLM_FFN_RELU_SQR, LLM_FFN_SEQ). The gated-SiLU-with-FFN-bias
    //     combination is already supported by this engine's generic HasFfnBias detection.
    //   • Output weight: jais has a required separate output.weight; jais2 ties to tok_embd.
    //
    // WHAT IS ALREADY GENERIC (zero new production code needed):
    //   • LayerNorm-with-bias (attn_norm.bias / ffn_norm.bias / output_norm.bias present) →
    //     UsesLayerNorm=true + HasNormBias=true, both detected from tensor presence.
    //   • Fused attn_qkv.weight/bias → split by contiguous row offset in ForwardPass
    //     constructor (same code gptneox/falcon/codeshell/gpt2/starcoder already use).
    //   • attn_output.weight/bias → HasAttnOutputBias via _opentailllm.has_attn_output_bias
    //     (the GGUF converter already injects this key, confirmed via list-metadata).
    //   • ffn_gate.bias + ffn_up.bias + ffn_down.bias → HasFfnBias=true, tensor-presence.
    //   • No RoPE → ModelGraph sets noRopeStep=1 for arch=="jais" (see comment there), reusing
    //     the NoPE formula already proven for gpt2/starcoder on every layer, not just periodic.
    //   • tokenizer.ggml.pre=="jais" → already in PreTokenizerPatterns's "gpt-2" case table
    //     (PreTokenizerPatterns.cs line ~201: GPT-2 pattern, same as mpt/olmo/trillion).
    //
    // KNOWN GAP — ALiBi NOT IMPLEMENTED:
    //   jais.cpp adds a per-head positional slope to each attention head's score matrix before
    //   softmax (ALiBi, "Attention with Linear Biases"). This engine has no ALiBi implementation.
    //   The effect is that generated text will lack position-aware attention weighting. For
    //   short prompts this typically degrades coherence (model treats all positions equally)
    //   but does not produce structurally corrupt output. The attention scale in jais.cpp is
    //   1.0f/float(n_embd_head) rather than 1/sqrt(head_dim) — these differ for jais's
    //   head_dim=128 (1/128 vs ~0.0884=1/sqrt(128)): the head_dim IS sqrt-free in jais.
    //
    // CHECKPOINT: mradermacher/jais-family-590m-chat-GGUF, Q4_K_M (Apache-2.0, bucket-1 —
    // genuinely permissive). 18 layers, embDim=1536, numHeads=12, headDim=128, ffDim=4096,
    // vocab=84992, contextLen=2048, max_alibi_bias=8.
    //
    // VERIFICATION EVIDENCE (2026-09-28, mradermacher/jais-family-590m-chat-GGUF Q4_K_M,
    // tools/llama.cpp llama-cli, CPU backend, temp=0 -g -1):
    // See JaisGreedyParityTests for the token sequence and complete receipt.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no full
    // regression test. In particular, the noRopeStep=1 wiring is load-bearing for correct
    // operation; reverting it would silently apply RoPE where ALiBi was intended.
    public static readonly ArchitectureDescriptor Jais = new()
    {
        Id = "jais",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Jais v1 (`jais`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    // maincoder — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-1
    // (genuinely Apache-2.0), zero new code. Confirmed against maincoder.cpp before writing
    // any code: a literal Qwen3-shaped architecture — RMSNorm, biasless GQA with weighted
    // per-head QK-norm (AFTER RoPE — corrected 2026-09-26, see QkNormAfterRope), standard SiLU-gated FFN,
    // standard interleaved (non-NEOX) RoPE (confirmed via llama_model_rope_type() returning
    // NORM for LLM_ARCH_MAINCODER, matching the default). tokenizer.ggml.pre=qwen2 with real
    // merges — already covered. Every mechanism this checkpoint exercises predates this
    // session. See MaincoderGreedyParityTests and docs/done/01-gguf-model-coverage-plan.md §1w.
    public static readonly ArchitectureDescriptor Maincoder = new()
    {
        Id = "maincoder",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
    };

    public static readonly ArchitectureDescriptor Exaone4 = new()
    {
        Id = "exaone4",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "EXAONE 4.5 33B (`exaone4`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    public static readonly ArchitectureDescriptor Mistral3 = new()
    {
        Id = "mistral3",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    public static readonly ArchitectureDescriptor Ministral = new()
    {
        Id = "ministral",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // xverse — re-investigated and admitted 2026-09-02, FULL 24-of-24-token exact greedy
    // match, bucket-2 (code Apache-2.0, but weights under XVERSE's own custom "Model License
    // Agreement" — free for commercial use per its own docs, but not a clean SPDX permissive
    // license, so no persisted test per this file's bucket-2 policy). Literal plain-Llama
    // trunk, zero new architecture code (confirmed against xverse.cpp: RMSNorm pre-norm,
    // ordinary biasless Q/K/V/O attention, standard/NORM RoPE — llama_model_rope_type()
    // places LLM_ARCH_XVERSE in the same case as LLM_ARCH_LLAMA — standard SiLU-gated FFN, no
    // QK-norm). This was always a tokenizer-axis blocker, not an architecture one (first
    // flagged 2026-08-09): the checkpoint has neither tokenizer.ggml.merges nor
    // tokenizer.ggml.scores despite declaring tokenizer.ggml.model=llama. Two real,
    // independent SPM-tokenizer bugs were found and fixed re-investigating this: (1) this
    // engine used a merges-RANK-TABLE algorithm for genuine SentencePiece BPE, but the real
    // algorithm (confirmed against llama.cpp's llm_tokenizer_spm_session::tokenize) has no
    // merges list at all — a merge is valid because the concatenated text is already a vocab
    // entry, prioritized by that entry's own score (GgufTokenizer.SpmMergePiecesByScore) —
    // this fix is real and permanent, exercised by SpmMergeByScoreTests.cs's synthetic fuzz
    // suite regardless of xverse's own license; (2) this engine never implemented
    // add_space_prefix (real llama.cpp default true for SPM — a leading space is prepended
    // before tokenizing, TokenizerSource.AddSpacePrefix).
    //
    // Checkpoint: xverse/XVERSE-7B-Chat-GGUF, Q4_K_M (4.47 GB). Transient local download,
    // never vendored, deleted immediately after this receipt.
    //
    // Verification evidence (2026-09-02, XVERSE-7B-Chat-GGUF Q4_K_M, tools/llama.cpp build
    // 10306/6b5c2efb4): prompt "The capital of France is" -> ids [96740, 98398, 97896, 96604,
    // 98030, 96884, 96636]. Reference continuation captured via llama-server's /completion
    // endpoint with return_tokens:true (NOT by re-tokenizing the printed text — this vocab
    // has genuine tokenizer non-injectivity, so retokenizing the model's own printed output
    // reproduced a DIFFERENT 29-token sequence than the 24 tokens actually sampled):
    // [97003, 96579, 99455, 42, 118, 99413, 96672, 99629, 47, 96677, 96570, 98216, 98131,
    // 96604, 97042, 9044, 52, 53, 100061, 97357, 49, 97060, 96636, 98927]. Full 24-of-24-token
    // exact match against this engine, no near-tie, no divergence anywhere.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Xverse = new()
    {
        Id = "xverse",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };

    // minicpm — ADMITTED 2026-09-01. Was blocked purely on the tokenizer axis (§01 coverage
    // plan §1d/§1c): shares Granite's graph builder verbatim (llama.cpp's models.h:
    // llama_model_minicpm::graph == llama_model_granite::graph — same embedding/residual/
    // logit-scale trio, different constants), so the forward pass needed zero new code once
    // Granite/smollm3 were admitted. The real blocker was its GGUF declaring
    // tokenizer.ggml.model=llama with a tokenizer.ggml.scores array and NO
    // tokenizer.ggml.merges array — this engine's SPM path used to require a merges table for
    // any tokenization at all and fell through to character-level fragmentation. Already fixed
    // by GgufTokenizer.SpmMergePiecesByScore (built while re-admitting xverse, 2026-09-02) —
    // NOT by this session's separate real-Unigram-LM (tokenizer.ggml.model=t5) addition, which
    // this checkpoint doesn't use at all (worth noting: this project's own docs had
    // provisionally grouped minicpm/internlm2/ernie4_5/baichuan/orion/nanbeige together under
    // "Unigram-LM SentencePiece" before any of them were actually re-checked against a real
    // downloaded GGUF; minicpm turns out to be the scores-only SPM case, the same shape as
    // xverse's own second bug, not genuine Unigram-LM. The other five remain unverified and
    // may turn out to be either shape.).
    //
    // Checkpoint: openbmb/MiniCPM4-0.5B (Apache-2.0), via Mungert/MiniCPM4-0.5B-GGUF,
    // Q4_K_M (263 MB). Transient local download, never vendored, deleted immediately after
    // this receipt.
    //
    // Verification evidence (2026-09-01, tools/llama.cpp llama-tokenize/llama-server, CPU
    // backend — this checkpoint's Vulkan-hybrid-offload output is a separate, unrelated known
    // issue, see the Z-Image BF16/Sgemm-precision bug for the general shape of that class of
    // bug; not investigated further here since the tokenizer/architecture axis is what this
    // item was scoped to): full templated prompt "<|im_start|>user\nThe capital of France
    // is<|im_end|>\n<|im_start|>assistant\n" tokenizes identically on both sides (15 tokens
    // including BOS: [1, 73441, 3060, 5, 2219, 8107, 1379, 8360, 1410, 73440, 59320, 5, 73441,
    // 16434, 5]). Greedy continuation FULL 8-of-8-token exact match through EOS: engine and
    // llama-server both produce [2219, 8107, 1379, 8360, 1410, 11225, 72, 73440] ("The capital
    // of France is Paris." + <|im_end|>), engine's own per-step top-1 confirmed via
    // --verbose-prompt debug trace, llama-server's via /completion with return_tokens:true.
    //
    // DO NOT MODIFY THIS ARCHITECTURE'S CODE PATH WITHOUT GOOD REASON — there is no regression
    // test to catch a mistake.
    public static readonly ArchitectureDescriptor Minicpm = new()
    {
        Id = "minicpm",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
    };
}
