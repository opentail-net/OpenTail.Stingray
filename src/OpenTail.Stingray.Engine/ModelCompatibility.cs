
namespace OpenTail.Stingray.Engine;

/// <summary>
/// Declares the GGUF model profiles that the text-generation forward passes implement.
/// GGUF is a container, not a promise of interchangeable model mathematics: accepting an
/// unfamiliar architecture merely because it happens to expose familiar tensor names can
/// produce plausible but incorrect tokens. Keep this gate deliberately conservative.
/// </summary>
public static class ModelCompatibility
{
    private static readonly HashSet<string> s_textGenerationArchitectures = new(StringComparer.OrdinalIgnoreCase)
    {
        // Decoder-only transformer profiles exercised by OpenTail's forward passes.
        "llama", "llama4",
        "qwen", "qwen2", "qwen2moe", "qwen3", "qwen3moe",
        // qwen35 — hybrid Gated-DeltaNet MoE + MTP path (docs/02-qwen35moe-plan.md). Ornith-1.0
        // 9B (dense-ish, no MoE) was the original end-to-end validation. Extended 2026-08-28 with
        // a FULL 24-of-24-token exact greedy match on Qwen3.8-27B UD-Q3_K_XL (Unsloth Dynamic
        // quant, Apache-2.0, obtained via local Ollama cache) against llama.cpp b10532-70aff2525:
        // prompt "The capital of France is" (raw completion, no chat template, temp 0,
        // repeat_penalty 1.0) -> "Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy
        // is Rome.\nThe capital of Spain is", byte-identical on both sides including the 5-token
        // prompt tokenization. This checkpoint also exercises IQ2_S/IQ2_XS/IQ3_XXS/IQ4_XS
        // per-tensor (see IsSupportedWeightDType's IQ2_XS/IQ2_S entries and
        // docs/01-gguf-model-coverage-plan.md §2), so the receipt covers both the forward-pass
        // architecture and the newly-admitted dequant formats in one shot. 64 layers, 5120d,
        // headDim=256, 248320 vocab, full_attention_interval=4, block 64 is an unused MTP
        // next-token-prediction head (correctly ignored by both engines in non-speculative mode).
        "qwen35", "qwen35moe",
        "mimo", "mimo2",
        "gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
        "phi2", "phi3", "phimoe",
        // olmoe — admitted 2026-08-08 on perplexity parity, NOT on token-for-token greedy parity,
        // which it does not achieve. On wikitext at a matched 2048-token context llama.cpp b8585
        // scores 7.4868 and this engine scores 7.3889 (1.3%). The greedy divergence is at a flat
        // position where the top five candidates span 1.55 logits, i.e. where a differently
        // quantised matmul reorders candidates. Evidence and the argument for accepting it:
        // docs/01-gguf-model-coverage-plan.md §1b. Note `olmo2` is deliberately NOT here — it
        // shares neither a fixture nor a receipt.
        "olmoe",
        // granite — admitted 2026-08-08 on FULL 24-token exact greedy match against llama.cpp
        // (stronger than the olmoe receipt above, which only reaches a 2-token prefix). Needs a
        // "scale trio" + attention-scale override beyond the plain llama trunk, read from GGUF
        // metadata (ModelHyperparams.ResidualScale/AttentionScaleOverride/LogitScale, generalized
        // EmbeddingScale) — see GraniteGreedyParityTests and docs/01-gguf-model-coverage-plan.md
        // §1d for the receipt and for what is NOT yet wired (TurboQuant prefill, continuous-batching
        // admission, CUDA/Vulkan). MiniCPM (not MiniCPM3 — that's MLA, a different architecture)
        // shares this exact graph in llama.cpp and reuses the same implementation, unvalidated here
        // pending a permissively-licensed checkpoint on a llama.cpp build that can serve as an oracle.
        "granite",
        // smollm3 — one twist over the plain llama trunk: NoPE every 4th layer, gated the same way
        // as llama4's noRopeStep. See SmolLm3GreedyParityTests for the full 24-token greedy receipt.
        "smollm3",
        // apertus — admitted 2026-08-08 on an 11-token EXACT prefix match (one full sentence)
        // against llama.cpp, diverging afterward into a different but still coherent, on-topic
        // completion (not degenerate output). The first "new-kernel" architecture admitted this
        // session: no ffn_gate tensor at all (plain up -> xIELU -> down, ModelHyperparams.Xielu*,
        // SimdKernels.XieluInPlace), detected from tensor inventory rather than architecture
        // string. See ApertusGreedyParityTests and docs/01-gguf-model-coverage-plan.md §1f for the
        // receipt, including a real defect found and fixed in the xIELU parameter transform
        // (GGUF stores pre-softplus values; llama.cpp's ggml_xielu() wrapper — not the compute
        // kernel — applies softplus before use, easy to miss by reading only the kernel).
        "apertus",
        // gptneox (Pythia) — LayerNorm (mean/variance + learned bias, not RMSNorm), a biased
        // non-gated GELU FFN, a fused blk.*.attn_qkv.weight/bias tensor pair (split by
        // contiguous row offset in ForwardPass's constructor — Q rows, then K rows, then V
        // rows; confirmed against examples/llama.cpp/llama.cpp/conversion/gptneox.py and
        // src/models/gptneox.cpp, NOT the interleaved per-head layout an earlier draft
        // assumed), and the metadata-driven parallel-residual graph (x + attn(ln1(x)) +
        // ffn(ln2(x)), both norms reading the SAME incoming residual — ModelHyperparams.
        // HasNormBias/HasFfnBias/UseParallelResidual). See GptNeoxGreedyParityTests and
        // docs/01-gguf-model-coverage-plan.md for the receipt. TurboQuant prefill, continuous-
        // batching admission, and CUDA/Vulkan are not wired to this profile.
        "gptneox",
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
        // docs/01-gguf-model-coverage-plan.md for the receipt.
        "falcon",
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
        // See Olmo2GreedyParityTests and docs/01-gguf-model-coverage-plan.md for the receipt.
        "olmo2",
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
        // MIT/Apache-2.0/BSD/MPL. Per the license policy in docs/01-gguf-model-coverage-plan.md
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
        "exaone",

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
        "orion",

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
        "ernie4_5",

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
        "internlm2",
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
        // policy in docs/01-gguf-model-coverage-plan.md, no permanent test persists.
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
        "starcoder2",
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
        "cohere2",
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
        // llama.cpp/src/models/glm4.cpp and docs/01-gguf-model-coverage-plan.md for the receipt,
        // including two real defects found and fixed while building it: a fused-tensor-slice
        // prefault-sizing bug that actually crashed with AccessViolationException (and was fixed
        // retroactively for the pre-existing GPT-NeoX/Falcon fused-QKV split too, since it's the
        // identical latent defect there, just not yet triggered), and a wholly missing partial-RoPE
        // implementation for the "normal"/interleaved (non-NEOX) rotation convention
        // (SimdKernels.ApplyRoPECachedPartial, new).
        "glm4",
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
        "stablelm",
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
        "hunyuan-dense",
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
        // Gpt2GreedyParityTests and docs/01-gguf-model-coverage-plan.md §1q for the receipt.
        "gpt2",
        // granitemoe — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-1
        // (genuinely Apache-2.0), essentially a free admission. llama_model_granite_moe::graph is
        // a type alias for llama_model_granite::graph (confirmed in models.h before writing any
        // code) — the SAME graph as dense Granite (already admitted), which already branches on
        // n_expert==0 internally. This engine's generic MoE dispatch and the Granite-family scale
        // block (ResidualScale/EmbeddingScale/LogitScale, isGraniteFamily in ModelGraph.cs) already
        // explicitly checked arch=="granitemoe" from when the dense receipt was built. Standard
        // softmax gating, no shared expert, standard GQA — every mechanism this checkpoint
        // exercises was already correct on the first real attempt (the only failure along the way
        // was a wrong test assertion, not an engine defect — LogitScale already carries the
        // reciprocal of the raw metadata value, documented but momentarily forgotten while writing
        // the test). See GraniteMoeGreedyParityTests and docs/01-gguf-model-coverage-plan.md §1r.
        "granitemoe",
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
        // docs/01-gguf-model-coverage-plan.md §1s.
        "olmo",
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
        "starcoder",
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
        "codeshell",
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
        "jais2",
        // maincoder — admitted 2026-08-09, FULL 24-of-24-token exact greedy match, bucket-1
        // (genuinely Apache-2.0), zero new code. Confirmed against maincoder.cpp before writing
        // any code: a literal Qwen3-shaped architecture — RMSNorm, biasless GQA with weighted
        // per-head QK-norm before RoPE (this engine's default timing), standard SiLU-gated FFN,
        // standard interleaved (non-NEOX) RoPE (confirmed via llama_model_rope_type() returning
        // NORM for LLM_ARCH_MAINCODER, matching the default). tokenizer.ggml.pre=qwen2 with real
        // merges — already covered. Every mechanism this checkpoint exercises predates this
        // session. See MaincoderGreedyParityTests and docs/01-gguf-model-coverage-plan.md §1w.
        "maincoder",
        "exaone4",
        "mistral3",
        "ministral",
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
        "xverse",

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
        "minicpm",
    };
    // deepseek2 — NOT admitted, closed for now. A CPU-only MLA implementation exists in
    // ForwardPass.cs (compressed-latent K/V, YaRN RoPE, kq_scale/mscale correction, per-layer
    // dense/MoE dispatch) and loads/generates with zero crashes against real GGUFs
    // (DeepSeek-V2-Lite-Chat, Q2_K and native Q8_0), but produces numerically wrong output
    // ("The capital of France is" does not continue with "Paris", while the same GGUFs via a
    // reference llama.cpp build do). Four investigation rounds found and fixed several real,
    // independently-useful bugs along the way (YaRN/kq_scale formula, expert_weights_norm/scale
    // GGUF handling, RMSNorm/softmax/SiLU ggml-fidelity fixes, and a genuine Q8_0
    // activation-quantization gap in MatVecQ8_0) but did not find a discrete root cause. The
    // decisive, twice-replicated finding (Q2_K and native Q8_0, both measured through
    // numerically-sound kernels) is that this checkpoint's MoE router has chronically near-tied
    // top-6-of-64 routing decisions (median boundary margin ~0.002), a property of its trained
    // weights that any tiny remaining numerical difference from ggml can flip, compounding
    // through 27 layers into a fully sign-flipped residual stream by ~layer 22. Decision
    // (2026-08-28): stop chasing this checkpoint further. Full writeup, including what was
    // ruled out and what's left untried (a larger DeepSeek-V2/V3 checkpoint, full graph-wide
    // SIMD reduction-order matching), is in
    // docs/done/032-deepseek2-mla-yarn-moe-routing-investigation.md. Do not re-admit without
    // either a passing greedy-parity receipt or an explicit decision to ship known-wrong behind
    // --allow-unverified-arch. GPU backends have no MLA support at all — CPU-only was the
    // deliberately agreed scope.
    //
    // deepseek4 — NOT admitted. DeepSeek4ForwardPass (DeepSeek4ForwardPass.cs) is a structurally
    // complete ALPHA/UNTESTED IForwardPass covering all three of V4's attention variants (raw,
    // HCA, CSA), hyper-connections, MoE (incl. hash routing + sqrt-softplus gating), and tensor
    // loading (DeepSeek4TensorSet.cs) — but it has NEVER been run: no DeepSeek-V4 GGUF has been
    // loaded, no output compared against any reference. Multiple specific pieces are known-
    // unverified or known-wrong (MQA K==V, output-side rope_ext_back, CSA's overlap-gather
    // hypothesis, Hadamard rotation entirely unimplemented) — see this file's own header and
    // docs/058-deepseek-full-lineage-implementation-plan.md Phase 0 for the full, current list.
    // Only synthetic shape/invariant/boundary unit tests exist (DeepSeek4AlphaTests.cs) — zero
    // real-weight verification. Do not admit under any circumstance until a real checkpoint
    // produces a passing greedy-parity or perplexity receipt.
    //
    // deepseek32 — NOT admitted. DeepSeek32ForwardPass (DeepSeek32ForwardPass.cs) is a
    // structurally complete ALPHA/UNTESTED IForwardPass: classic MLA attention with absorption,
    // DSA/lightning-indexer sparse masking over a real raw indexer K cache, dense/MoE FFN
    // dispatch. NEVER RUN — no DeepSeek-V3.2 GGUF available. Four specific, deliberate
    // simplifications documented in the file's header: (1) YaRN RoPE scaling IS implemented
    // (ported from the deepseek2 investigation's already-verified formula chain,
    // docs/done/032-...md), but not independently re-verified in this new context, and the
    // "YaRN active" detection is a RopeYarnFactor>1 proxy rather than reading
    // rope.scaling.type directly; (2) Hadamard rotation on the indexer's Q/K not implemented
    // (same gap/reason as deepseek4); (3) MTP not implemented; (4) the MLA absorption math was
    // re-derived from reading deepseek32.cpp, not reused from this codebase's dead
    // Engine.MlaAttention/DeepSeekMoeGraph classes, and is unverified. Phase
    // 1 of docs/058-...md. IMPORTANT: deepseek32 is NOT a subset of deepseek4 despite the version
    // ordering — classic MLA is a different attention design from V4's CSA/HCA compressed-state
    // mechanism; none of deepseek4's CSA/HCA/hyper-connection code is reusable here. Shares the
    // lightning-indexer/DSA concept with deepseek4, but deepseek32's indexer is simpler (a real
    // raw indexer_attn_k tensor, no compression needed).
    //
    // minicpm — NOT admitted. The forward-pass scale trio (reusing Granite's graph, see
    // GraniteGreedyParityTests) is implemented and presumed correct, but MiniCPM4-0.5B — the only
    // Apache-2.0 checkpoint tried (2026-08-08) — declares tokenizer.ggml.model=llama with a
    // `scores` array and NO `merges` array: Unigram-LM SentencePiece (Viterbi segmentation), not
    // the BPE-order SPM (explicit merges list) that Llama/Gemma use and this engine implements.
    // Measured: our tokenizer produces unrelated single-token-per-fragment ids for a 5-token
    // reference prompt. A different, unimplemented tokenization algorithm, not a scale-trio bug —
    // see docs/01-gguf-model-coverage-plan.md §1d.

    /// <summary>Whether the architecture has an implemented text-generation forward profile.</summary>
    public static bool IsTextGenerationArchitectureSupported(string architecture) =>
        s_textGenerationArchitectures.Contains(architecture);

    /// <summary>
    /// Matrix weight formats implemented by the portable CPU path. CUDA/Vulkan routes share
    /// this conservative baseline at model-load time, so a model cannot defer a missing CPU
    /// fallback/dequantizer error until its first request.
    /// </summary>
    public static bool IsSupportedWeightDType(DType dtype) => dtype is
        DType.Float32 or DType.Float16 or DType.BFloat16 or
        DType.Q4_0 or DType.Q4_1 or DType.Q5_0 or DType.Q5_1 or DType.Q8_0 or DType.Q8_1 or
        DType.Q2_K or DType.Q3_K or DType.Q4_K or DType.Q5_K or DType.Q6_K or
        DType.IQ4_NL or DType.IQ2_S or DType.IQ2_XS or DType.IQ2_XXS or DType.IQ3_XXS or DType.IQ3_S or DType.IQ4_XS or
        DType.IQ1_S or DType.IQ1_M or
        DType.MXFP4 or DType.NVFP4 or DType.Q1_0 or DType.Q2_0 or
        DType.TQ2_0 or DType.TQ1_0;

    /// <summary>
    /// Validates that a GGUF can be served by OpenTail's text-generation engine. Call this
    /// after reading metadata and before selecting a backend or constructing a forward pass.
    /// </summary>
    public static void ValidateForTextGeneration(GgufModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        string architecture = model.Metadata.TryGetValue("general.architecture", out var value)
            ? Convert.ToString(value) ?? ""
            : "llama";

        if (!IsTextGenerationArchitectureSupported(architecture)
            && Environment.GetEnvironmentVariable("STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH") != "1")
        {
            throw new NotSupportedException(
                $"GGUF architecture '{architecture}' is not supported for text generation by OpenTail.Stingray. " +
                "The model was rejected before inference because GGUF tensor naming alone does not establish " +
                "compatible attention, RoPE, normalization, and FFN semantics. Supported profiles: " +
                $"{string.Join(", ", s_textGenerationArchitectures.Order())}.");
        }

        var unsupported = model.Tensors
            .Where(t => !IsSupportedWeightDType(t.DType))
            .Select(t => $"{t.Name} ({t.DType})")
            .Take(4)
            .ToArray();
        if (unsupported.Length > 0)
        {
            throw new NotSupportedException(
                "This GGUF uses tensor storage formats that OpenTail.Stingray cannot execute on its portable " +
                "text-generation path: " + string.Join(", ", unsupported) + ". " +
                "Use a model quantized as Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, Q2_K–Q6_K, IQ4_NL, " +
                "IQ2_S, IQ2_XS, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS, IQ1_S, IQ1_M, MXFP4, NVFP4, Q1_0, Q2_0, TQ2_0, TQ1_0, F16, BF16, or F32.");
        }
    }
}
