namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
    public static readonly ArchitectureDescriptor Mimo = new()
    {
        Id = "mimo",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'mimo'.
    public static readonly ArchitectureDescriptor Mimo2 = new()
    {
        Id = "mimo2",
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
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
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
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
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
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
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
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
        UsesNeoxRope = true,
        ApplyModelSemantics = ctx => FamilyModelSemantics.Granite(ctx, miniCpm: true),
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
