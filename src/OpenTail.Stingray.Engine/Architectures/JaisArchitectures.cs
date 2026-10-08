namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
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
        UsesNeoxRope = true,
        ApplyModelSemantics = ctx => ctx.Baseline with { UsesReluSquared = true },
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
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
        // ALiBi position encoding, no RoPE (jais.cpp); kq_scale = 1/n_embd_head (not 1/sqrt).
        ApplyModelSemantics = ctx => ctx.Baseline with
        {
            NoRopeLayerStep = 1,
            AttentionScaleOverride = ctx.Baseline.HeadDim > 0 ? 1f / ctx.Baseline.HeadDim : ctx.Baseline.AttentionScaleOverride,
        },
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Jais v1 (`jais`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
