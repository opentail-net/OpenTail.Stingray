namespace OpenTail.Stingray.Engine;

internal static class QwenArchitectures
{
    // Decoder-only transformer profiles exercised by OpenTail's forward passes.
    // llama, llama4 -> Architectures/LlamaArchitecture.cs, Llama4Architecture.cs (descriptors win over this list).
    public static readonly ArchitectureDescriptor Qwen = new()
    {
        Id = "qwen",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the SafeTensors row is a separate loader path.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'qwen'.
    public static readonly ArchitectureDescriptor Qwen2 = new()
    {
        Id = "qwen2",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the SafeTensors row is a separate loader path.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'qwen'.
    public static readonly ArchitectureDescriptor Qwen2moe = new()
    {
        Id = "qwen2moe",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Qwen2-MoE (`qwen2moe`",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'qwen'.
    public static readonly ArchitectureDescriptor Qwen3 = new()
    {
        Id = "qwen3",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; the SafeTensors row is a separate loader path.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'qwen'.
    public static readonly ArchitectureDescriptor Qwen3moe = new()
    {
        Id = "qwen3moe",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md does not have a separate Qwen3-MoE row.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // qwen2vl -- admitted 2026-08-18 (docs/089) SCOPED TO TEXT-ONLY use (no image/video
    // tokens ever fed through this engine): confirmed against the real vendored
    // `transformers.models.qwen2_vl.modeling_qwen2_vl` source that pure-text M-RoPE reduces
    // exactly to standard 1D NEOX rope (see ModelGraph.cs's `isNeoxRope` dispatch comment for
    // the full derivation). Used by Qwen Image's real LLM text-conditioning extraction
    // (Qwen2.5-VL-7B-Instruct's text backbone), not for real vision/multimodal generation --
    // that would need the real multi-section M-RoPE this engine does not implement.
    public static readonly ArchitectureDescriptor Qwen2vl = new()
    {
        Id = "qwen2vl",
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Admitted only for text-only decoding; the Qwen Image row concerns a separate diffusion text-conditioning path.",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // qwen35 — hybrid Gated-DeltaNet MoE + MTP path (docs/2-coverage/02-qwen35moe-plan.md). Ornith-1.0
    // 9B (dense-ish, no MoE) was the original end-to-end validation. Extended 2026-08-28 with
    // a FULL 24-of-24-token exact greedy match on Qwen3.8-27B UD-Q3_K_XL (Unsloth Dynamic
    // quant, Apache-2.0, obtained via local Ollama cache) against llama.cpp b10532-70aff2525:
    // prompt "The capital of France is" (raw completion, no chat template, temp 0,
    // repeat_penalty 1.0) -> "Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy
    // is Rome.\nThe capital of Spain is", byte-identical on both sides including the 5-token
    // prompt tokenization. This checkpoint also exercises IQ2_S/IQ2_XS/IQ3_XXS/IQ4_XS
    // per-tensor (see IsSupportedWeightDType's IQ2_XS/IQ2_S entries and
    // docs/done/01-gguf-model-coverage-plan.md §2), so the receipt covers both the forward-pass
    // architecture and the newly-admitted dequant formats in one shot. 64 layers, 5120d,
    // headDim=256, 248320 vocab, full_attention_interval=4, block 64 is an unused MTP
    // next-token-prediction head (correctly ignored by both engines in non-speculative mode).
    public static readonly ArchitectureDescriptor Qwen35 = new()
    {
        Id = "qwen35",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Qwen3.5 / 3.6 / 3.8 hybrid Gated DeltaNet + MoE",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // Shares the evidence comment and receipt for 'qwen35'.
    public static readonly ArchitectureDescriptor Qwen35moe = new()
    {
        Id = "qwen35moe",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Qwen3.5 / 3.6 / 3.8 hybrid Gated DeltaNet + MoE",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    // qwen3vl -- Qwen3-VL text decoder (llama.cpp src/models/qwen3vl.cpp): the qwen3 block with IMROPE (interleaved
    // M-RoPE sections [24,20,20,0]; for text the 4th component is 0, so pairs 61-62 never rotate) and deepstack
    // (vision slices added after layers 0..n_deepstack-1). ADMITTED 2026-09-27. Evidence: Qwen3VL-2B-Instruct Q8_0
    // wikitext second-half PPL at -c 2048 9.8356 vs llama-perplexity --chunks 1 9.8513 (it was 2654 with the
    // interleaved-pair rotation the arch fell back to before). Images (CPU): LlamaMtmdVisionParityTests.Qwen3Vl_Rainbow448
    // pins the encoder to llama-mtmd-debug, and on test-1.jpeg the answer matches llama-mtmd-cli but for one
    // capitalisation token.
    public static readonly ArchitectureDescriptor Qwen3vl = new()
    {
        Id = "qwen3vl",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "Vision: Qwen3-VL",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
