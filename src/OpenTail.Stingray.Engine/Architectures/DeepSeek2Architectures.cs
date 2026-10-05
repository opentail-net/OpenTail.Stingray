namespace OpenTail.Stingray.Engine;

internal static class DeepSeek2Architectures
{
    // deepseek2 — admitted 2026-09-26 (CPU only; GPU backends have no MLA). docs/done/032's
    // "trained router margins" conclusion was wrong: two plain bugs made it emit garbage from
    // token 0. (1) The shared expert is n_shared x expert dim wide (2 x 1408 = 2816) but ran at
    // 1408 (ModelHyperparams.SharedExpertIntermediateDim). (2) MLA decode reordered Q
    // [nope, rope] -> [rope, nope] IN PLACE, clobbering nope channels (decode diverged from
    // prefill from the 2nd position; now via _mlaQRaw). Receipt vs llama-server:
    // DeepSeek-V2-Lite-Chat Q2_K, templated "The capital of France is", 16/16 exact
    // (DeepSeek2GreedyParityTests); Q8_0 on a 195-token prompt: first-token top-5 in the same
    // order with gaps within 0.17 logits, 14/24 exact then a flip at a 0.15-logit near-tie.
    public static readonly ArchitectureDescriptor DeepSeek2 = new()
    {
        Id = "deepseek2",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "DeepSeek-V2/V3/R1 (`deepseek2`)",
        EvidenceDoc = "docs/STATUS.md",
    };

    // deepseek2-ocr — DeepSeek-OCR2 text decoder (llama.cpp src/models/deepseek2.cpp is_ocr branch: plain MHA
    // with q/k/v, full-head NeoX RoPE, theta 1e4, then the DeepSeek MoE FFN). ADMITTED 2026-09-27 after adding
    // it to the NeoX list. Evidence (deepseek-ocr-2-Q4_K_M.gguf, CPU): wikitext [512,1024) PPL 8.7955 vs
    // llama-perplexity 8.7905; OCR of a rendered invoice image matches llama-server --mmproj token for token
    // except one near-tie line break (two newlines vs two spaces + newline, logprob -0.63 vs -0.78 in llama.cpp).
    public static readonly ArchitectureDescriptor DeepSeek2Ocr = new()
    {
        Id = "deepseek2-ocr",
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "text arch `deepseek2-ocr` is admitted",
        EvidenceDoc = "docs/STATUS.md",
    };
}
