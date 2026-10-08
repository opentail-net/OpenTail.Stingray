namespace OpenTail.Stingray.Engine;

internal static class RwkvArchitectures
{
    // rwkv7 — admitted 2026-10-01. Runs on its own Rwkv7ForwardPass (CPU only, recurrent, no
    // KV cache), routed by RunCommand. Receipt: Rwkv7GreedyParityTests vs llama-server (vendored
    // tools/llama.cpp) on RWKV7-Goose-World3-1.5B-HF Q8_0, teacher-forced over llama's greedy
    // tokens: 16/16 on "The capital of France is" and 31/32 on a 20-token story prompt, the one
    // miss at llama's own 0.027-nat top-2 tie; chosen-token |Δlogprob| mean 0.013, max 0.11.
    // Tokenizer (RWKV world trie) matches llama-tokenize on CJK/emoji/CR-LF goldens. Caveat:
    // the same model's Q4_K_S file tracks llama less closely (30/32, two misses at 0.12 and
    // 0.21 nat margins, mean 0.053); activation q8_K and BF16 rounding were tested and ruled out.
    // Its perplexity still matches: wikitext-2 test, 2048-token window, positions 1024+ (what
    // llama-perplexity -c 2048 --chunks 1 scores): Q8_0 6.9318 vs llama.cpp 6.9136, Q4_K_S
    // 7.2912 vs 7.3133 (both within 0.3%, inside llama's ±0.52/±0.56).
    public static readonly ArchitectureDescriptor Rwkv7 = new()
    {
        Id = "rwkv7",
        Traits = new() { HeadDimFromWkvHeadSize = true },
        Status = AdmissionStatus.Admitted,
        // CPU-only routing is enforced by RunCommand.cs:2016-2026 and InferenceEngineLoader.cs:603-612.
        ForwardPassFamily = ForwardPassFamily.Rwkv,
        SupportedBackends = SupportedBackends.Cpu,
        BackendLimitation = "RWKV has a CPU forward pass only; GPU requests fall back to CPU.",
        StatusAnchor = "RWKV-7 Goose (`rwkv7`)",
        EvidenceDoc = "docs/STATUS.md",
        SupportsContinuousBatching = false,
        CreateForwardPass = ctx =>
        {
            var rwkv = RwkvForwardPassBase.Create(ctx.Probe.Gguf!);
            ctx.TrackDisposable(rwkv);
            return rwkv;
        },
    };

    // rwkv6 — admitted 2026-10-01. Runs on Rwkv6ForwardPass (CPU only, recurrent; shares
    // RwkvForwardPassBase with rwkv7). Receipt: Rwkv6GreedyParityTests vs llama-server on
    // rwkv-6-world-1.6b Q8_0, teacher-forced: 16/16 and 31/32 argmax, the miss a 0.007-nat tie in
    // llama's own top-2; chosen-token |Δlogprob| mean 0.022/0.031. Wikitext-2 PPL, positions
    // 1024+ of a 2048 window: Q8_0 8.9786 vs llama.cpp 8.9287 (±0.72), Q4_K_S 9.9818 vs 9.9510
    // (±0.81). The QRWKV variant (no time_first, gated linear attention) is not handled.
    public static readonly ArchitectureDescriptor Rwkv6 = new()
    {
        Id = "rwkv6",
        Traits = new() { HeadDimFromWkvHeadSize = true },
        Status = AdmissionStatus.Admitted,
        // CPU-only routing is enforced by RunCommand.cs:2016-2026 and InferenceEngineLoader.cs:603-612.
        ForwardPassFamily = ForwardPassFamily.Rwkv,
        SupportedBackends = SupportedBackends.Cpu,
        BackendLimitation = "RWKV has a CPU forward pass only; GPU requests fall back to CPU.",
        StatusAnchor = "RWKV-6 Finch (`rwkv6`)",
        EvidenceDoc = "docs/STATUS.md",
        SupportsContinuousBatching = false,
        CreateForwardPass = ctx =>
        {
            var rwkv = RwkvForwardPassBase.Create(ctx.Probe.Gguf!);
            ctx.TrackDisposable(rwkv);
            return rwkv;
        },
    };
}
