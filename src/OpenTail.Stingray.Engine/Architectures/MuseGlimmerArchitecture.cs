namespace OpenTail.Stingray.Engine;

internal static class MuseGlimmerArchitecture
{
    // Admitted 2026-10-03, TEXT ONLY, CPU ONLY. Ported from llama.cpp src/models/muse-glimmer.cpp
    // (SWA period 4, RoPE only on SWA layers, attention output gate, 1e-8 post-norms, unweighted
    // embedding RMSNorm, logit scale then optional softcap). Evidence vs llama.cpp bed0a8566 on
    // Muse-Glimmer-30B UD-Q4_K_XL (greedy): admit-arch 8/8 exact, three further prompts identical until
    // near-ties (0.006 and 0.05 nats), and a 3,748-token prompt past the real sliding window identical.
    // Limits: per-token prefill only (batched prefill, PrefillWithCache and BatchForwardMulti refuse it,
    // the server does not batch it), GPU passes refuse it, vision tower and DFlash drafter not ported.
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "muse-glimmer",
        Aliases = ["muse_glimmer"],
        Status = AdmissionStatus.Admitted,
        // GPU passes reject the embedding norm / attention output gate: GpuForwardPass.cs:546-551.
        ForwardPassFamily = ForwardPassFamily.Dense,
        SupportedBackends = SupportedBackends.Cpu,
        BackendLimitation = "Muse-Glimmer's attention output gate and embedding norm are supported by the CPU pass only.",
        StatusAnchor = "Muse-Glimmer 30B (`muse-glimmer`)",
        EvidenceDoc = "docs/2-coverage/2026-10-03-muse-glimmer-port-plan.md",
        SupportsContinuousBatching = false,
    };
}
