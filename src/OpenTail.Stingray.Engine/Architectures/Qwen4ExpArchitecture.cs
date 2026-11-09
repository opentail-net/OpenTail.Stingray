namespace OpenTail.Stingray.Engine;

internal static class Qwen4ExpArchitecture
{
    public static readonly ArchitectureDescriptor Descriptor = new()
    {
        Id = "qwen4exp",
        Status = AdmissionStatus.Admitted,
        EvidenceDoc = "docs/2-coverage/2026-11-08-qwen38-remaining-work-plan.md",
        StatusAnchor = "Qwen 3.8 Flash Next (`qwen4exp`)",
        ForwardPassFamily = ForwardPassFamily.HybridGdn, // Gated-DeltaNet hybrid, like qwen35
        ImageProjector = ImageProjectorKind.QwenVl, // the loader accepts an mmproj; SupportsImageInput stays false until verified
        // Admitted 2026-11-09 for text generation and tool calls on the UD-IQ1_S checkpoint. Image input, video,
        // MTP speculation and the per-step MTP oracle are not admitted (waivers in the plan's admission entry).
        CreateForwardPass = BuildForwardPass,
    };

    /// <summary>Loads the qwen4exp trunk for the server. Local investigation only until admission (CLAUDE.md rule 14).</summary>
    private static OpenTail.Stingray.Core.IForwardPass BuildForwardPass(ArchitectureLoadContext ctx)
    {
        var model = ctx.Probe.Gguf ?? throw new InvalidOperationException("qwen4exp needs a GGUF model.");
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        ctx.TrackDisposable(pass);
        return pass;
    }
}
