namespace OpenTail.Stingray.Engine;

public sealed class ExperimentalQuantGate
{
    public required string Id { get; init; }
    public required Func<DType, bool> TensorTypePredicate { get; init; }
    public required string EnvironmentVariable { get; init; }
    public required string Reason { get; init; }
    public required string EvidenceDoc { get; init; }

    public bool IsEnabled() => Environment.GetEnvironmentVariable(EnvironmentVariable) == "1";

    public string RefusalMessage =>
        $"{Reason}; set {EnvironmentVariable}=1 to try it (CPU only, outputs unverified).";
}

public static class ExperimentalQuantGateRegistry
{
    public static IReadOnlyList<ExperimentalQuantGate> All { get; } =
    [
        new()
        {
            Id = "bonsai2-prism",
            TensorTypePredicate = Cpu.BonsaiQuant.IsBonsaiType,
            EnvironmentVariable = "STINGRAY_EXPERIMENTAL_PRISM",
            Reason = "This GGUF uses Bonsai2 PRISM weights (PQ2_0/PTQ1_0 with Hadamard transforms). " +
                "Support is ported but not yet verified against the publisher's reference",
            EvidenceDoc = "docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md",
        },
    ];
}
