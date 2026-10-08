namespace OpenTail.Stingray.Engine;

// Part of OtherAdmittedArchitectures (partial), grouped by family. Descriptors are independent of one another.
internal static partial class OtherAdmittedArchitectures
{
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
        UsesNeoxRope = true,
        Status = AdmissionStatus.Admitted,
        StatusExemption = "Covered by the generic 'LLM inference (GGUF)' row; STATUS.md is a capability matrix, not an architecture catalog.",
        EvidenceDoc = "docs/done/01-gguf-model-coverage-plan.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };

    public static readonly ArchitectureDescriptor Exaone4 = new()
    {
        Id = "exaone4",
        UsesNeoxRope = true,
        ApplyModelSemantics = FamilyModelSemantics.Exaone4,
        Status = AdmissionStatus.Admitted,
        StatusAnchor = "EXAONE 4.5 33B (`exaone4`)",
        EvidenceDoc = "docs/STATUS.md",
        CreateForwardPass = CommonForwardPassFactory.CreateDense,
    };
}
