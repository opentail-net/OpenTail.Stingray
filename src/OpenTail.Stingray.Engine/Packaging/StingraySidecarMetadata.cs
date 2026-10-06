using System.Text.Json.Serialization;

namespace OpenTail.Stingray.Engine.Packaging;

/// <summary>
/// Advisory metadata sidecar (stingray.json) tied to a model/package digest (§3, §8 &amp; §9 of plan).
/// Contains cached evidence, verified capabilities, and architecture interpretations.
/// Invariant: stingray.json never grants admission and never overrides ArchitectureRegistry or current executable policy.
/// </summary>
public sealed record StingraySidecarMetadata(
    [property: JsonPropertyName("schema_version")] int SchemaVersion,
    [property: JsonPropertyName("model_digest")] string ModelDigest,
    [property: JsonPropertyName("verification_profile_version")] string VerificationProfileVersion,
    [property: JsonPropertyName("architecture")] string Architecture,
    [property: JsonPropertyName("semantic_family")] string SemanticFamily,
    [property: JsonPropertyName("state_model")] string StateModel,
    [property: JsonPropertyName("admission")] StingraySidecarAdmission Admission,
    [property: JsonPropertyName("capabilities")] StingraySidecarCapabilities Capabilities
);

/// <summary>
/// Cached admission status and evidence record in stingray.json sidecar.
/// </summary>
public sealed record StingraySidecarAdmission(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("evidence")] string? Evidence = null
);

/// <summary>
/// Verified capability profile in stingray.json sidecar.
/// </summary>
public sealed record StingraySidecarCapabilities(
    [property: JsonPropertyName("verified_backends")] IReadOnlyList<string> VerifiedBackends,
    [property: JsonPropertyName("continuous_batching")] bool ContinuousBatching = false,
    [property: JsonPropertyName("speculation")] IReadOnlyList<string>? Speculation = null,
    [property: JsonPropertyName("notes")] string? Notes = null
);
