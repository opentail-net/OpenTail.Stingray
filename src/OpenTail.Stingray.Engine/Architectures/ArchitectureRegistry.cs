namespace OpenTail.Stingray.Engine;

/// <summary>
/// Lookup over <see cref="BuiltInArchitectures"/>. Admission and per-family behavior are defined
/// by descriptors in the explicit built-in manifest.
/// </summary>
public static class ArchitectureRegistry
{
    private static readonly Dictionary<string, ArchitectureDescriptor> s_byId = Build();

    public static IReadOnlyCollection<ArchitectureDescriptor> All => s_byId.Values.Distinct().ToArray();

    public static bool TryGet(string architecture, out ArchitectureDescriptor descriptor)
    {
        if (architecture is not null && s_byId.TryGetValue(architecture, out var d))
        {
            descriptor = d;
            return true;
        }
        descriptor = null!;
        return false;
    }

    /// <summary>Descriptor for the architecture, or null when it is not registered.</summary>
    public static ArchitectureDescriptor? Find(string? architecture) =>
        architecture is not null && s_byId.TryGetValue(architecture, out var d) ? d : null;

    /// <summary>
    /// Attempts to resolve an architecture descriptor for the specified probe, supporting relabelled file recognition
    /// and metadata-free tensor detection.
    /// </summary>
    public static bool TryResolve(ArchitectureProbe probe, out ArchitectureDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(probe);
        if (!string.IsNullOrEmpty(probe.Architecture))
        {
            if (TryGet(probe.Architecture, out var d))
            {
                // Allow relabelled recognizers to claim specialized files (e.g. llama -> Mistral or MoE)
                foreach (var candidate in All)
                {
                    if (candidate.RecognizeRelabelledFile != null &&
                        candidate.RecognizeRelabelledFile(probe.Architecture, probe))
                    {
                        descriptor = candidate;
                        return true;
                    }
                }
                descriptor = d;
                return true;
            }

            // Check relabelled claimants when alias is unknown
            foreach (var candidate in All)
            {
                if (candidate.RecognizeRelabelledFile != null &&
                    candidate.RecognizeRelabelledFile(probe.Architecture, probe))
                {
                    descriptor = candidate;
                    return true;
                }
            }

            descriptor = null!;
            return false;
        }

        // Metadata-free fallback
        foreach (var candidate in All)
        {
            if (candidate.DetectFromProbe != null && candidate.DetectFromProbe(probe))
            {
                descriptor = candidate;
                return true;
            }
        }

        descriptor = null!;
        return false;
    }

    /// <summary>
    /// Resolves an architecture descriptor for the specified probe, throwing <see cref="NotSupportedException"/> if unrecognized.
    /// </summary>
    public static ArchitectureDescriptor Resolve(ArchitectureProbe probe)
    {
        if (TryResolve(probe, out var descriptor))
            return descriptor;

        if (!string.IsNullOrEmpty(probe?.Architecture))
        {
            throw new NotSupportedException(
                $"GGUF architecture '{probe.Architecture}' is not supported for text generation by OpenTail.Stingray. " +
                "The model was rejected before inference because GGUF tensor naming alone does not establish " +
                "compatible attention, RoPE, normalization, and FFN semantics. Supported profiles: " +
                $"{string.Join(", ", All.Where(d => d.Status == AdmissionStatus.Admitted).SelectMany(d => d.Aliases.Prepend(d.Id)).Distinct().Order())}.");
        }

        throw new NotSupportedException("Model file declares no architecture and no tensor layout detector recognized it.");
    }

    /// <summary>
    /// Resolves an architecture descriptor by name, throwing <see cref="NotSupportedException"/> if unknown.
    /// </summary>
    public static ArchitectureDescriptor Resolve(string architecture)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(architecture);
        if (TryGet(architecture, out var descriptor))
            return descriptor;
        throw new NotSupportedException($"Unsupported architecture '{architecture}'.");
    }

    public static bool ThinkingDefaultOff(string? architecture) => Find(architecture)?.ThinkingDefaultOff ?? false;

    public static FallbackChatFormat FallbackChat(string? architecture) =>
        Find(architecture)?.FallbackChat ?? FallbackChatFormat.ChatMl;

    private static Dictionary<string, ArchitectureDescriptor> Build()
    {
        var map = new Dictionary<string, ArchitectureDescriptor>(StringComparer.OrdinalIgnoreCase);
        foreach (var d in BuiltInArchitectures.Create())
        {
            d.Validate();
            foreach (var name in d.Aliases.Prepend(d.Id))
                if (!map.TryAdd(name, d))
                    throw new InvalidOperationException($"Architecture '{name}' is registered twice.");
        }
        return map;
    }
}
