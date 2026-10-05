namespace OpenTail.Stingray.Engine;

/// <summary>
/// Lookup over <see cref="BuiltInArchitectures"/>. Architectures not yet migrated to a descriptor
/// are still governed by <see cref="ModelCompatibility"/>'s legacy allowlist; a descriptor, when
/// present, wins.
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

    /// <summary>Descriptor for the architecture, or null when it is still on the legacy path.</summary>
    public static ArchitectureDescriptor? Find(string? architecture) =>
        architecture is not null && s_byId.TryGetValue(architecture, out var d) ? d : null;

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
