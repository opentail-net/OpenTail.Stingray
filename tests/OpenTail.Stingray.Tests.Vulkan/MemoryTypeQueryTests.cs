using System.Text;
using Vortice.Vulkan;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Diagnostic query for A31 (docs/00-current-work.md): which memory types this device advertises, and whether a host-visible,
/// host-cached type exists. The download staging buffer is created HostVisible | HostCoherent only, so a host-cached type would
/// be the candidate for faster device-to-host reads. Prints the table and asserts only that the query is consistent; whether to
/// use a cached type is decided by the timing in the next step, not here.
/// </summary>
public sealed class MemoryTypeQueryTests
{
    [Fact]
    public void DeviceMemoryTypes_AreListed_AndTheHostCachedOptionIsReported()
    {
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        var types = gpu!.MemoryTypes;
        Console.WriteLine($"[mem-types] {types.Count} memory types");
        foreach (var t in types)
            Console.WriteLine($"[mem-types] index {t.Index}  flags 0x{(uint)t.Flags:X2}  {Describe(t.Flags),-44}  heap {t.HeapIndex} ({t.HeapSize / (1ul << 20)} MiB)");

        bool hostCached = types.Any(t => t.Flags.HasFlag(VkMemoryPropertyFlags.HostVisible) && t.Flags.HasFlag(VkMemoryPropertyFlags.HostCached));
        Console.WriteLine($"[mem-types] host-visible + host-cached type present: {hostCached}");

        Assert.NotEmpty(types);
        Assert.Contains(types, t => t.Flags.HasFlag(VkMemoryPropertyFlags.HostVisible));
    }

    private static string Describe(VkMemoryPropertyFlags flags)
    {
        var sb = new StringBuilder();
        void Add(VkMemoryPropertyFlags bit, string name)
        {
            if (flags.HasFlag(bit)) { if (sb.Length > 0) sb.Append(" | "); sb.Append(name); }
        }
        Add(VkMemoryPropertyFlags.DeviceLocal, "DeviceLocal");
        Add(VkMemoryPropertyFlags.HostVisible, "HostVisible");
        Add(VkMemoryPropertyFlags.HostCoherent, "HostCoherent");
        Add(VkMemoryPropertyFlags.HostCached, "HostCached");
        Add(VkMemoryPropertyFlags.LazilyAllocated, "LazilyAllocated");
        return sb.Length == 0 ? "none" : sb.ToString();
    }
}
