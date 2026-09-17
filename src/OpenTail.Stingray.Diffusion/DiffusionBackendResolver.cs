using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Resolves the compute backend for diffusion pipelines.
/// Probes for Vulkan GPU acceleration by default, falling back cleanly to CPU if
/// Vulkan is unavailable or if <c>STINGRAY_BACKEND=cpu</c> is configured.
/// </summary>
public static class DiffusionBackendResolver
{
    public static (IComputeBackend? Backend, bool OwnsBackend) Resolve(IComputeBackend? explicitBackend)
    {
        if (explicitBackend is not null)
        {
            // Caller explicitly supplied a backend; caller owns its lifecycle.
            return (explicitBackend, false);
        }

        if (string.Equals(Environment.GetEnvironmentVariable("STINGRAY_BACKEND"), "cpu", StringComparison.OrdinalIgnoreCase))
        {
            return (null, false);
        }

        try
        {
            var vk = new VulkanBackend();
            return (vk, true);
        }
        catch
        {
            return (null, false);
        }
    }
}
