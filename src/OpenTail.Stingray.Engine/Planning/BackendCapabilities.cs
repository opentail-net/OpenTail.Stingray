#nullable enable

using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Engine.Planning;

/// <summary>
/// Probed hardware and accelerator capabilities available to <see cref="ExecutionPlanner"/>.
/// </summary>
public sealed record BackendCapabilities(
    bool CudaAvailable,
    bool VulkanAvailable,
    HardwareProfile HardwareProfile,
    string? CudaDeviceName = null,
    string? VulkanDeviceName = null,
    int RecommendedThreadCount = 0)
{
    /// <summary>
    /// Probes the local system for available accelerator backends and memory capacity.
    /// </summary>
    public static BackendCapabilities Detect(bool noGpuProbe = false)
    {
        if (noGpuProbe)
        {
            return new BackendCapabilities(
                CudaAvailable: false,
                VulkanAvailable: false,
                HardwareProfile: HardwareProfile.Detect(null as VulkanBackend),
                RecommendedThreadCount: Environment.ProcessorCount);
        }

        bool cudaAvailable = false;
        string? cudaName = null;
        try
        {
            cudaAvailable = CudaBackend.IsAvailable();
            if (cudaAvailable) cudaName = "CUDA Device 0";
        }
        catch { /* best effort */ }

        bool vulkanAvailable = false;
        string? vulkanName = null;
        VulkanBackend? vk = null;
        try
        {
            vk = TryCreateVulkan();
            if (vk != null)
            {
                vulkanAvailable = true;
                vulkanName = "Vulkan Device 0";
            }
        }
        catch { /* best effort */ }

        var hw = HardwareProfile.Detect(vk);
        return new BackendCapabilities(
            CudaAvailable: cudaAvailable,
            VulkanAvailable: vulkanAvailable,
            HardwareProfile: hw,
            CudaDeviceName: cudaName,
            VulkanDeviceName: vulkanName,
            RecommendedThreadCount: Environment.ProcessorCount);
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(deviceIndex: 0); }
        catch { return null; }
    }
}
