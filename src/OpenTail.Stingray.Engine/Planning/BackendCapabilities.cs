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
    public static BackendCapabilities Detect(bool noGpuProbe = false, int deviceIndex = -1)
    {
        if (noGpuProbe)
        {
            return new BackendCapabilities(
                CudaAvailable: false,
                VulkanAvailable: false,
                HardwareProfile: HardwareProfile.Detect(null as VulkanBackend),
                RecommendedThreadCount: Environment.ProcessorCount);
        }

        // CUDA_VISIBLE_DEVICES is read at first CUDA init, so pin before the availability probe below.
        OpenTail.Stingray.Engine.Runtime.GpuDeviceSelection.PinCudaDevice(deviceIndex);
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
            vk = TryCreateVulkan(deviceIndex < 0 ? 0 : deviceIndex);
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

    private static VulkanBackend? TryCreateVulkan(int deviceIndex)
    {
        try { return new VulkanBackend(deviceIndex); }
        catch { return null; }
    }
}
