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

        HardwareProfile hw;
        try
        {
            hw = HardwareProfile.Detect(vk);
        }
        finally
        {
            // Detect reads only VramBytes from this probe device and keeps no reference to it. Left undisposed, each call leaked a
            // Vulkan instance and device, and every model load runs it (InferenceEngineLoader.BuildPlan).
            vk?.Dispose();
        }
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
