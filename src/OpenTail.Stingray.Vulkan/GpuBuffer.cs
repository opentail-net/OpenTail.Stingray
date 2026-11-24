
namespace OpenTail.Stingray.Vulkan;

/// <summary>
/// A Vulkan buffer with its backing device memory.
/// Tracks size, usage flags, and whether the memory is host-visible (mappable).
/// </summary>
public sealed unsafe class GpuBuffer : IDisposable
{
    private readonly VkDeviceApi _vkd;

    public VkBuffer Buffer { get; }
    public VkDeviceMemory Memory { get; }
    /// <summary>The size requested for the buffer.</summary>
    public ulong Size { get; }
    /// <summary>
    /// The size of the device memory allocation the driver actually made (the allocator's <c>allocationSize</c>, which can
    /// be larger than <see cref="Size"/> after alignment). This is the figure counted by <see cref="VulkanBackend.LiveDeviceLocalBytes"/>.
    /// </summary>
    public ulong AllocatedBytes { get; }
    public bool IsHostVisible { get; }

    /// <summary>
    /// The backend whose <see cref="VulkanBackend.LiveDeviceLocalBytes"/> includes this buffer's <see cref="AllocatedBytes"/>;
    /// null when the memory is not device-local (host-visible staging or fallback memory is not counted).
    /// </summary>
    private readonly VulkanBackend? _countedOn;
    private bool _disposed;

    private GpuBuffer(VkDeviceApi vkd, VkBuffer buffer, VkDeviceMemory memory, ulong size, ulong allocatedBytes,
        bool hostVisible, VulkanBackend? countedOn)
    {
        _vkd = vkd;
        Buffer = buffer;
        Memory = memory;
        Size = size;
        AllocatedBytes = allocatedBytes;
        IsHostVisible = hostVisible;
        _countedOn = countedOn;
    }

    /// <summary>
    /// Create a device-local buffer (VRAM, not host-visible).
    /// Used for weight storage and compute scratch buffers.
    /// Falls back to host-visible / host-coherent memory on unified-memory GPUs (APUs/iGPUs)
    /// or when dedicated VRAM is exhausted.
    /// </summary>
    public static GpuBuffer CreateDeviceLocal(VulkanBackend backend, ulong size, VkBufferUsageFlags usage)
    {
        try
        {
            return Create(backend, size, usage | VkBufferUsageFlags.TransferDst,
                VkMemoryPropertyFlags.DeviceLocal);
        }
        catch
        {
            return Create(backend, size, usage | VkBufferUsageFlags.TransferDst,
                VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent);
        }
    }

    /// <summary>
    /// Create a host-visible, host-coherent staging buffer.
    /// Used for CPU→GPU and GPU→CPU data transfers.
    /// </summary>
    public static GpuBuffer CreateStaging(VulkanBackend backend, ulong size, VkBufferUsageFlags usage)
    {
        return Create(backend, size, usage,
            VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent);
    }

    /// <summary>
    /// Create a pinned host-visible buffer accessible from both CPU and GPU.
    /// Tries BAR memory (HostVisible + DeviceLocal) first for zero-copy GPU access,
    /// falls back to plain host-visible if BAR is not available.
    /// Used for small, frequently-transferred data like hidden states at GPU↔CPU boundaries.
    /// </summary>
    public static GpuBuffer CreatePinned(VulkanBackend backend, ulong size, VkBufferUsageFlags usage)
    {
        // Try resizable BAR: host-visible + device-local (best perf, zero-copy from GPU)
        try
        {
            return Create(backend, size,
                usage | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst,
                VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent | VkMemoryPropertyFlags.DeviceLocal);
        }
        catch
        {
            // Fallback: plain host-visible (requires staging for GPU access)
            return Create(backend, size,
                usage | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst,
                VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent);
        }
    }

    /// <summary>
    /// Map the buffer memory for CPU access. Only valid for host-visible buffers.
    /// Returns a pointer to the mapped region.
    /// </summary>
    public void* Map()
    {
        if (!IsHostVisible)
            throw new InvalidOperationException("Cannot map device-local buffer");
        void* ptr;
        _vkd.vkMapMemory(Memory, 0, Size, 0, &ptr).CheckResult();
        return ptr;
    }

    /// <summary>Unmap previously mapped memory.</summary>
    public void Unmap() => _vkd.vkUnmapMemory(Memory);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _vkd.vkDestroyBuffer(Buffer, null);
        _vkd.vkFreeMemory(Memory, null);
        _countedOn?.TrackDeviceLocalFree(AllocatedBytes);
    }

    private static GpuBuffer Create(VulkanBackend backend, ulong size,
        VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags)
    {
        var vkd = backend.Vkd;

        // Create buffer
        VkBufferCreateInfo bufferCI = new()
        {
            size = size,
            usage = usage,
            sharingMode = VkSharingMode.Exclusive,
        };
        VkBuffer buffer;
        vkd.vkCreateBuffer(&bufferCI, null, &buffer).CheckResult();

        // Query memory requirements
        VkMemoryRequirements memReqs;
        vkd.vkGetBufferMemoryRequirements(buffer, &memReqs);

        // Allocate device memory
        VkMemoryAllocateInfo allocInfo = new()
        {
            allocationSize = memReqs.size,
            memoryTypeIndex = backend.FindMemoryType(memReqs.memoryTypeBits, memoryFlags),
        };
        VkDeviceMemory memory;
        var allocRes = vkd.vkAllocateMemory(&allocInfo, null, &memory);
        if (allocRes != VkResult.Success)
        {
            vkd.vkDestroyBuffer(buffer, null);
            allocRes.CheckResult();
        }

        // Bind buffer to memory
        var bindRes = vkd.vkBindBufferMemory(buffer, memory, 0);
        if (bindRes != VkResult.Success)
        {
            vkd.vkDestroyBuffer(buffer, null);
            vkd.vkFreeMemory(memory, null);
            bindRes.CheckResult();
        }

        bool hostVisible = (memoryFlags & VkMemoryPropertyFlags.HostVisible) != 0;
        // Only memory requested as DeviceLocal counts toward the backend's live device-local total. Staging buffers
        // (HostVisible only) and the host-visible fallback taken by CreateDeviceLocal are not device-local and are excluded.
        // Counted after the bind succeeds, so a failed allocation/bind never leaves a count behind.
        bool deviceLocal = (memoryFlags & VkMemoryPropertyFlags.DeviceLocal) != 0;
        if (deviceLocal)
            backend.TrackDeviceLocalAllocation(allocInfo.allocationSize);
        return new GpuBuffer(vkd, buffer, memory, size, allocInfo.allocationSize, hostVisible,
            deviceLocal ? backend : null);
    }
}
