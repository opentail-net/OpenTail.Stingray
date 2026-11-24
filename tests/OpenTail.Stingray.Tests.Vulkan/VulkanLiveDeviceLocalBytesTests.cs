using global::OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// <see cref="VulkanBackend.LiveDeviceLocalBytes"/>: device-local memory held by live buffers of one backend instance.
/// Needs a real Vulkan device, so it is heavy-gated with the rest of this project (set <c>STINGRAY_RUN_HEAVY_TESTS=1</c>).
/// </summary>
public sealed class VulkanLiveDeviceLocalBytesTests : HeavyTestBase
{
    private static VulkanBackend CreateBackendOrSkip()
    {
        try
        {
            return new VulkanBackend();
        }
        catch (Exception ex)
        {
            Assert.Skip($"Vulkan device could not be created in this environment: {ex.Message}");
            throw;   // unreachable: Assert.Skip throws
        }
    }

    [Fact]
    public void DeviceLocalAllocationRaisesCounterByActualAllocationAndFreeRestoresIt()
    {
        using var backend = CreateBackendOrSkip();
        long before = backend.LiveDeviceLocalBytes;

        // 1 Mi float32 = 4 MiB requested; the counter must use the allocator's actual allocationSize, which can be larger.
        var tensor = backend.Allocate(TensorShape.D1(1 << 20));
        GpuBuffer buf = backend.GetBuffer(tensor);
        if (buf.IsHostVisible)
        {
            backend.Free(tensor);
            Assert.Skip("CreateDeviceLocal fell back to host-visible memory on this device; no device-local allocation to count.");
        }

        long afterAlloc = backend.LiveDeviceLocalBytes;
        Assert.True(buf.AllocatedBytes >= buf.Size, "allocator returned less memory than requested");
        Assert.Equal(before + (long)buf.AllocatedBytes, afterAlloc);

        // Host-visible staging memory is not device-local and must not move the counter, not even while it is alive.
        using (var staging = GpuBuffer.CreateStaging(backend, 1 << 20, Vortice.Vulkan.VkBufferUsageFlags.TransferSrc))
        {
            Assert.True(staging.IsHostVisible);
            Assert.Equal(afterAlloc, backend.LiveDeviceLocalBytes);
        }
        Assert.Equal(afterAlloc, backend.LiveDeviceLocalBytes);

        backend.Free(tensor);
        Assert.Equal(before, backend.LiveDeviceLocalBytes);
    }
}
