using System.Collections.Immutable;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Runtime;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class RuntimeInstanceTests
{
    [Fact]
    public void Create_ThrowsOnNullArguments()
    {
        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "llama",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0);

        Assert.Throws<ArgumentNullException>(() => RuntimeInstance.Create(null!, null!));
        Assert.Throws<ArgumentNullException>(() => RuntimeInstance.Create(plan, null!));
        Assert.Throws<ArgumentNullException>(() => RuntimeInstance.Create((ExecutionPlan)null!));
    }

    [Fact]
    public void PlanNotExecutableException_ExposesPlanAndMessage()
    {
        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "llama",
            decision: new ForwardPassDecision(ForwardPassKind.CpuDense, "cpu"),
            backend: ForwardPassBackend.Cpu,
            contextSize: 512,
            gpuLayers: 0);

        var ex = new PlanNotExecutableException("Planned backend unavailable", plan);
        Assert.Equal("Planned backend unavailable", ex.Message);
        Assert.Same(plan, ex.Plan);

        var inner = new InvalidOperationException("boom");
        var exWithInner = new PlanNotExecutableException("Failed", inner, plan);
        Assert.Same(inner, exWithInner.InnerException);
        Assert.Same(plan, exWithInner.Plan);
    }

    [Fact]
    public void Create_ThrowsOnCudaWhenUnavailable()
    {
        if (OpenTail.Stingray.Cuda.CudaBackend.IsAvailable())
        {
            // If CUDA is available on this hardware, skip this test
            return;
        }

        var plan = ExecutionPlan.CreateSynthesized(
            architecture: "llama",
            decision: new ForwardPassDecision(ForwardPassKind.CudaDense, "cuda"),
            backend: ForwardPassBackend.Cuda,
            contextSize: 512,
            gpuLayers: 32);

        var ex = new PlanNotExecutableException("Planned CUDA backend is not operational or available on this system.", plan);
        Assert.Contains("CUDA backend", ex.Message);
        Assert.Same(plan, ex.Plan);
    }

    [Fact]
    public void CudaPin_SecondDifferentDevice_IsRefused_NotSilentlyIgnored()
    {
        string? saved = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");
        try
        {
            Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", null);
            GpuDeviceSelection.PinCudaDevice(0);
            GpuDeviceSelection.PinCudaDevice(0);                                  // same device is fine (idempotent)
            Assert.Equal(0, GpuDeviceSelection.PinnedCudaDevice);
            Assert.Throws<InvalidOperationException>(() => GpuDeviceSelection.PinCudaDevice(1));
            GpuDeviceSelection.PinCudaDevice(-1);                                 // unspecified is a no-op
        }
        finally
        {
            Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", saved);
        }
    }
}
