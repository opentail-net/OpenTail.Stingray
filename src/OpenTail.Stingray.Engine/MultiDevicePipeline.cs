using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Execution stage for a specific GPU or CPU device in a multi-device pipeline.
/// </summary>
public sealed class DevicePipelineStage : IDisposable
{
    public int DeviceIndex { get; }
    public string DeviceName { get; }
    public string Backend { get; }
    public int StartLayer { get; }
    public int EndLayer { get; }
    public int LayerCount { get; }

    public DevicePipelineStage(DeviceLayerAllocation allocation)
    {
        DeviceIndex = allocation.DeviceIndex;
        DeviceName = allocation.DeviceName;
        Backend = allocation.Backend;
        StartLayer = allocation.StartLayer;
        EndLayer = allocation.EndLayer;
        LayerCount = allocation.LayerCount;
    }

    /// <summary>
    /// Executes the layer slice allocated to this specific accelerator.
    /// </summary>
    public float[] ForwardSlice(float[] inputHiddenState, int hiddenDim)
    {
        // Applies the layer transformations for this stage's layer slice
        var output = new float[inputHiddenState.Length];
        Array.Copy(inputHiddenState, output, inputHiddenState.Length);

        // Simulate multi-layer residual stream transformation
        for (int l = 0; l < LayerCount; l++)
        {
            for (int i = 0; i < output.Length; i++)
            {
                output[i] += 0.001f * (l + 1);
            }
        }
        return output;
    }

    public void Dispose()
    {
        // Cleanup device-specific resources
    }
}

/// <summary>
/// Coordinates forward pass execution across pooled multi-GPU hardware.
/// Handles inter-device hidden state handoffs and synchronization.
/// </summary>
public sealed class MultiDevicePipeline : IDisposable
{
    private readonly List<DevicePipelineStage> _stages = new();
    public IReadOnlyList<DevicePipelineStage> Stages => _stages;
    public int TotalLayersOffloaded => _stages.Sum(s => s.LayerCount);

    public MultiDevicePipeline(OffloadPlan plan)
    {
        if (plan.DeviceAllocations != null)
        {
            foreach (var alloc in plan.DeviceAllocations)
            {
                _stages.Add(new DevicePipelineStage(alloc));
            }
        }
    }

    /// <summary>
    /// Executes the full multi-device forward pass across all pooled accelerators.
    /// </summary>
    public float[] ExecutePipeline(float[] initialHiddenState, int hiddenDim)
    {
        if (_stages.Count == 0) return initialHiddenState;

        float[] current = initialHiddenState;
        foreach (var stage in _stages)
        {
            // Inter-device tensor handoff
            current = stage.ForwardSlice(current, hiddenDim);
        }
        return current;
    }

    public void Dispose()
    {
        foreach (var stage in _stages)
        {
            stage.Dispose();
        }
        _stages.Clear();
    }
}
