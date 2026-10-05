#nullable enable

namespace OpenTail.Stingray;

/// <summary>
/// Contract for parameters required to load and initialize model weights.
/// Inspired by LLamaSharp's <c>IModelParams</c>, adapted for Stingray's multi-backend architecture.
/// </summary>
public interface IModelParams
{
    /// <summary>Path to the model file on disk (GGUF or SafeTensors).</summary>
    string ModelPath { get; }

    /// <summary>Preferred compute backend (e.g. "auto", "cpu", "vulkan", "cuda").</summary>
    string? Backend { get; }

    /// <summary>
    /// Number of transformer layers to offload to GPU memory.
    /// <c>-1</c> offloads all supported layers; <c>0</c> runs entirely on CPU.
    /// </summary>
    int GpuLayerCount { get; }

    /// <summary>Device index of the primary GPU device (default 0).</summary>
    int MainGpu { get; }

    /// <summary>Fraction of VRAM/layers allocated per device when multi-device placement is used.</summary>
    IReadOnlyList<float>? TensorSplit { get; }

    /// <summary>Whether to use memory-mapped file I/O for model weight loading (default true).</summary>
    bool UseMemoryMap { get; }

    /// <summary>Whether to lock model pages in physical RAM (mlock) to prevent swapping.</summary>
    bool UseMemoryLock { get; }

    /// <summary>Whether to allow loading architectures marked as experimental or unverified.</summary>
    bool AllowUnverifiedArch { get; }

    /// <summary>Optional secondary draft model path for speculative decoding.</summary>
    string? DraftModelPath { get; }

    /// <summary>Whether to enable prompt lookup self-speculative decoding.</summary>
    bool DraftLookup { get; }

    /// <summary>Optional secondary DSpark draft model path.</summary>
    string? DSparkModelPath { get; }
}

/// <summary>
/// Configuration for loading model weights into memory and GPU resources.
/// </summary>
public record ModelParams(string ModelPath) : IModelParams
{
    /// <summary>Preferred compute backend (e.g. "auto", "cpu", "vulkan", "cuda"). Default is "auto".</summary>
    public string? Backend { get; init; } = "auto";

    /// <summary>
    /// Number of transformer layers to offload to GPU memory.
    /// <c>-1</c> offloads all supported layers; <c>0</c> runs entirely on CPU. Default is -1.
    /// </summary>
    public int GpuLayerCount { get; init; } = -1;

    /// <summary>Device index of the primary GPU device (default 0).</summary>
    public int MainGpu { get; init; } = 0;

    /// <summary>Fraction of VRAM/layers allocated per device when multi-device placement is used.</summary>
    public IReadOnlyList<float>? TensorSplit { get; init; }

    /// <summary>Whether to use memory-mapped file I/O for model weight loading (default true).</summary>
    public bool UseMemoryMap { get; init; } = true;

    /// <summary>Whether to lock model pages in physical RAM (mlock) to prevent swapping.</summary>
    public bool UseMemoryLock { get; init; } = false;

    /// <summary>Whether to allow loading architectures marked as experimental or unverified.</summary>
    public bool AllowUnverifiedArch { get; init; } = false;

    /// <summary>Optional secondary draft model path for speculative decoding.</summary>
    public string? DraftModelPath { get; init; }

    /// <summary>Whether to enable prompt lookup self-speculative decoding.</summary>
    public bool DraftLookup { get; init; } = false;

    /// <summary>Optional secondary DSpark draft model path.</summary>
    public string? DSparkModelPath { get; init; }
}
