
namespace OpenTail.Stingray.Vulkan;

/// <summary>
/// Opt-in feature flag configuration for Vulkan Path 2 Cooperative Tiled GEMM (<c>MatMulTiledQ4K</c>).
/// Gated behind <c>STINGRAY_VULKAN_PATH2=1</c>.
/// Defaults to <b>false (disabled)</b>.
/// </summary>
public static class VulkanPath2Config
{
    private static bool _isEnabled = ReadFromEnvironment();

    /// <summary>
    /// Gets or sets whether experimental Vulkan Path 2 (<c>MatMulTiledQ4K</c>) is active.
    /// </summary>
    public static bool IsEnabled
    {
        get => _isEnabled;
        set => _isEnabled = value;
    }

    /// <summary>
    /// Maximum sequence batch size for Path 2 tiled GEMM dispatches.
    /// Default: 16 tokens.
    /// </summary>
    public static int MaxTileTokens { get; set; } = 16;

    /// <summary>
    /// Reads <c>STINGRAY_VULKAN_PATH2</c> from environment.
    /// </summary>
    public static bool ReadFromEnvironment()
    {
        string? val = Environment.GetEnvironmentVariable("STINGRAY_VULKAN_PATH2");
        if (string.IsNullOrWhiteSpace(val)) return false;
        val = val.Trim();
        return val == "1" || val.Equals("true", StringComparison.OrdinalIgnoreCase);
    }
}

/// <summary>
/// Dispatcher for experimental Vulkan Path 2 shared-memory tiled Q4_K matrix multiplication.
/// Operates in a dedicated, isolated file without mutating core Vulkan pipelines.
/// </summary>
public static class VulkanPath2Dispatcher
{
    /// <summary>
    /// Evaluates whether Vulkan Path 2 tiled GEMM can be attempted for the given shape.
    /// </summary>
    public static bool CanAttemptPath2(DType weightDType, int nTok, int cols)
    {
        if (!VulkanPath2Config.IsEnabled)
            return false;

        if (weightDType is not (DType.Q4_K or DType.Q6_K))
            return false;

        if (nTok > VulkanPath2Config.MaxTileTokens || cols % 256 != 0)
            return false;

        return true;
    }
}
