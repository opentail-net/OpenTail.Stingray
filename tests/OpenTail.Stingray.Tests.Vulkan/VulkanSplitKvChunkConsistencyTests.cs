using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// The split-KV slice size exists in two places that cannot share a symbol: the C# constant
/// <see cref="VulkanBackend.SplitKvChunk"/> (which drives the nSplits formulas, the partial-buffer
/// sizing in GpuForwardPass, and the tests) and the GLSL <c>const uint CHUNK</c> plus its
/// <c>sk_scores[CHUNK]</c> shared array inside each AttentionSplitKvPartial* shader.
///
/// <para>When those two disagree the host allocates partial buffers for one split count while the
/// shader writes another — silently producing wrong attention output rather than failing loudly.
/// That is exactly how changing the chunk from 512 to 256 broke six split-KV tests: a fifth,
/// unnoticed copy of the constant lived in the test helpers. This test pins the remaining
/// unavoidable duplication so the next change to the slice size fails here, immediately and with
/// an obvious message, instead of in a numerical parity test.</para>
/// </summary>
public sealed class VulkanSplitKvChunkConsistencyTests
{
    // Referenced directly rather than by reflection: the Vulkan assembly already grants
    // InternalsVisibleTo this test project, and the trim analyzers (errors, not warnings, in this
    // repo) reject Assembly.GetType/GetField.
    public static TheoryData<string, string> Shaders() => new()
    {
        { nameof(Vulkan.Shaders.AttentionSplitKvPartial),     Vulkan.Shaders.AttentionSplitKvPartial },
        { nameof(Vulkan.Shaders.AttentionSplitKvPartialBf16), Vulkan.Shaders.AttentionSplitKvPartialBf16 },
        { nameof(Vulkan.Shaders.AttentionSplitKvPartialQ8),   Vulkan.Shaders.AttentionSplitKvPartialQ8 },
    };

    [Theory]
    [MemberData(nameof(Shaders))]
    public void ShaderChunkMatchesBackendConstant(string shaderName, string src)
    {
        uint chunk = VulkanBackend.SplitKvChunk;

        Assert.True(src.Contains($"const uint CHUNK = {chunk}u;", StringComparison.Ordinal),
            $"{shaderName}: GLSL 'const uint CHUNK' does not match VulkanBackend.SplitKvChunk ({chunk}). " +
            "Host and shader must agree or the partial buffers are sized for the wrong split count.");

        Assert.True(src.Contains($"sk_scores[{chunk}]", StringComparison.Ordinal),
            $"{shaderName}: shared score array is not sized to CHUNK ({chunk}). " +
            "A slice can hold CHUNK scores; a smaller array overruns, a larger one wastes LDS.");
    }

    /// <summary>The combine shader indexes a 256-entry shared array, which caps the split count.</summary>
    [Fact]
    public void MaxSeqLenRespectsTheCombineShaderSplitBound()
    {
        Assert.Equal(VulkanBackend.SplitKvChunk * 256, VulkanBackend.SplitKvMaxSeqLen);
    }
}
