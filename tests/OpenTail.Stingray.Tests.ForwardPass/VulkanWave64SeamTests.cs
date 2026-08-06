using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Hand-computed-reference seam tests specifically targeting Wave64 (AMD Vega/GCN) hardware
/// subgroup behavior on Vulkan matvec operations. Verifies that all logical rows in a workgroup
/// compute correct dot products and write their outputs without subgroup reduction corruption or
/// unwritten (0.0000) slots.
/// </summary>
public sealed unsafe class VulkanWave64SeamTests
{
    private static VulkanBackend? TryCreateBackend()
    {
        try
        {
            return new VulkanBackend();
        }
        catch
        {
            return null;
        }
    }

    [Fact]
    public void MatVecF32Wave64Seam_EveryRowWrittenCorrectly()
    {
        using var backend = TryCreateBackend();
        if (backend is null) return;

        // 8 output rows (1 workgroup of 256 threads, 32 lanes per row), 64 columns per row.
        const int rows = 8;
        const int cols = 64;

        var weights = new float[rows * cols];
        var input = new float[cols];

        for (int c = 0; c < cols; c++) input[c] = c + 1.0f;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                weights[r * cols + c] = (r + 1) * 0.1f;
            }
        }

        // Hand-computed expected outputs for each row r:
        // dot(row r, input) = (r + 1) * 0.1 * sum(1..64)
        // sum(1..64) = 64 * 65 / 2 = 2080.
        // expected[r] = (r + 1) * 0.1 * 2080 = (r + 1) * 208.0
        var expected = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            expected[r] = (r + 1) * 208.0f;
        }

        var gpuW = backend.Upload(weights, TensorShape.D2(rows, cols));
        var gpuIn = backend.Upload(input, TensorShape.D1(cols));
        var gpuOut = backend.Allocate(TensorShape.D1(rows));

        backend.MatMul(gpuOut, gpuW, gpuIn, DType.Float32);

        var actual = new float[rows];
        backend.Download(gpuOut, actual);

        for (int r = 0; r < rows; r++)
        {
            Assert.False(actual[r] == 0.0f, $"Row [{r}] output was 0.0000 (never written)!");
            Assert.True(MathF.Abs(actual[r] - expected[r]) < 1e-2f,
                $"Row [{r}] mismatch: expected {expected[r]}, got {actual[r]}");
        }

        backend.Free(gpuW);
        backend.Free(gpuIn);
        backend.Free(gpuOut);
    }
}
