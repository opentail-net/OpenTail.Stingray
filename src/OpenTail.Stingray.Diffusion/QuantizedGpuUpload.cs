using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>Raw GGUF tensor access (e.g. <see cref="IWeightLoader.TryGetRaw"/>), used to keep
/// block-quantized weights quantized on the GPU.</summary>
public delegate bool RawWeightSource(string name, out nint data, out long byteLen, out DType dtype, out int rows, out int cols);

/// <summary>
/// Shared "upload this weight still quantized" step for the diffusion GPU weight uploaders.
/// When the backend's Sgemm consumes raw K-quant blocks (Vulkan: dequant-in-shader
/// SgemmQ*K / MatVecDqQ*K / SgemmSiluGateQ*K), a Q3_K/Q4_K/Q5_K tensor is uploaded as its GGUF
/// bytes (3.4-5.5 bits per weight instead of 16) and never dequantized on the host.
/// </summary>
internal static class QuantizedGpuUpload
{
    public static bool IsSupported(DType dtype) => dtype is DType.Q3_K or DType.Q4_K or DType.Q5_K;

    /// <summary>Returns the raw-quantized tensor, or null when the caller should take its normal
    /// FP16/BF16/FP32 path (unsupported dtype or shape, backend without quantized Sgemm, or
    /// <paramref name="disableEnvVar"/> set to "1").</summary>
    public static unsafe CoreTensor? TryUpload(IComputeBackend backend, RawWeightSource? raw, string name,
        TensorShape shape, string disableEnvVar)
    {
        if (raw is null || !backend.SupportsQuantizedSgemm
            || Environment.GetEnvironmentVariable(disableEnvVar) == "1")
            return null;
        if (!raw(name, out nint data, out long len, out DType dt, out int rows, out int cols)
            || !IsSupported(dt) || rows != shape.Dims[0] || cols != shape.Dims[1] || cols % 256 != 0)
            return null;
        return backend.UploadRaw(new ReadOnlySpan<byte>((void*)data, checked((int)len)), shape, dt);
    }
}
