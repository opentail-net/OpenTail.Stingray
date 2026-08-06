namespace OpenTail.Stingray.Cuda;

/// <summary>
/// Domain-focused partial class holding CUDA Gated-DeltaNet (GDN) recurrent state kernels.
/// </summary>
internal static partial class CudaTextKernels
{
    public const string GdnKernelsSource = @"
// Gated-DeltaNet state kernels and device helpers
__device__ __forceinline__ float opentail_llm_gdn_gate(float state, float gate)
{
    return state * gate;
}
";
}
