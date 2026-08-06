namespace OpenTail.Stingray.Cuda;

/// <summary>
/// Domain-focused partial class holding CUDA attention kernel sources (GQA, SWA, FlashAttention).
/// </summary>
internal static partial class CudaTextKernels
{
    public const string AttentionKernelsSource = @"
// Attention kernel helpers and device functions
__device__ __forceinline__ float opentail_llm_attn_scale(float val, float scale)
{
    return val * scale;
}
";
}
