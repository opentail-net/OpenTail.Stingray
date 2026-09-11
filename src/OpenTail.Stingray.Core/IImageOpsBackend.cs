namespace OpenTail.Stingray.Core;

/// <summary>
/// Extended compute-backend interface for image-processing operations (conv2d, activations,
/// spatial rearrangements). Implemented by backends that can run a full convolutional forward
/// pass without CPU round-trips.
///
/// Inherits from <see cref="IComputeBackend"/> so callers receive Upload/Download/Free/AddInPlace
/// from the same object.
///
/// Tensor layout convention: all spatial tensors are CHW (channels-first), flat float32.
/// </summary>
public interface IImageOpsBackend : IComputeBackend
{
    /// <summary>
    /// 2D convolution (stride=1, same padding by default).
    /// input  [inCh, H, W], weight [outCh, inCh, k, k], bias [outCh]
    /// → output [outCh, H, W]
    /// </summary>
    Tensor Conv2d(Tensor input, Tensor weight, Tensor bias,
                  int inCh, int outCh, int h, int w, int ksize, int padding = -1);

    /// <summary>
    /// 2D convolution via a tiled "implicit GEMM" kernel (im2col computed on the fly inside the
    /// kernel, GEMM-style shared-memory blocking) instead of <see cref="Conv2d"/>'s naive
    /// one-thread-per-pixel accumulation. Same input/weight/bias/output layout and semantics as
    /// <see cref="Conv2d"/> -- combines GEMM-style compute efficiency with zero CPU-side im2col.
    /// </summary>
    Tensor Conv2dImplicitGemm(Tensor input, Tensor weight, Tensor bias,
                              int inCh, int outCh, int h, int w, int ksize, int padding = -1);

    /// <summary>
    /// Fused GroupNorm + SiLU for a GPU-resident [C,H,W] tensor -- input/weight/bias/output all
    /// stay on-device, no CPU round-trip. Exists specifically so a ResBlock's conv→norm→silu→conv
    /// chain can run entirely GPU-resident (see VaeDecoder's ResBlockGpu) instead of downloading
    /// to CPU between every op purely to run GroupNorm/SiLU there.
    /// </summary>
    Tensor GroupNormSilu(Tensor x, Tensor weight, Tensor bias, int c, int hw, int groups = 32, float eps = 1e-5f);

    /// <summary>
    /// In-place per-channel scalar broadcast-add: x[c,h,w] += bias[c] for every spatial position.
    /// Added for the SDXL UNet GPU-residency rewrite (docs/067) -- ResBlock's timestep-embedding
    /// injection needs this and no existing op covers it (not same-shape, not GroupNorm/SiLU).
    /// </summary>
    void AddChannelBroadcastInPlace(Tensor x, Tensor perChannelBias, int c, int hw);

    /// <summary>
    /// Multi-head scaled-dot-product attention (bidirectional, no causal mask, no KV cache) for
    /// vision-transformer-shaped self/cross attention. Q [qSeq, numHeads*headDim], K/V
    /// [kvSeq, numHeads*headDim] -> output [qSeq, numHeads*headDim]. Math matches
    /// DiffusionOps.MultiHeadAttention exactly (scale=1/sqrt(headDim), stable softmax, no mask).
    /// headDim must be &lt;=128 (fixed-size shader accumulator; every real caller uses 64).
    /// </summary>
    Tensor MultiHeadAttention(Tensor q, Tensor k, Tensor v, int qSeq, int kvSeq, int numHeads, int headDim);

    /// <summary>
    /// Tiled ("flash-attention"-style) variant of <see cref="MultiHeadAttention"/> -- same math,
    /// same input/output layout, replaces the naive shader's one-thread-per-(query,head) design
    /// (which regressed performance, see PerformanceLeague.md) with row/column tiling and shared
    /// memory reuse of K/V. Fixed headDim=64 only (this codebase's only real usage). Not
    /// necessarily bit-exact with the CPU reference (tiled floating-point accumulation order) --
    /// verify against a tolerance, not exact equality.
    /// </summary>
    Tensor MultiHeadAttentionTiled(Tensor q, Tensor k, Tensor v, int qSeq, int kvSeq, int numHeads, int headDim);

    /// <summary>LeakyReLU in-place: x[i] = x[i] >= 0 ? x[i] : negSlope * x[i]</summary>
    void LeakyReluInPlace(Tensor x, float negSlope);

    /// <summary>Scale in-place: x[i] *= scale</summary>
    void ScaleInPlace(Tensor x, float scale);

    /// <summary>Scaled add in-place: dst[i] += src[i] * scale</summary>
    void AddScaledInPlace(Tensor dst, Tensor src, float scale);

    /// <summary>Clamp in-place: x[i] = clamp(x[i], min, max)</summary>
    void ClampInPlace(Tensor x, float min, float max);

    /// <summary>
    /// Channel concatenation along C axis.
    /// a [aCh, hw], b [bCh, hw] → output [(aCh+bCh), hw]
    /// </summary>
    Tensor CatChannels(Tensor a, int aCh, Tensor b, int bCh, int hw);

    /// <summary>
    /// Pixel shuffle: [inCh, H, W] → [inCh/r², H*r, W*r]
    /// where r = upscaleFactor.
    /// </summary>
    Tensor PixelShuffleGpu(Tensor input, int inCh, int h, int w, int upscaleFactor);

    /// <summary>
    /// Pixel unshuffle (inverse of pixel shuffle): [inCh, H*r, W*r] → [inCh*r², H, W]
    /// where r = downscaleFactor.
    /// </summary>
    Tensor PixelUnshuffleGpu(Tensor input, int inCh, int h, int w, int downscaleFactor);

    /// <summary>
    /// Nearest-neighbor 2× upsample: [ch, H, W] → [ch, 2H, 2W]
    /// </summary>
    Tensor Upsample2xGpu(Tensor input, int ch, int h, int w);

    // ── Batch recording ──────────────────────────────────────────────────────
    // BeginBatch/EndBatch enable dispatching the entire forward pass as a single
    // GPU command buffer submission, eliminating per-dispatch queue-submit overhead.
    // Free() calls between Begin/EndBatch are automatically deferred until after submit.

    /// <summary>Begin recording: subsequent dispatches are batched into one submission.</summary>
    void BeginBatch();

    /// <summary>
    /// Insert a compute→compute memory barrier (all prior writes visible to subsequent reads).
    /// Required between dispatches that have data dependencies in batch recording mode.
    /// No-op when not batching.
    /// </summary>
    void BatchBarrier();

    /// <summary>End batch: submit all recorded dispatches, wait for completion, process deferred frees.</summary>
    void EndBatch();
}
