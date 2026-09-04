using System.Numerics.Tensors;
using CoreTensor = OpenTail.Stingray.Core.Tensor;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 flow-matching DiT (`MiniMaxMusic3Transformer1DModel`), transcribed directly
/// from the real, already-installed `diffusers==0.40.0` source
/// (`diffusers/models/transformers/transformer_minimax_music3.py`) -- see
/// docs/066-minimax-music3-future-plan.md.
///
/// <para><b>Status: code written and real tensor names cross-checked against the checkpoint's own
/// `diffusion_pytorch_model.safetensors.index.json` (a free, small file -- no real weight download
/// needed for this), but NOT YET real-weight-tested</b>: the two big real downloads
/// (`language_model/` ~17.2GB, `transformer/` ~9.7GB) are blocked on this session's current disk
/// space. Written now so it is ready to golden-verify immediately once space is available, matching
/// this session's established "write structure from real source, verify once weights land" pattern
/// (used for `StableAudioMediumDiT` before its checkpoint finished downloading).</para>
///
/// <para><b>Real, remarkably simple mechanism -- NO AdaLN</b> (unlike every other real DiT this
/// project has ported this session, all of which use adaptive-norm timestep conditioning): the
/// timestep embedding is PREPENDED as one extra sequence token before the transformer blocks and
/// STRIPPED OFF after them -- a real, plain "timestep-as-a-token" scheme. Standard `LayerNorm`
/// pre-norm blocks (not RMSNorm), full bidirectional self-attention (no causal mask, no windowing),
/// partial GPT-J-style RoPE (`rotary_dim=32` of `head_dim=64`), and a real GLU-style FF (`ff_in`
/// produces `2*ff_inner_dim`, split into `[gate_states, gate]`, `gate_states * silu(gate)`).</para>
///
/// <para><b>Real input assembly</b>: `concat([noisy_latent(128), zeros(128),
/// condition.transpose(128)])` along the channel axis (`2*in_channels + condition_dim = 2304`
/// channels total) -&gt; `preprocess_conv` (real `Conv1d(k=1)`, no bias) added as a RESIDUAL (`conv(x)
/// + x`, not just `conv(x)`) -&gt; `proj_in` (Linear, no bias) to the real inner dim (`2048`). Output
/// side mirrors this: `proj_out` (Linear, no bias) back to `in_channels`(128) -&gt; `postprocess_conv`
/// (`Conv1d(k=1)`, no bias) added as a residual again.</para>
/// </summary>
public sealed class MiniMaxMusic3TransformerWeights : IDisposable
{
    private MiniMaxMusic3GpuTransformerWeights? _gpuWeights;
    private readonly Lock _gpuLock = new();

    public required float[] TimeProjWeight { get; init; } // MiniMaxMusic3FourierEmbedding: [fourierDim/2, 1], no bias
    public required TimestepEmbeddingWeights TimeEmbed { get; init; }
    public required float[] PreprocessConvWeight { get; init; } // Conv1d k=1, no bias, [concatChannels, concatChannels, 1]
    public required float[] ProjInWeight { get; init; } // Linear [innerDim, concatChannels], no bias
    public required MiniMaxMusic3TransformerBlockWeights[] Blocks { get; init; } // 36 real layers
    public required float[] ProjOutWeight { get; init; } // Linear [inChannels, innerDim], no bias
    public required float[] PostprocessConvWeight { get; init; } // Conv1d k=1, no bias, [inChannels, inChannels, 1]

    public MiniMaxMusic3GpuTransformerWeights GetOrCreateGpuWeights(IComputeBackend backend)
    {
        if (_gpuWeights is not null) return _gpuWeights;
        lock (_gpuLock)
        {
            _gpuWeights ??= new MiniMaxMusic3GpuTransformerWeights(this, backend);
            return _gpuWeights;
        }
    }

    public void Dispose()
    {
        lock (_gpuLock)
        {
            _gpuWeights?.Dispose();
            _gpuWeights = null;
        }
    }

    public static MiniMaxMusic3TransformerWeights Load(SafetensorsLoader loader)
    {
        var blocks = new MiniMaxMusic3TransformerBlockWeights[MiniMaxMusic3Config.TransformerNumLayers];
        for (int i = 0; i < blocks.Length; i++)
            blocks[i] = LoadBlock(loader, $"transformer_blocks.{i}");

        return new MiniMaxMusic3TransformerWeights
        {
            TimeProjWeight = loader.ReadF32("time_proj.weight"),
            TimeEmbed = new TimestepEmbeddingWeights
            {
                Linear1Weight = loader.ReadF32("time_embed.linear_1.weight"),
                Linear1Bias = loader.ReadF32("time_embed.linear_1.bias"),
                Linear2Weight = loader.ReadF32("time_embed.linear_2.weight"),
                Linear2Bias = loader.ReadF32("time_embed.linear_2.bias"),
            },
            PreprocessConvWeight = loader.ReadF32("preprocess_conv.weight"),
            ProjInWeight = loader.ReadF32("proj_in.weight"),
            Blocks = blocks,
            ProjOutWeight = loader.ReadF32("proj_out.weight"),
            PostprocessConvWeight = loader.ReadF32("postprocess_conv.weight"),
        };
    }

    private static MiniMaxMusic3TransformerBlockWeights LoadBlock(SafetensorsLoader loader, string p)
    {
        return new MiniMaxMusic3TransformerBlockWeights
        {
            Norm1Weight = loader.ReadF32($"{p}.norm1.weight"),
            Norm1Bias = loader.ReadF32($"{p}.norm1.bias"),
            QWeight = loader.ReadF32($"{p}.attn.to_q.weight"),
            KWeight = loader.ReadF32($"{p}.attn.to_k.weight"),
            VWeight = loader.ReadF32($"{p}.attn.to_v.weight"),
            OWeight = loader.ReadF32($"{p}.attn.to_out.0.weight"),
            Norm2Weight = loader.ReadF32($"{p}.norm2.weight"),
            Norm2Bias = loader.ReadF32($"{p}.norm2.bias"),
            FfInWeight = loader.ReadF32($"{p}.ff_in.weight"), // [2*ffn, inner]
            FfInBias = loader.ReadF32($"{p}.ff_in.bias"),
            FfOutWeight = loader.ReadF32($"{p}.ff_out.weight"), // [inner, ffn]
            FfOutBias = loader.ReadF32($"{p}.ff_out.bias"),
        };
    }
}

public sealed class TimestepEmbeddingWeights
{
    public required float[] Linear1Weight { get; init; }
    public required float[] Linear1Bias { get; init; }
    public required float[] Linear2Weight { get; init; }
    public required float[] Linear2Bias { get; init; }
}

public sealed class MiniMaxMusic3TransformerBlockWeights
{
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] QWeight { get; init; }
    public required float[] KWeight { get; init; }
    public required float[] VWeight { get; init; }
    public required float[] OWeight { get; init; }
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
    public required float[] FfInWeight { get; init; }
    public required float[] FfInBias { get; init; }
    public required float[] FfOutWeight { get; init; }
    public required float[] FfOutBias { get; init; }
}

public static class MiniMaxMusic3Transformer
{
    /// <summary>Real forward: predicts the flow-matching velocity. `latent`/`condition` are
    /// time-major (`[length][channels]`); `condition` must already be resampled onto the latent
    /// timeline (real <see cref="MiniMaxMusic3ConditionEncoder"/> output). Pass an all-zero
    /// `condition` for the real unconditional CFG branch. Returns `[length][inChannels(128)]`.</summary>
    public static unsafe float[][] Forward(
        MiniMaxMusic3TransformerWeights w,
        float[][] latent,
        float[][] condition,
        float timestep,
        IComputeBackend? backend = null)
    {
        var gpu = backend is not null ? w.GetOrCreateGpuWeights(backend) : null;

        int inChannels = MiniMaxMusic3Config.TransformerInChannels; // 128
        int condDim = MiniMaxMusic3Config.TransformerConditionDim; // 2048
        int concatChannels = 2 * inChannels + condDim; // 2304
        int inner = MiniMaxMusic3Config.TransformerNumAttentionHeads * MiniMaxMusic3Config.TransformerAttentionHeadDim;
        int length = latent.Length;
        int seqLen = length + 1;
        int ffn = MiniMaxMusic3Config.TransformerFfInnerDim;
        int heads = MiniMaxMusic3Config.TransformerNumAttentionHeads;
        int headDim = MiniMaxMusic3Config.TransformerAttentionHeadDim;

        // Flatten latent and condition directly into flat concat buffer [length * concatChannels]
        var concat = new float[length * concatChannels];
        for (int t = 0; t < length; t++)
        {
            int baseIdx = t * concatChannels;
            Array.Copy(latent[t], 0, concat, baseIdx, inChannels);
            // zeros already default-initialized in [inChannels, 2*inChannels)
            Array.Copy(condition[t], 0, concat, baseIdx + 2 * inChannels, condDim);
        }

        // preprocess_conv (1x1) + residual
        var preNet = new float[length * concatChannels];
        fixed (float* cp = concat, pp = preNet, wpPre = w.PreprocessConvWeight)
        {
            MatMul(backend, gpu?.PreprocessConvWeight, pp, wpPre, cp, length, concatChannels, concatChannels);
            for (int i = 0; i < length * concatChannels; i++) pp[i] += cp[i];
        }

        var temb = TimestepEmbed(w, timestep);

        // proj_in: [length, concatChannels] -> [length, inner], prepend temb -> x [seqLen * inner]
        var x = new float[seqLen * inner];
        Array.Copy(temb, 0, x, 0, inner);
        fixed (float* xp = x, pp = preNet, wpIn = w.ProjInWeight)
        {
            MatMul(backend, gpu?.ProjInWeight, xp + inner, wpIn, pp, length, inner, concatChannels);
        }

        // Scratch buffers allocated ONCE for all 36 blocks:
        var normed1 = new float[seqLen * inner];
        var q = new float[seqLen * inner];
        var k = new float[seqLen * inner];
        var v = new float[seqLen * inner];
        var context = new float[seqLen * inner];
        var attnOut = new float[seqLen * inner];
        var normed2 = new float[seqLen * inner];
        var ffProj = new float[seqLen * 2 * ffn];
        var gated = new float[seqLen * ffn];
        var ffOut = new float[seqLen * inner];

        var (cos, sin) = BuildPartialRope(seqLen);

        fixed (float* xp = x, n1p = normed1, qp = q, kp = k, vp = v, ctxp = context,
                      aop = attnOut, n2p = normed2, fpp = ffProj, gp = gated, fop = ffOut)
        {
            for (int li = 0; li < w.Blocks.Length; li++)
            {
                var lw = w.Blocks[li];
                var bGpu = gpu?.Blocks[li];

                // 1. LayerNorm 1
                fixed (float* wNorm1 = lw.Norm1Weight, bNorm1 = lw.Norm1Bias)
                {
                    for (int t = 0; t < seqLen; t++)
                        LayerNormRow(xp + t * inner, wNorm1, bNorm1, n1p + t * inner, inner);
                }

                // 2. Q, K, V projections
                fixed (float* wQ = lw.QWeight, wK = lw.KWeight, wV = lw.VWeight)
                {
                    MatMulQkv(backend, bGpu?.QWeight, bGpu?.KWeight, bGpu?.VWeight,
                              qp, kp, vp, wQ, wK, wV, n1p, seqLen, inner);
                }

                // 3. Partial RoPE
                ApplyPartialRope(q, seqLen, heads, headDim, cos, sin);
                ApplyPartialRope(k, seqLen, heads, headDim, cos, sin);

                // 4. Attention
                float scale = 1f / MathF.Sqrt(headDim);
                Parallel.For(0, heads, h =>
                {
                    int off = h * headDim;
                    Span<float> scores = stackalloc float[seqLen];
                    for (int i = 0; i < seqLen; i++)
                    {
                        var qSpan = q.AsSpan(i * inner + off, headDim);
                        for (int j = 0; j < seqLen; j++)
                        {
                            var kSpan = k.AsSpan(j * inner + off, headDim);
                            scores[j] = TensorPrimitives.Dot(qSpan, kSpan) * scale;
                        }
                        SoftmaxRange(scores, 0, seqLen);

                        var ctxSpan = context.AsSpan(i * inner + off, headDim);
                        ctxSpan.Clear();
                        for (int j = 0; j < seqLen; j++)
                        {
                            var vSpan = v.AsSpan(j * inner + off, headDim);
                            TensorPrimitives.MultiplyAdd(vSpan, scores[j], ctxSpan, ctxSpan);
                        }
                    }
                });

                // 5. O projection
                fixed (float* wO = lw.OWeight)
                {
                    MatMul(backend, bGpu?.OWeight, aop, wO, ctxp, seqLen, inner, inner);
                }

                // 6. Residual add 1
                for (int i = 0; i < seqLen * inner; i++) xp[i] += aop[i];

                // 7. LayerNorm 2
                fixed (float* wNorm2 = lw.Norm2Weight, bNorm2 = lw.Norm2Bias)
                {
                    for (int t = 0; t < seqLen; t++)
                        LayerNormRow(xp + t * inner, wNorm2, bNorm2, n2p + t * inner, inner);
                }

                // 8. FeedForward: in -> silu*gate -> out
                fixed (float* wFfIn = lw.FfInWeight, bFfIn = lw.FfInBias,
                              wFfOut = lw.FfOutWeight, bFfOut = lw.FfOutBias)
                {
                    MatMul(backend, bGpu?.FfInWeight, fpp, wFfIn, n2p, seqLen, 2 * ffn, inner);
                    for (int t = 0; t < seqLen; t++)
                    {
                        int ppBase = t * 2 * ffn;
                        int gpBase = t * ffn;
                        for (int i = 0; i < ffn; i++)
                        {
                            float gateState = fpp[ppBase + i] + bFfIn[i];
                            float gateVal = fpp[ppBase + ffn + i] + bFfIn[ffn + i];
                            gp[gpBase + i] = gateState * Silu(gateVal);
                        }
                    }
                    MatMul(backend, bGpu?.FfOutWeight, fop, wFfOut, gp, seqLen, inner, ffn);
                    for (int t = 0; t < seqLen; t++)
                    {
                        int foBase = t * inner;
                        for (int i = 0; i < inner; i++)
                            xp[foBase + i] += fop[foBase + i] + bFfOut[i];
                    }
                }
            }
        }

        // Strip timestep token (rows 1..seqLen-1), proj_out, postprocess_conv + residual
        var stripped = new float[length * inner];
        Array.Copy(x, inner, stripped, 0, length * inner);

        var projOut = new float[length * inChannels];
        var postConv = new float[length * inChannels];
        fixed (float* sp = stripped, pop = projOut, wOut = w.ProjOutWeight, pcp = postConv, wPost = w.PostprocessConvWeight)
        {
            MatMul(backend, gpu?.ProjOutWeight, pop, wOut, sp, length, inChannels, inner);
            MatMul(backend, gpu?.PostprocessConvWeight, pcp, wPost, pop, length, inChannels, inChannels);
        }

        var output = new float[length][];
        for (int t = 0; t < length; t++)
        {
            var row = new float[inChannels];
            int baseIdx = t * inChannels;
            for (int c = 0; c < inChannels; c++) row[c] = postConv[baseIdx + c] + projOut[baseIdx + c];
            output[t] = row;
        }
        return output;
    }

    /// <summary>
    /// Co-evaluates conditional and unconditional CFG branches in a single pass (batch=2),
    /// streaming the 36-layer FP32 weights from memory ONCE instead of TWICE.
    /// </summary>
    public static unsafe (float[][] Cond, float[][] Uncond) ForwardPair(
        MiniMaxMusic3TransformerWeights w,
        float[][] latent,
        float[][] condition,
        float[][] zeroCondition,
        float timestep,
        IComputeBackend? backend = null)
    {
        var gpu = backend is not null ? w.GetOrCreateGpuWeights(backend) : null;

        int length = latent.Length;
        int seqLen = length + 1;
        int heads = MiniMaxMusic3Config.TransformerNumAttentionHeads;
        int headDim = MiniMaxMusic3Config.TransformerAttentionHeadDim;
        int inner = heads * headDim;
        int inChannels = MiniMaxMusic3Config.TransformerInChannels;
        int condDim = MiniMaxMusic3Config.TransformerConditionDim;
        int concatChannels = 2 * inChannels + condDim;
        int ffn = MiniMaxMusic3Config.TransformerFfInnerDim;
        int totalTokens = 2 * seqLen;

        // Flatten latent and conditions directly into flat concat buffer [2 * length * concatChannels]
        var concat = new float[2 * length * concatChannels];
        for (int t = 0; t < length; t++)
        {
            int base0 = t * concatChannels;
            Array.Copy(latent[t], 0, concat, base0, inChannels);
            Array.Copy(condition[t], 0, concat, base0 + 2 * inChannels, condDim);

            int base1 = (length + t) * concatChannels;
            Array.Copy(latent[t], 0, concat, base1, inChannels);
            Array.Copy(zeroCondition[t], 0, concat, base1 + 2 * inChannels, condDim);
        }

        // preprocess_conv (1x1) + residual for both sequences
        var preNet = new float[2 * length * concatChannels];
        fixed (float* cp = concat, pp = preNet, wpPre = w.PreprocessConvWeight)
        {
            MatMul(backend, gpu?.PreprocessConvWeight, pp, wpPre, cp, 2 * length, concatChannels, concatChannels);
            for (int i = 0; i < 2 * length * concatChannels; i++) pp[i] += cp[i];
        }

        var temb = TimestepEmbed(w, timestep);

        // proj_in: [2 * length, concatChannels] -> prepend temb to each sequence
        var x = new float[totalTokens * inner];
        Array.Copy(temb, 0, x, 0, inner);
        int seq1Base = seqLen * inner;
        Array.Copy(temb, 0, x, seq1Base, inner);

        fixed (float* xp = x, pp = preNet, wpIn = w.ProjInWeight)
        {
            MatMul(backend, gpu?.ProjInWeight, xp + inner, wpIn, pp, length, inner, concatChannels);
            MatMul(backend, gpu?.ProjInWeight, xp + seq1Base + inner, wpIn, pp + (long)length * concatChannels, length, inner, concatChannels);
        }

        // Scratch buffers allocated ONCE for both sequences across all 36 blocks:
        var normed1 = new float[totalTokens * inner];
        var q = new float[totalTokens * inner];
        var k = new float[totalTokens * inner];
        var v = new float[totalTokens * inner];
        var context = new float[totalTokens * inner];
        var attnOut = new float[totalTokens * inner];
        var normed2 = new float[totalTokens * inner];
        var ffProj = new float[totalTokens * 2 * ffn];
        var gated = new float[totalTokens * ffn];
        var ffOut = new float[totalTokens * inner];

        var (cos, sin) = BuildPartialRope(seqLen);

        fixed (float* xp = x, n1p = normed1, qp = q, kp = k, vp = v, ctxp = context,
                      aop = attnOut, n2p = normed2, fpp = ffProj, gp = gated, fop = ffOut)
        {
            for (int li = 0; li < w.Blocks.Length; li++)
            {
                var lw = w.Blocks[li];
                var bGpu = gpu?.Blocks[li];

                // 1. LayerNorm 1
                fixed (float* wNorm1 = lw.Norm1Weight, bNorm1 = lw.Norm1Bias)
                {
                    for (int t = 0; t < totalTokens; t++)
                        LayerNormRow(xp + t * inner, wNorm1, bNorm1, n1p + t * inner, inner);
                }

                // 2. Q, K, V projections
                fixed (float* wQ = lw.QWeight, wK = lw.KWeight, wV = lw.VWeight)
                {
                    MatMulQkv(backend, bGpu?.QWeight, bGpu?.KWeight, bGpu?.VWeight,
                              qp, kp, vp, wQ, wK, wV, n1p, totalTokens, inner);
                }

                // 3. Partial RoPE (applied per sequence)
                ApplyPartialRope(q.AsSpan(0, seqLen * inner), seqLen, heads, headDim, cos, sin);
                ApplyPartialRope(q.AsSpan(seq1Base, seqLen * inner), seqLen, heads, headDim, cos, sin);
                ApplyPartialRope(k.AsSpan(0, seqLen * inner), seqLen, heads, headDim, cos, sin);
                ApplyPartialRope(k.AsSpan(seq1Base, seqLen * inner), seqLen, heads, headDim, cos, sin);

                // 4. Attention (per sequence across all heads in parallel)
                float scale = 1f / MathF.Sqrt(headDim);
                Parallel.For(0, 2 * heads, bh =>
                {
                    int b = bh / heads;
                    int h = bh % heads;
                    int bOffset = b * seq1Base;
                    int headOff = h * headDim;
                    Span<float> scores = stackalloc float[seqLen];
                    for (int i = 0; i < seqLen; i++)
                    {
                        var qSpan = q.AsSpan(bOffset + i * inner + headOff, headDim);
                        for (int j = 0; j < seqLen; j++)
                        {
                            var kSpan = k.AsSpan(bOffset + j * inner + headOff, headDim);
                            scores[j] = TensorPrimitives.Dot(qSpan, kSpan) * scale;
                        }
                        SoftmaxRange(scores, 0, seqLen);

                        var ctxSpan = context.AsSpan(bOffset + i * inner + headOff, headDim);
                        ctxSpan.Clear();
                        for (int j = 0; j < seqLen; j++)
                        {
                            var vSpan = v.AsSpan(bOffset + j * inner + headOff, headDim);
                            TensorPrimitives.MultiplyAdd(vSpan, scores[j], ctxSpan, ctxSpan);
                        }
                    }
                });

                // 5. O projection
                fixed (float* wO = lw.OWeight)
                {
                    MatMul(backend, bGpu?.OWeight, aop, wO, ctxp, totalTokens, inner, inner);
                }

                // 6. Residual add 1
                for (int i = 0; i < totalTokens * inner; i++) xp[i] += aop[i];

                // 7. LayerNorm 2
                fixed (float* wNorm2 = lw.Norm2Weight, bNorm2 = lw.Norm2Bias)
                {
                    for (int t = 0; t < totalTokens; t++)
                        LayerNormRow(xp + t * inner, wNorm2, bNorm2, n2p + t * inner, inner);
                }

                // 8. FeedForward
                fixed (float* wFfIn = lw.FfInWeight, bFfIn = lw.FfInBias,
                              wFfOut = lw.FfOutWeight, bFfOut = lw.FfOutBias)
                {
                    MatMul(backend, bGpu?.FfInWeight, fpp, wFfIn, n2p, totalTokens, 2 * ffn, inner);
                    for (int t = 0; t < totalTokens; t++)
                    {
                        int ppBase = t * 2 * ffn;
                        int gpBase = t * ffn;
                        for (int i = 0; i < ffn; i++)
                        {
                            float gateState = fpp[ppBase + i] + bFfIn[i];
                            float gateVal = fpp[ppBase + ffn + i] + bFfIn[ffn + i];
                            gp[gpBase + i] = gateState * Silu(gateVal);
                        }
                    }
                    MatMul(backend, bGpu?.FfOutWeight, fop, wFfOut, gp, totalTokens, inner, ffn);
                    for (int t = 0; t < totalTokens; t++)
                    {
                        int foBase = t * inner;
                        for (int i = 0; i < inner; i++)
                            xp[foBase + i] += fop[foBase + i] + bFfOut[i];
                    }
                }
            }
        }

        // Strip timestep token (rows 1..seqLen-1) for each sequence
        var stripped = new float[2 * length * inner];
        Array.Copy(x, inner, stripped, 0, length * inner);
        Array.Copy(x, seq1Base + inner, stripped, length * inner, length * inner);

        var projOut = new float[2 * length * inChannels];
        var postConv = new float[2 * length * inChannels];
        fixed (float* sp = stripped, pop = projOut, wOut = w.ProjOutWeight, pcp = postConv, wPost = w.PostprocessConvWeight)
        {
            MatMul(backend, gpu?.ProjOutWeight, pop, wOut, sp, 2 * length, inChannels, inner);
            MatMul(backend, gpu?.PostprocessConvWeight, pcp, wPost, pop, 2 * length, inChannels, inChannels);
        }

        var vCond = new float[length][];
        var vUncond = new float[length][];
        for (int t = 0; t < length; t++)
        {
            var rCond = new float[inChannels];
            var rUncond = new float[inChannels];
            int idxCond = t * inChannels;
            int idxUncond = (length + t) * inChannels;
            for (int c = 0; c < inChannels; c++)
            {
                rCond[c] = postConv[idxCond + c] + projOut[idxCond + c];
                rUncond[c] = postConv[idxUncond + c] + projOut[idxUncond + c];
            }
            vCond[t] = rCond;
            vUncond[t] = rUncond;
        }

        return (vCond, vUncond);
    }

    private static unsafe float[] TimestepEmbed(MiniMaxMusic3TransformerWeights w, float timestep)
    {
        int fourierDim = MiniMaxMusic3Config.TransformerFourierEmbeddingDim;
        int half = fourierDim / 2;
        int inner = MiniMaxMusic3Config.TransformerNumAttentionHeads * MiniMaxMusic3Config.TransformerAttentionHeadDim;

        // Real MiniMaxMusic3FourierEmbedding: angles = 2*pi*t*weight; cat(cos, sin).
        var freq = new float[fourierDim];
        for (int i = 0; i < half; i++)
        {
            float angle = 2f * MathF.PI * timestep * w.TimeProjWeight[i];
            freq[i] = MathF.Cos(angle);
            freq[half + i] = MathF.Sin(angle);
        }

        var mid = new float[inner];
        fixed (float* fp = freq, mp = mid, w1 = w.TimeEmbed.Linear1Weight, b1 = w.TimeEmbed.Linear1Bias)
        {
            SimdKernels.MatMulBatchedF32(mp, w1, fp, 1, inner, fourierDim);
            for (int i = 0; i < inner; i++) mp[i] = Silu(mp[i] + b1[i]);
        }

        var output = new float[inner];
        fixed (float* mp = mid, op = output, w2 = w.TimeEmbed.Linear2Weight, b2 = w.TimeEmbed.Linear2Bias)
        {
            SimdKernels.MatMulBatchedF32(op, w2, mp, 1, inner, inner);
            for (int i = 0; i < inner; i++) op[i] += b2[i];
        }
        return output;
    }

    private static unsafe void LayerNormRow(float* x, float* weight, float* bias, float* output, int n)
    {
        float mean = 0f;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        float varSum = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; varSum += d * d; }
        float invStd = 1f / MathF.Sqrt(varSum / n + 1e-5f);
        for (int i = 0; i < n; i++) output[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
    }

    /// <summary>Real `MiniMaxMusic3RotaryEmbedding`: standard GPT-J-style partial rotary
    /// (`rotary_dim=32` of `headDim=64`), theta=10000, `freqs = cat([freqs,freqs])` (full-width
    /// cos/sin table duplicated across both halves of `rotary_dim`, standard HF convention).</summary>
    private static (float[] cos, float[] sin) BuildPartialRope(int seqLen)
    {
        int rotaryDim = MiniMaxMusic3Config.TransformerRotaryDim; // 32
        int half = rotaryDim / 2; // 16
        float theta = 10000f;
        var cos = new float[seqLen * rotaryDim];
        var sin = new float[seqLen * rotaryDim];
        for (int s = 0; s < seqLen; s++)
        {
            for (int i = 0; i < half; i++)
            {
                float invFreq = MathF.Pow(theta, -2f * i / rotaryDim);
                float angle = s * invFreq;
                float c = MathF.Cos(angle), sn = MathF.Sin(angle);
                cos[s * rotaryDim + i] = c; cos[s * rotaryDim + half + i] = c;
                sin[s * rotaryDim + i] = sn; sin[s * rotaryDim + half + i] = sn;
            }
        }
        return (cos, sin);
    }

    /// <summary>Real `_apply_partial_rotary_emb` (rotate_half convention over just the leading `rotaryDim` channels; the remaining `headDim-rotaryDim` channels pass through untouched).</summary>
    private static void ApplyPartialRope(Span<float> qOrK, int seqLen, int numHeads, int headDim, float[] cos, float[] sin)
    {
        int rotaryDim = MiniMaxMusic3Config.TransformerRotaryDim;
        int half = rotaryDim / 2;
        int rowDim = numHeads * headDim;
        for (int t = 0; t < seqLen; t++)
        {
            int cosBase = t * rotaryDim;
            for (int h = 0; h < numHeads; h++)
            {
                int off = t * rowDim + h * headDim;
                for (int i = 0; i < half; i++)
                {
                    float x1 = qOrK[off + i];
                    float x2 = qOrK[off + half + i];
                    float c1 = cos[cosBase + i], s1 = sin[cosBase + i];
                    float c2 = cos[cosBase + half + i], s2 = sin[cosBase + half + i];
                    qOrK[off + i] = x1 * c1 - x2 * s1;
                    qOrK[off + half + i] = x2 * c2 + x1 * s2;
                }
                // channels [rotaryDim, headDim) left untouched -- real partial-rotary behavior.
            }
        }
    }

    private static void SoftmaxRange(Span<float> scores, int start, int end)
    {
        float max = float.NegativeInfinity;
        for (int i = start; i < end; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = start; i < end; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = start; i < end; i++) scores[i] *= invSum;
    }

    private static unsafe void MatMul(
        IComputeBackend? backend,
        CoreTensor? gpuWeight,
        float* output,
        float* weights,
        float* input,
        int batchSize,
        int rows,
        int cols)
    {
        if (backend is not null && gpuWeight is not null)
        {
            int inCount = batchSize * cols;
            int outCount = batchSize * rows;
            var xGpu = backend.Upload(new ReadOnlySpan<float>(input, inCount), TensorShape.D1(inCount));
            var cGpu = backend.Allocate(TensorShape.D1(outCount));
            try
            {
                backend.Sgemm(cGpu, xGpu, gpuWeight, batchSize, cols, rows);
                backend.Synchronize();
                backend.Download(cGpu, new Span<float>(output, outCount));
                return;
            }
            finally
            {
                backend.Free(xGpu);
                backend.Free(cGpu);
            }
        }

        MatMulRowMajor(output, weights, input, batchSize, rows, cols);
    }

    private static unsafe void MatMulQkv(
        IComputeBackend? backend,
        CoreTensor? wQGpu, CoreTensor? wKGpu, CoreTensor? wVGpu,
        float* qOut, float* kOut, float* vOut,
        float* wQ, float* wK, float* wV,
        float* input,
        int batchSize,
        int dim)
    {
        if (backend is not null && wQGpu is not null && wKGpu is not null && wVGpu is not null)
        {
            int count = batchSize * dim;
            var xGpu = backend.Upload(new ReadOnlySpan<float>(input, count), TensorShape.D1(count));
            var cGpu = backend.Allocate(TensorShape.D1(count));
            try
            {
                backend.Sgemm(cGpu, xGpu, wQGpu, batchSize, dim, dim);
                backend.Synchronize();
                backend.Download(cGpu, new Span<float>(qOut, count));

                backend.Sgemm(cGpu, xGpu, wKGpu, batchSize, dim, dim);
                backend.Synchronize();
                backend.Download(cGpu, new Span<float>(kOut, count));

                backend.Sgemm(cGpu, xGpu, wVGpu, batchSize, dim, dim);
                backend.Synchronize();
                backend.Download(cGpu, new Span<float>(vOut, count));
                return;
            }
            finally
            {
                backend.Free(xGpu);
                backend.Free(cGpu);
            }
        }

        MatMulRowMajor(qOut, wQ, input, batchSize, dim, dim);
        MatMulRowMajor(kOut, wK, input, batchSize, dim, dim);
        MatMulRowMajor(vOut, wV, input, batchSize, dim, dim);
    }

    private static unsafe void MatMulRowMajor(float* output, float* weights, float* input,
        int batchSize, int rows, int cols)
    {
        if (batchSize <= 1)
        {
            if (batchSize == 1) SimdKernels.MatVecF32(output, weights, input, rows, cols);
            return;
        }

        int numThreads = Math.Min(Environment.ProcessorCount, (rows + 63) / 64);
        if (numThreads <= 1)
        {
            int r = 0;
            for (; r + 4 <= rows; r += 4)
            {
                float* m0 = weights + (long)r * cols;
                float* m1 = weights + (long)(r + 1) * cols;
                float* m2 = weights + (long)(r + 2) * cols;
                float* m3 = weights + (long)(r + 3) * cols;
                for (int n = 0; n < batchSize; n++)
                {
                    float* x = input + (long)n * cols;
                    SimdKernels.MatVecF32_4Row(m0, m1, m2, m3, x, cols, out float r0, out float r1, out float r2, out float r3);
                    long baseIdx = (long)n * rows + r;
                    output[baseIdx] = r0;
                    output[baseIdx + 1] = r1;
                    output[baseIdx + 2] = r2;
                    output[baseIdx + 3] = r3;
                }
            }
            for (; r < rows; r++)
            {
                float* wRow = weights + (long)r * cols;
                for (int n = 0; n < batchSize; n++)
                    output[(long)n * rows + r] = SimdKernels.DotF32(wRow, input + (long)n * cols, cols);
            }
            return;
        }

        int chunkSize = ((rows + numThreads - 1) / numThreads + 3) & ~3;
        nint outAddr = (nint)output;
        nint wAddr = (nint)weights;
        nint inAddr = (nint)input;

        Parallel.For(0, numThreads, t =>
        {
            float* outp = (float*)outAddr;
            float* wp = (float*)wAddr;
            float* inp = (float*)inAddr;

            int start = t * chunkSize;
            int end = Math.Min(rows, start + chunkSize);
            int r = start;
            for (; r + 4 <= end; r += 4)
            {
                float* m0 = wp + (long)r * cols;
                float* m1 = wp + (long)(r + 1) * cols;
                float* m2 = wp + (long)(r + 2) * cols;
                float* m3 = wp + (long)(r + 3) * cols;
                for (int n = 0; n < batchSize; n++)
                {
                    float* x = inp + (long)n * cols;
                    SimdKernels.MatVecF32_4Row(m0, m1, m2, m3, x, cols, out float r0, out float r1, out float r2, out float r3);
                    long baseIdx = (long)n * rows + r;
                    outp[baseIdx] = r0;
                    outp[baseIdx + 1] = r1;
                    outp[baseIdx + 2] = r2;
                    outp[baseIdx + 3] = r3;
                }
            }
            for (; r < end; r++)
            {
                float* wRow = wp + (long)r * cols;
                for (int n = 0; n < batchSize; n++)
                    outp[(long)n * rows + r] = SimdKernels.DotF32(wRow, inp + (long)n * cols, cols);
            }
        });
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));
}
