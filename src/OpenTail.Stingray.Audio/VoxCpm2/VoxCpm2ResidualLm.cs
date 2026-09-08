using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Native C# port of VoxCPM2's "residual LM" -- a real, SEPARATE small (8-layer) MiniCPM-style
/// CAUSAL autoregressive decoder, from `minicpm.cpp`'s `residual_lm_config`/`minicpm_layer`
/// (`is_causal=true` for this model, unlike the local encoder/DiT decoder's bidirectional use of
/// the same per-layer math) and `minicpm_blocks.h` (not guessed). Real config: same
/// `hidden_size=2048`/`num_attention_heads=16`/`num_key_value_heads=2`/`head_dim=128`/
/// `intermediate_size=6144` as `base_lm` (inherited via `residual_lm_config`), but only 8 layers
/// and `no_rope=true` (RoPE completely disabled -- confirmed via `apply_minicpm_rope`'s real
/// `if (config.no_rope) return input;` early-out). No token embedding or LM head at all (real
/// tensor dump confirms neither `embed_tokens` nor `lm_head` exist for `residual_lm` -- this
/// model NEVER does a vocabulary lookup or computes logits, it only ever consumes precomputed
/// embeddings and returns hidden states, exactly like `base_lm`'s own per-step
/// `ForwardEmbedding` calls -- but unlike `base_lm`, `ForwardPass`'s generic `minicpm`-
/// architecture graph cannot be reused here because it has no way to disable RoPE per
/// checkpoint, so this is a genuinely bespoke, from-scratch causal decoder with its own real
/// persistent per-step KV cache (`use_mup=false` inherited too, so no residual/embedding
/// scaling anywhere, same as `base_lm`).
/// </summary>
public sealed class VoxCpm2ResidualLm
{
    public const int HiddenDim = 2048;
    public const int NumHeads = 16;
    public const int NumKvHeads = 2;
    public const int HeadDim = 128;
    public const int FfnDim = 6144;
    public const int NumLayers = 8;
    public const float RmsNormEps = 1e-5f;

    private readonly VoxCpm2MiniCpmLayerWeights[] _layers;
    private readonly float[] _finalNorm;
    private readonly List<float[]>[] _keyCache;
    private readonly List<float[]>[] _valueCache;
    private float[] _scoresBuf = new float[128];

    public VoxCpm2ResidualLm(VoxCpm2MiniCpmLayerWeights[] layers, float[] finalNorm)
    {
        if (layers.Length != NumLayers) throw new ArgumentException($"Expected {NumLayers} layers.", nameof(layers));
        _layers = layers;
        _finalNorm = finalNorm;
        _keyCache = new List<float[]>[NumLayers];
        _valueCache = new List<float[]>[NumLayers];
        Reset();
    }

    public static VoxCpm2ResidualLm Load(Func<string, float[]> get)
    {
        var layers = new VoxCpm2MiniCpmLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"weights/residual_lm.layers.{i}.";
            layers[i] = new VoxCpm2MiniCpmLayerWeights
            {
                InputNorm = get(p + "input_layernorm.weight"),
                QProjWeight = get(p + "self_attn.q_proj.weight"),
                KProjWeight = get(p + "self_attn.k_proj.weight"),
                VProjWeight = get(p + "self_attn.v_proj.weight"),
                OProjWeight = get(p + "self_attn.o_proj.weight"),
                PostNorm = get(p + "post_attention_layernorm.weight"),
                GateProjWeight = get(p + "mlp.gate_proj.weight"),
                UpProjWeight = get(p + "mlp.up_proj.weight"),
                DownProjWeight = get(p + "mlp.down_proj.weight"),
            };
        }
        var finalNorm = get("weights/residual_lm.norm.weight");
        return new VoxCpm2ResidualLm(layers, finalNorm);
    }

    /// <summary>Clears the KV cache (real `.reset()` before a fresh generation).</summary>
    public void Reset()
    {
        for (int l = 0; l < NumLayers; l++)
        {
            _keyCache[l] = [];
            _valueCache[l] = [];
        }
    }

    /// <summary>Appends one embedding as the next causal position and returns the post-final-norm
    /// hidden state at that position (real `run_step(...).hidden`). No RoPE applied anywhere
    /// (`no_rope=true` for this model).</summary>
    public unsafe float[] Step(float[] embedding)
    {
        if (embedding.Length != HiddenDim) throw new ArgumentException($"Expected length {HiddenDim}.", nameof(embedding));

        int kvRepeats = NumHeads / NumKvHeads;
        float scale = 1f / MathF.Sqrt(HeadDim);
        int qOut = NumHeads * HeadDim;
        int kvOut = NumKvHeads * HeadDim;

        var hidden = (float[])embedding.Clone();
        var normed = new float[HiddenDim];
        var q = new float[qOut];
        var context = new float[qOut];
        var attnOut = new float[HiddenDim];
        var x = new float[HiddenDim];
        var ffnNormed = new float[HiddenDim];
        var gate = new float[FfnDim];
        var up = new float[FfnDim];
        var down = new float[HiddenDim];
        var output = new float[HiddenDim];

        fixed (float* hiddenPtr = hidden, normedPtr = normed,
               qPtr = q, contextPtr = context, attnOutPtr = attnOut,
               xPtr = x, ffnNormedPtr = ffnNormed, gatePtr = gate, upPtr = up, downPtr = down,
               finalNormPtr = _finalNorm, outputPtr = output)
        {
            for (int l = 0; l < NumLayers; l++)
            {
                var layer = _layers[l];
                var k = new float[kvOut];
                var v = new float[kvOut];

                fixed (float* inNormPtr = layer.InputNorm,
                       qWPtr = layer.QProjWeight, kWPtr = layer.KProjWeight, vWPtr = layer.VProjWeight,
                       oWPtr = layer.OProjWeight, postNormPtr = layer.PostNorm,
                       gateWPtr = layer.GateProjWeight, upWPtr = layer.UpProjWeight, downWPtr = layer.DownProjWeight,
                       kPtr = k, vPtr = v)
                {
                    // 1. Input Norm
                    SimdKernels.RmsNorm(normedPtr, hiddenPtr, inNormPtr, HiddenDim, RmsNormEps);

                    // 2. Q, K, V Projections
                    SimdKernels.MatVecF32(qPtr, qWPtr, null, normedPtr, qOut, HiddenDim);
                    SimdKernels.MatVecF32(kPtr, kWPtr, null, normedPtr, kvOut, HiddenDim);
                    SimdKernels.MatVecF32(vPtr, vWPtr, null, normedPtr, kvOut, HiddenDim);

                    _keyCache[l].Add(k);
                    _valueCache[l].Add(v);
                    int availableKeys = _keyCache[l].Count;

                    if (_scoresBuf.Length < availableKeys)
                        _scoresBuf = new float[Math.Max(availableKeys, _scoresBuf.Length * 2)];
                    var scores = _scoresBuf.AsSpan(0, availableKeys);

                    // 3. Attention
                    new Span<float>(contextPtr, qOut).Clear();
                    for (int h = 0; h < NumHeads; h++)
                    {
                        int hOff = h * HeadDim;
                        int kvHOff = (h / kvRepeats) * HeadDim;
                        float* qHead = qPtr + hOff;

                        for (int t = 0; t < availableKeys; t++)
                        {
                            fixed (float* ktPtr = _keyCache[l][t])
                            {
                                scores[t] = SimdKernels.DotF32(qHead, ktPtr + kvHOff, HeadDim) * scale;
                            }
                        }

                        DenseKernels.SoftmaxInPlace(scores);

                        float* ctxHead = contextPtr + hOff;
                        for (int t = 0; t < availableKeys; t++)
                        {
                            float p = scores[t];
                            fixed (float* vtPtr = _valueCache[l][t])
                            {
                                float* vHead = vtPtr + kvHOff;
                                if (Avx2.IsSupported && Fma.IsSupported)
                                {
                                    var vp = Vector256.Create(p);
                                    for (int d = 0; d < HeadDim; d += 8)
                                    {
                                        var vc = Avx.LoadVector256(ctxHead + d);
                                        var vv = Avx.LoadVector256(vHead + d);
                                        vc = Fma.MultiplyAdd(vp, vv, vc);
                                        Avx.Store(ctxHead + d, vc);
                                    }
                                }
                                else
                                {
                                    for (int d = 0; d < HeadDim; d++)
                                        ctxHead[d] += p * vHead[d];
                                }
                            }
                        }
                    }

                    // 4. O Projection
                    SimdKernels.MatVecF32(attnOutPtr, oWPtr, null, contextPtr, HiddenDim, qOut);

                    // 5. Residual Add
                    TensorPrimitives.Add(new ReadOnlySpan<float>(hiddenPtr, HiddenDim),
                                         new ReadOnlySpan<float>(attnOutPtr, HiddenDim),
                                         new Span<float>(xPtr, HiddenDim));

                    // 6. Post Norm
                    SimdKernels.RmsNorm(ffnNormedPtr, xPtr, postNormPtr, HiddenDim, RmsNormEps);

                    // 7. Gate & Up Projections
                    SimdKernels.MatVecF32(gatePtr, gateWPtr, null, ffnNormedPtr, FfnDim, HiddenDim);
                    SimdKernels.MatVecF32(upPtr, upWPtr, null, ffnNormedPtr, FfnDim, HiddenDim);

                    // 8. Fused SiLU(gate) * up
                    SimdKernels.SiLuMul(gatePtr, upPtr, FfnDim);

                    // 9. Down Projection
                    SimdKernels.MatVecF32(downPtr, downWPtr, null, gatePtr, HiddenDim, FfnDim);

                    // 10. Residual Add
                    TensorPrimitives.Add(new ReadOnlySpan<float>(xPtr, HiddenDim),
                                         new ReadOnlySpan<float>(downPtr, HiddenDim),
                                         new Span<float>(hiddenPtr, HiddenDim));
                }
            }

            // Final RmsNorm
            SimdKernels.RmsNorm(outputPtr, hiddenPtr, finalNormPtr, HiddenDim, RmsNormEps);
        }

        return output;
    }
}

