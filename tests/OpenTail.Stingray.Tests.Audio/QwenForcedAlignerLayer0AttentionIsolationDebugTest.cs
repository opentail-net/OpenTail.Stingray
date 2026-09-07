namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric isolation test for layer 0's full attention+MLP block, extending the same
/// methodology that proved layer 2's MLP formula correct (see
/// <see cref="QwenForcedAlignerLayer2MlpIsolationDebugTest"/> and
/// docs/audio-review-progress.md's "CONCLUSIVE" / "MAJOR CORRECTION" entries, 2026-09-07):
/// hand-computes layer 0's REAL attention (RMSNorm -> Q/K/V proj -> per-head Q/K RMSNorm -> NEOX
/// RoPE -> GQA causal softmax attention -> o_proj -> residual) and MLP (SwiGLU) directly from the
/// real checkpoint's raw Safetensors weights, fed this port's OWN real input embedding matrix
/// (dumped via <see cref="Audio.QwenASR.QwenAsrForcedAligner.AlignReal"/>'s `STINGRAY_FA_DUMP_DIR`
/// hook, `our_input_embeddings.bin` -- the SAME combined embedding table `ForwardPass` uses
/// internally, already established to be internally consistent), across ALL 130 real prompt
/// positions -- bypassing `ForwardPass`/`ModelGraph` entirely. Compared against the reference's
/// real `layer_0_out.bin` (dumped per-position, trustworthy per the same per-layer-loop-capture
/// mechanism already cross-validated for layer 2).
///
/// If this hand computation matches the reference closely: this port's UNDERSTANDING of layer 0's
/// real formula is correct, and the actual bug is in HOW `ForwardPass`/`ModelGraph`'s generic
/// "qwen3" architecture graph executes it (an engine-level bug, not a math misunderstanding). If
/// it diverges: the divergence is either in this port's own INPUT embeddings (audio-token splicing
/// specifically, the one thing this test does not independently re-derive) or in this hand-coded
/// formula itself (which would also indict `ForwardPass`'s identical shared formula).
/// </summary>
public sealed class QwenForcedAlignerLayer0AttentionIsolationDebugTest : HeavyTestBase
{
    private const string DumpDir = "C:/Users/Dmitri/AppData/Local/Temp/claude/C--Git-Public/e20d4458-3906-46e3-a5d1-af57aa8bcc0d/scratchpad/fa_dump";

    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadRawF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    private static float[] RmsNorm(ReadOnlySpan<float> x, float[] weight, float eps)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + eps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * invRms * weight[i];
        return output;
    }

    private static float[] Linear(ReadOnlySpan<float> input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = 0f;
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }

    /// <summary>Real standard NEOX RoPE: pairs (i, i+half) rotated, freq_i = theta^(-2i/dim).</summary>
    private static void ApplyNeoxRope(float[] vec, int headDim, int position, float theta)
    {
        int half = headDim / 2;
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Pow(theta, -2.0 * i / headDim);
            double angle = position * freq;
            float cos = (float)Math.Cos(angle), sin = (float)Math.Sin(angle);
            float a = vec[i], b = vec[i + half];
            vec[i] = a * cos - b * sin;
            vec[i + half] = a * sin + b * cos;
        }
    }

    [Fact]
    public void HandComputedLayer0_OnRealInput_ComparedAgainstRealReferenceOutput()
    {
        string inputPath = Path.Combine(DumpDir, "our_input_embeddings.bin");
        string outPath = Path.Combine(DumpDir, "layer_0_out.bin");
        Assert.SkipUnless(File.Exists(inputPath) && File.Exists(outPath), "real dumps not found -- see class doc comment for how to regenerate");

        string? safetensorsPath = FindRepoFile("models/qwen3-forcedaligner/model.safetensors");
        Assert.SkipUnless(safetensorsPath != null, "models/qwen3-forcedaligner/model.safetensors not found");

        const int hiddenDim = 1024;
        const int numHeads = 16, numKvHeads = 8, headDim = 128;
        const int qDim = numHeads * headDim, kvDim = numKvHeads * headDim;
        const float eps = 1e-6f;
        const float ropeTheta = 1_000_000f;
        const int kvRepeats = numHeads / numKvHeads;
        float scale = 1f / MathF.Sqrt(headDim);

        var input = ReadRawF32(inputPath);
        var realOut = ReadRawF32(outPath);
        Assert.Equal(input.Length, realOut.Length);
        int steps = input.Length / hiddenDim;

        using var loader = SafetensorsLoader.Open(safetensorsPath!);
        const string p = "thinker.model.layers.0.";
        var inputNorm = loader.ReadF32(p + "input_layernorm.weight");
        var qProj = loader.ReadF32(p + "self_attn.q_proj.weight");
        var kProj = loader.ReadF32(p + "self_attn.k_proj.weight");
        var vProj = loader.ReadF32(p + "self_attn.v_proj.weight");
        var oProj = loader.ReadF32(p + "self_attn.o_proj.weight");
        var qNorm = loader.ReadF32(p + "self_attn.q_norm.weight");
        var kNorm = loader.ReadF32(p + "self_attn.k_norm.weight");
        var postNorm = loader.ReadF32(p + "post_attention_layernorm.weight");
        var gateWeight = loader.ReadF32(p + "mlp.gate_proj.weight");
        var upWeight = loader.ReadF32(p + "mlp.up_proj.weight");
        var downWeight = loader.ReadF32(p + "mlp.down_proj.weight");
        int ffDim = loader.GetShape(p + "mlp.gate_proj.weight")[0];

        // Real per-position Q/K/V, post-norm, post-RoPE (real NEOX, per QwenDecoderLayerConfig's
        // hardcoded default confirmed earlier this session).
        var allQ = new float[steps][];
        var allK = new float[steps][];
        var allV = new float[steps][];
        for (int t = 0; t < steps; t++)
        {
            var x = input.AsSpan(t * hiddenDim, hiddenDim);
            var normed = RmsNorm(x, inputNorm, eps);
            var q = Linear(normed, qProj, hiddenDim, qDim);
            var k = Linear(normed, kProj, hiddenDim, kvDim);
            var v = Linear(normed, vProj, hiddenDim, kvDim);

            for (int h = 0; h < numHeads; h++)
            {
                var qh = RmsNorm(q.AsSpan(h * headDim, headDim), qNorm, eps);
                ApplyNeoxRope(qh, headDim, t, ropeTheta);
                Array.Copy(qh, 0, q, h * headDim, headDim);
            }
            for (int h = 0; h < numKvHeads; h++)
            {
                var kh = RmsNorm(k.AsSpan(h * headDim, headDim), kNorm, eps);
                ApplyNeoxRope(kh, headDim, t, ropeTheta);
                Array.Copy(kh, 0, k, h * headDim, headDim);
            }
            allQ[t] = q; allK[t] = k; allV[t] = v;
        }

        var handOut = new float[input.Length];
        for (int t = 0; t < steps; t++)
        {
            var x = input.AsSpan(t * hiddenDim, hiddenDim);
            var context = new float[qDim];
            for (int h = 0; h < numHeads; h++)
            {
                int kvH = h / kvRepeats;
                var scores = new float[t + 1];
                for (int s = 0; s <= t; s++)
                {
                    float dot = 0f;
                    var qh = allQ[t].AsSpan(h * headDim, headDim);
                    var kh = allK[s].AsSpan(kvH * headDim, headDim);
                    for (int d = 0; d < headDim; d++) dot += qh[d] * kh[d];
                    scores[s] = dot * scale;
                }
                float max = float.NegativeInfinity;
                foreach (var sc in scores) if (sc > max) max = sc;
                float sum = 0f;
                for (int s = 0; s <= t; s++) { scores[s] = MathF.Exp(scores[s] - max); sum += scores[s]; }
                for (int s = 0; s <= t; s++) scores[s] /= sum;
                for (int s = 0; s <= t; s++)
                {
                    var vh = allV[s].AsSpan(kvH * headDim, headDim);
                    for (int d = 0; d < headDim; d++) context[h * headDim + d] += scores[s] * vh[d];
                }
            }
            var attnOut = Linear(context, oProj, qDim, hiddenDim);
            var postAttn = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) postAttn[i] = x[i] + attnOut[i];

            var normed2 = RmsNorm(postAttn, postNorm, eps);
            var gate = new float[ffDim];
            var up = new float[ffDim];
            for (int o = 0; o < ffDim; o++)
            {
                float g = 0f, u = 0f;
                int wBase = o * hiddenDim;
                for (int i = 0; i < hiddenDim; i++) { g += gateWeight[wBase + i] * normed2[i]; u += upWeight[wBase + i] * normed2[i]; }
                gate[o] = (g / (1f + MathF.Exp(-g))) * u;
            }
            var down = Linear(gate, downWeight, ffDim, hiddenDim);
            for (int i = 0; i < hiddenDim; i++) handOut[t * hiddenDim + i] = postAttn[i] + down[i];
        }

        double sumSqDiff = 0, sumSqReal = 0;
        double lastRowSumSqDiff = 0, lastRowSumSqReal = 0;
        int lastRowStart = (steps - 1) * hiddenDim;
        for (int i = 0; i < realOut.Length; i++)
        {
            double d = handOut[i] - realOut[i];
            sumSqDiff += d * d;
            sumSqReal += (double)realOut[i] * realOut[i];
            if (i >= lastRowStart)
            {
                lastRowSumSqDiff += d * d;
                lastRowSumSqReal += (double)realOut[i] * realOut[i];
            }
        }
        double relError = Math.Sqrt(sumSqDiff / Math.Max(1e-12, sumSqReal));
        double lastRowRelError = Math.Sqrt(lastRowSumSqDiff / Math.Max(1e-12, lastRowSumSqReal));
        Console.Error.WriteLine($"[FA-L0-Isolation] hand-computed vs real reference, ALL positions relative L2 error = {relError:F6}");
        Console.Error.WriteLine($"[FA-L0-Isolation] hand-computed vs real reference, LAST position relative L2 error = {lastRowRelError:F6}");

        for (int t = 0; t < steps; t++)
        {
            double d2 = 0, r2 = 0;
            for (int i = 0; i < hiddenDim; i++)
            {
                double d = handOut[t * hiddenDim + i] - realOut[t * hiddenDim + i];
                d2 += d * d;
                r2 += (double)realOut[t * hiddenDim + i] * realOut[t * hiddenDim + i];
            }
            double rowErr = Math.Sqrt(d2 / Math.Max(1e-12, r2));
            Console.Error.WriteLine($"[FA-L0-ROW] t={t} relError={rowErr:F6}");
        }

        Assert.True(double.IsFinite(relError));
    }
}
