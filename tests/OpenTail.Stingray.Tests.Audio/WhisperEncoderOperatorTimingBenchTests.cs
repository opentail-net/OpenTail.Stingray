using System;
using System.Diagnostics;
using System.IO;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using OpenTail.Stingray.Audio.Whisper;
using OpenTail.Stingray.Cpu;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway "Experiment 2" from this session's phase-timing investigation (see
/// docs/audio-review-progress.md): the prior WhisperPhaseTimingBenchTests found the encoder is
/// 87-88% of total wall time at every model size. This bench breaks that 87% down into buckets
/// (QKV projections / attention compute / output projection / FFN up+down / LayerNorm+residual)
/// to find which operator inside the encoder actually dominates, per the same external-advice
/// methodology: measure before choosing what to optimize.
///
/// WhisperEncoder's per-operator methods (LinearReal, ComputeMultiHeadSelfAttentionReal, etc.) are
/// `private static` and take `Span&lt;float&gt;`/`ReadOnlySpan&lt;float&gt;` parameters, which reflection
/// cannot invoke (ref structs can't be boxed for MethodInfo.Invoke) -- so instead of reflecting
/// into the private wrapper, this drives the actual real, PUBLIC kernel those wrappers call
/// (<see cref="SimdKernels.MatVecF32"/>, with the identical parallel-per-row dispatch WhisperEncoder's
/// real LinearReal uses, see its own doc comment) directly against real weights
/// (<see cref="WhisperEncoderWeights"/>'s public fields). This measures the exact same expensive
/// kernel calls the production code makes, not a "similar" reimplementation -- only the thin
/// orchestration around them is duplicated (copied verbatim from WhisperEncoder.cs, not modified).
/// Zero production code changed. Not part of the permanent suite, delete after use.
/// </summary>
public sealed class WhisperEncoderOperatorTimingBenchTests : HeavyTestBase
{
    private static readonly (string Label, string FileName)[] Models =
    [
        ("Small", "ggml-small.bin"),
        ("Medium", "ggml-medium.bin"),
        ("Large-v3", "ggml-large-v3.bin"),
    ];

    private sealed class Buckets
    {
        public double AttnLn, QProj, KProj, VProj, AttnCompute, OutProj, AttnResidual;
        public double MlpLn, FfnUp, Activation, FfnDown, MlpResidual;
        public double Total => AttnLn + QProj + KProj + VProj + AttnCompute + OutProj + AttnResidual
                              + MlpLn + FfnUp + Activation + FfnDown + MlpResidual;
    }

    [Fact]
    public void EncoderOperatorBreakdown_AcrossModelSizes()
    {
        var report = new System.Text.StringBuilder();
        report.AppendLine($"ProcessorCount={Environment.ProcessorCount}");
        bool anyRan = false;

        foreach (var (label, fileName) in Models)
        {
            string? modelPath = FindModelPath(fileName);
            if (modelPath is null)
            {
                report.AppendLine($"{label}: SKIPPED (models/{fileName} not found)");
                continue;
            }
            anyRan = true;

            var ggml = WhisperGgmlModel.Load(modelPath);
            var config = ggml.ToConfig();
            var weights = new WhisperEncoderWeights(ggml);
            var melExtractor = new WhisperMelExtractor(config.NumMels);

            const int seconds = 12;
            int numSamples = 16000 * seconds;
            float[] audio = new float[numSamples];
            var rng = new Random(42);
            for (int i = 0; i < numSamples; i++)
                audio[i] = MathF.Sin(2.0f * MathF.PI * 220.0f * i / 16000.0f) * 0.3f
                         + MathF.Sin(2.0f * MathF.PI * 880.0f * i / 16000.0f) * 0.1f
                         + (float)(rng.NextDouble() - 0.5) * 0.02f;

            float[] mel = melExtractor.ExtractMel(audio, padTo30Seconds: true);
            int numFrames = mel.Length / config.NumMels;

            // Reuse the real conv downsample via the public Forward() once to get real encFrames
            // and a real starting hidden state -- avoids duplicating the conv/positional-embed
            // logic (not part of the bucket breakdown we care about; conv is a tiny, one-off cost
            // already known to be small from the phase-timing bench's "mel" bucket).
            var encoder = new WhisperEncoder(config, weights);
            encoder.Forward(mel, numFrames); // warmup, JIT, weight prefault
            int encFrames = Math.Min((numFrames + 1) / 2, config.AudioCtx);

            var x = new float[encFrames * config.AudioState];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

            const int n = 3;
            var totals = new Buckets[n];
            for (int i = 0; i < n; i++)
                totals[i] = RunLayers(x, encFrames, config, weights);

            double Mean(Func<Buckets, double> sel) { double s = 0; foreach (var b in totals) s += sel(b); return s / n; }
            var m = new Buckets
            {
                AttnLn = Mean(b => b.AttnLn), QProj = Mean(b => b.QProj), KProj = Mean(b => b.KProj),
                VProj = Mean(b => b.VProj), AttnCompute = Mean(b => b.AttnCompute), OutProj = Mean(b => b.OutProj),
                AttnResidual = Mean(b => b.AttnResidual), MlpLn = Mean(b => b.MlpLn), FfnUp = Mean(b => b.FfnUp),
                Activation = Mean(b => b.Activation), FfnDown = Mean(b => b.FfnDown), MlpResidual = Mean(b => b.MlpResidual),
            };
            double total = m.Total;

            report.AppendLine($"{label} ({fileName}): encFrames={encFrames} dModel={config.AudioState} layers={config.AudioLayer} total={total:F0}ms");
            report.AppendLine($"  AttnLayerNorm  {m.AttnLn,8:F0}ms ({Pct(m.AttnLn, total)})");
            report.AppendLine($"  Q projection   {m.QProj,8:F0}ms ({Pct(m.QProj, total)})");
            report.AppendLine($"  K projection   {m.KProj,8:F0}ms ({Pct(m.KProj, total)})");
            report.AppendLine($"  V projection   {m.VProj,8:F0}ms ({Pct(m.VProj, total)})");
            report.AppendLine($"  Attn compute   {m.AttnCompute,8:F0}ms ({Pct(m.AttnCompute, total)})");
            report.AppendLine($"  Out projection {m.OutProj,8:F0}ms ({Pct(m.OutProj, total)})");
            report.AppendLine($"  Attn residual  {m.AttnResidual,8:F0}ms ({Pct(m.AttnResidual, total)})");
            report.AppendLine($"  MlpLayerNorm   {m.MlpLn,8:F0}ms ({Pct(m.MlpLn, total)})");
            report.AppendLine($"  FFN up         {m.FfnUp,8:F0}ms ({Pct(m.FfnUp, total)})");
            report.AppendLine($"  Activation     {m.Activation,8:F0}ms ({Pct(m.Activation, total)})");
            report.AppendLine($"  FFN down       {m.FfnDown,8:F0}ms ({Pct(m.FfnDown, total)})");
            report.AppendLine($"  Mlp residual   {m.MlpResidual,8:F0}ms ({Pct(m.MlpResidual, total)})");
            report.AppendLine($"  -- QKV+Out total: {Pct(m.QProj + m.KProj + m.VProj + m.OutProj, total)}  |  FFN total: {Pct(m.FfnUp + m.FfnDown, total)}  |  Attn-math total: {Pct(m.AttnCompute, total)}  |  Norm/residual total: {Pct(m.AttnLn + m.MlpLn + m.AttnResidual + m.MlpResidual, total)}");
        }

        Assert.SkipUnless(anyRan, "No Whisper ggml models found under models/");

        var reportText = report.ToString();
        Console.Error.WriteLine(reportText);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "whisper_encoder_operator_result.txt"), reportText);
    }

    private static string Pct(double part, double total) => total > 0 ? $"{100.0 * part / total:F1}%" : "0%";

    /// <summary>Runs all encoder transformer layers once, timing each operator bucket. Mirrors WhisperEncoder.Forward's per-layer loop exactly (copied, not modified).</summary>
    private static Buckets RunLayers(float[] xInit, int encFrames, WhisperConfig config, WhisperEncoderWeights weights)
    {
        int dModel = config.AudioState;
        int nHeads = config.AudioHead;
        int headDim = dModel / nHeads;
        float eps = config.LayerNormEps;
        var b = new Buckets();
        var sw = new Stopwatch();

        var x = (float[])xInit.Clone();
        var normed = new float[encFrames * dModel];
        var q = new float[encFrames * dModel];
        var k = new float[encFrames * dModel];
        var v = new float[encFrames * dModel];
        var attnRaw = new float[encFrames * dModel];
        var attnOut = new float[encFrames * dModel];
        var hidden = new float[encFrames * dModel * 4];
        var mlpOut = new float[encFrames * dModel];
        float scale = 1.0f / MathF.Sqrt(headDim);

        for (int l = 0; l < config.AudioLayer; l++)
        {
            var lw = weights.Layers[l];

            sw.Restart();
            Parallel.For(0, encFrames, t => LayerNormAffine(x.AsSpan(t * dModel, dModel), lw.AttnLnWeight, lw.AttnLnBias, normed.AsSpan(t * dModel, dModel), eps));
            b.AttnLn += sw.Elapsed.TotalMilliseconds;

            sw.Restart(); LinearBucket(normed, encFrames, dModel, lw.QueryWeight, lw.QueryBias, dModel, q); b.QProj += sw.Elapsed.TotalMilliseconds;
            sw.Restart(); LinearBucket(normed, encFrames, dModel, lw.KeyWeight, null, dModel, k); b.KProj += sw.Elapsed.TotalMilliseconds;
            sw.Restart(); LinearBucket(normed, encFrames, dModel, lw.ValueWeight, lw.ValueBias, dModel, v); b.VProj += sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            Parallel.For(0, nHeads, h =>
            {
                int headOff = h * headDim;
                var scores = new float[encFrames];
                for (int i = 0; i < encFrames; i++)
                {
                    var querySpan = q.AsSpan(i * dModel + headOff, headDim);
                    for (int j = 0; j < encFrames; j++)
                        scores[j] = TensorPrimitives.Dot(querySpan, k.AsSpan(j * dModel + headOff, headDim)) * scale;
                    TensorPrimitives.SoftMax(scores.AsSpan(0, encFrames), scores.AsSpan(0, encFrames));
                    var weighted = attnRaw.AsSpan(i * dModel + headOff, headDim);
                    weighted.Clear();
                    for (int j = 0; j < encFrames; j++)
                        TensorPrimitives.MultiplyAdd(v.AsSpan(j * dModel + headOff, headDim), scores[j], weighted, weighted);
                }
            });
            b.AttnCompute += sw.Elapsed.TotalMilliseconds;

            sw.Restart(); LinearBucket(attnRaw, encFrames, dModel, lw.OutWeight, lw.OutBias, dModel, attnOut); b.OutProj += sw.Elapsed.TotalMilliseconds;

            sw.Restart(); TensorPrimitives.Add(x, attnOut, x); b.AttnResidual += sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            Parallel.For(0, encFrames, t => LayerNormAffine(x.AsSpan(t * dModel, dModel), lw.MlpLnWeight, lw.MlpLnBias, normed.AsSpan(t * dModel, dModel), eps));
            b.MlpLn += sw.Elapsed.TotalMilliseconds;

            sw.Restart(); LinearBucket(normed, encFrames, dModel, lw.Mlp0Weight, lw.Mlp0Bias, dModel * 4, hidden); b.FfnUp += sw.Elapsed.TotalMilliseconds;
            sw.Restart(); Parallel.For(0, hidden.Length, i => hidden[i] = Gelu(hidden[i])); b.Activation += sw.Elapsed.TotalMilliseconds;
            sw.Restart(); LinearBucket(hidden, encFrames, dModel * 4, lw.Mlp2Weight, lw.Mlp2Bias, dModel, mlpOut); b.FfnDown += sw.Elapsed.TotalMilliseconds;

            sw.Restart(); TensorPrimitives.Add(x, mlpOut, x); b.MlpResidual += sw.Elapsed.TotalMilliseconds;
        }

        return b;
    }

    /// <summary>Identical dispatch to WhisperEncoder's real (private) LinearReal: parallel per-row MatVecF32 for seqLen>=8, serial MatMulBatchedF32 otherwise. Copied verbatim, not modified.</summary>
    private static unsafe void LinearBucket(float[] input, int seqLen, int inDim, float[] weight, float[]? bias, int outDim, float[] output)
    {
        fixed (float* pIn = input, pW = weight, pOut = output)
        {
            if (seqLen >= 8)
            {
                nint inAddr = (nint)pIn, wAddr = (nint)pW, outAddr = (nint)pOut;
                Parallel.For(0, seqLen, t =>
                {
                    unsafe
                    {
                        float* rowIn = (float*)inAddr + (nuint)t * (nuint)inDim;
                        float* rowOut = (float*)outAddr + (nuint)t * (nuint)outDim;
                        SimdKernels.MatVecF32(rowOut, (float*)wAddr, rowIn, outDim, inDim);
                    }
                });
            }
            else
            {
                SimdKernels.MatMulBatchedF32(pOut, pW, pIn, seqLen, outDim, inDim);
            }
        }

        if (bias != null)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var row = output.AsSpan(t * outDim, outDim);
                TensorPrimitives.Add(row, bias, row);
            }
        }
    }

    private static void LayerNormAffine(ReadOnlySpan<float> input, float[] weight, float[] bias, Span<float> output, float eps)
    {
        int n = input.Length;
        float mean = TensorPrimitives.Sum(input) / n;
        float variance = 0f;
        for (int i = 0; i < n; i++) { float diff = input[i] - mean; variance += diff * diff; }
        variance /= n;
        float invStd = 1.0f / MathF.Sqrt(variance + eps);
        for (int i = 0; i < n; i++)
            output[i] = (input[i] - mean) * invStd * weight[i] + bias[i];
    }

    private static float Gelu(float x) => 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
