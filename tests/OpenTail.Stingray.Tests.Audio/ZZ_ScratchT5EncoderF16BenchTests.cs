using OpenTail.Stingray.Audio.MusicGen;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Scratch: real A/B timing of the shared T5EncoderKernels forward pass with F16C-converted
/// weights (now the default via CfmLinearWeight.FromF32WithF16Conversion) vs. plain F32
/// (CfmLinearWeight.FromF32), same real MusicGen t5-base checkpoint, same real token sequence.
/// Not a golden-parity check (see MusicGenTextEncoderGoldenParityTests for that). Delete once
/// superseded.
/// </summary>
public sealed class ZZ_ScratchT5EncoderF16BenchTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    // Plain-F32 clone of T5EncoderKernels.Load, for the "before" side of the A/B.
    private static NonGatedT5EncoderWeights LoadF32(SafetensorsLoader loader, T5EncoderDims dims, string prefix)
    {
        var layers = new T5EncoderLayerWeights[dims.NumLayers];
        int qkvDim = dims.NumHeads * dims.DKv;
        for (int i = 0; i < dims.NumLayers; i++)
        {
            string p = $"{prefix}encoder.block.{i}";
            layers[i] = new T5EncoderLayerWeights
            {
                SelfAttnQWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.q.weight"), outDim: qkvDim, inDim: dims.DModel),
                SelfAttnKWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.k.weight"), outDim: qkvDim, inDim: dims.DModel),
                SelfAttnVWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.v.weight"), outDim: qkvDim, inDim: dims.DModel),
                SelfAttnOWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.0.SelfAttention.o.weight"), outDim: dims.DModel, inDim: qkvDim),
                SelfAttnLayerNormWeight = loader.ReadF32($"{p}.layer.0.layer_norm.weight"),
                FfnWiWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.1.DenseReluDense.wi.weight"), outDim: dims.DFf, inDim: dims.DModel),
                FfnWoWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.layer.1.DenseReluDense.wo.weight"), outDim: dims.DModel, inDim: dims.DFf),
                FfnLayerNormWeight = loader.ReadF32($"{p}.layer.1.layer_norm.weight"),
            };
        }
        return new NonGatedT5EncoderWeights
        {
            SharedEmbedding = loader.ReadF32($"{prefix}shared.weight"),
            Layers = layers,
            FinalLayerNormWeight = loader.ReadF32($"{prefix}encoder.final_layer_norm.weight"),
            RelativeAttentionBias = loader.ReadF32($"{prefix}encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
        };
    }

    [Fact]
    public void Forward_F16VsF32_RealWeights_MeasuresSpeedupAndAccuracy()
    {
        string? modelPath = FindRepoFile("models/musicgen-small/musicgen-small.safetensors");
        Assert.SkipUnless(modelPath != null, "models/musicgen-small/musicgen-small.safetensors not found");

        using var loaderF16 = SafetensorsLoader.Open(modelPath!);
        var weightsF16 = MusicGenTextEncoderWeights.Load(loaderF16); // real path: F16C-converted

        using var loaderF32 = SafetensorsLoader.Open(modelPath!);
        var weightsF32 = LoadF32(loaderF32, MusicGenTextEncoderWeights.Dims, prefix: "text_encoder.");

        // A real-length token sequence (~20 tokens), fixed for reproducibility.
        int[] tokenIds = [100, 202, 33, 4521, 88, 12, 900, 4, 76, 231, 55, 999, 1, 42, 613, 720, 8, 3, 250, 61];

        // Warmup both.
        _ = MusicGenTextEncoder.Forward(weightsF16, tokenIds);
        _ = MusicGenTextEncoder.Forward(weightsF32, tokenIds);

        const int n = 5;
        var msF16 = new double[n];
        var msF32 = new double[n];
        float[][] outF16 = null!, outF32 = null!;
        for (int i = 0; i < n; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            outF16 = MusicGenTextEncoder.Forward(weightsF16, tokenIds);
            msF16[i] = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            outF32 = MusicGenTextEncoder.Forward(weightsF32, tokenIds);
            msF32[i] = sw.Elapsed.TotalMilliseconds;
        }

        double MeanOf(double[] a) { double s = 0; foreach (var v in a) s += v; return s / a.Length; }
        double meanF16 = MeanOf(msF16), meanF32 = MeanOf(msF32);

        double dot = 0, nF16 = 0, nF32 = 0;
        for (int t = 0; t < outF16.Length; t++)
            for (int c = 0; c < outF16[t].Length; c++)
            {
                dot += (double)outF16[t][c] * outF32[t][c];
                nF16 += (double)outF16[t][c] * outF16[t][c];
                nF32 += (double)outF32[t][c] * outF32[t][c];
            }
        double cosine = dot / (Math.Sqrt(nF16) * Math.Sqrt(nF32));

        Console.WriteLine($"[bench] T5 encoder (MusicGen t5-base, {tokenIds.Length} tokens): F16={meanF16:F2}ms F32={meanF32:F2}ms speedup={(meanF32 / meanF16):F2}x cosine={cosine:F6}");
        Assert.True(cosine > 0.999, $"F16 vs F32 cosine similarity {cosine:F6} too low");
    }
}
