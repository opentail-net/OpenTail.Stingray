using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// Real numeric golden-parity check for <see cref="AceStepDiT"/> against the real
/// `diffusers.models.transformers.ace_step_transformer.AceStepTransformer1DModel` reference,
/// loaded with the SAME real `turbo.safetensors` weights (remapped via a real, mechanical
/// key-rename this session verified with `load_state_dict(strict=False)` reporting zero missing
/// and zero unexpected keys -- confirming our tensor-name understanding matches diffusers' own
/// module shapes exactly) and run on fixed-seed synthetic inputs. See
/// docs/064-acestep-implementation-plan.md's "Golden-parity" section for how the reference dump
/// (`golden_dit_*.bin`) was produced -- NOT checked into the repo (gitignored scratch directory),
/// regenerate via the Python script referenced there if needed.
///
/// <para>This test is what caught a real bug this session: `AceStepDiTWeights` already loaded
/// the real `decoder.condition_embedder.weight/bias` tensors, but `AceStepDiT.PrepareCrossAttention`
/// never applied them to the condition sequence -- a real, silent correctness bug (finite,
/// shape-correct, non-degenerate output that was nonetheless numerically wrong) that only a real
/// reference comparison against the actual `diffusers` source, not just "does it run" testing,
/// would catch.</para>
/// </summary>
public sealed class AceStepDiTGoldenParityTests
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

    private static float[] ReadBin(string path, int count)
    {
        var bytes = File.ReadAllBytes(path);
        Assert.Equal(count * 4, bytes.Length);
        var result = new float[count];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }

    private static float[][] ToRows(float[] flat, int rows, int cols)
    {
        var result = new float[rows][];
        for (int r = 0; r < rows; r++)
        {
            result[r] = new float[cols];
            Array.Copy(flat, r * cols, result[r], 0, cols);
        }
        return result;
    }

    [Fact]
    public void Forward_RealWeights_MatchesRealDiffusersReference()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        Assert.SkipUnless(turboPath != null, "models/acestep-v15/turbo.safetensors not found");

        string scratchDir = @"C:\Users\Dmitri\AppData\Local\Temp\claude\C--Git-Public-OpenTail-Stingray\6cb31b57-ce45-49d6-9926-8736cdcfcfa9\scratchpad\acestep";
        string hiddenPath = Path.Combine(scratchDir, "golden_dit_hidden_states.bin");
        Assert.SkipUnless(File.Exists(hiddenPath), "golden_dit_*.bin reference dump not found -- regenerate via golden_dit.py (see docs/064)");

        using var loader = SafetensorsLoader.Open(turboPath!);
        var weights = AceStepDiTWeights.Load(loader);

        int seqLen = 50, condLen = 32;
        var hiddenFlat = ReadBin(Path.Combine(scratchDir, "golden_dit_hidden_states.bin"), seqLen * 64);
        var contextFlat = ReadBin(Path.Combine(scratchDir, "golden_dit_context_latents.bin"), seqLen * 128);
        var condFlat = ReadBin(Path.Combine(scratchDir, "golden_dit_encoder_hidden_states.bin"), condLen * 2048);
        var expectedFlat = ReadBin(Path.Combine(scratchDir, "golden_dit_output.bin"), seqLen * 64);

        var hidden = ToRows(hiddenFlat, seqLen, 64);
        var context = ToRows(contextFlat, seqLen, 128);
        var condition = ToRows(condFlat, condLen, 2048);
        var expected = ToRows(expectedFlat, seqLen, 64);

        var (patches, originalSeqLen) = AceStepDiT.ProjIn(weights, context, hidden);
        var ctx = AceStepDiT.PrepareCrossAttention(weights, condition, patches.Length);
        var patchesOut = AceStepDiT.Forward(weights, patches, timestep: 0.75f, timestepR: 0.75f, ctx);
        var actual = AceStepDiT.ProjOut(weights, patchesOut, originalSeqLen);

        Assert.Equal(seqLen, actual.Length);

        double maxAbsDiff = 0, sumAbsDiff = 0, sumAbsExpected = 0;
        int count = 0;
        for (int t = 0; t < seqLen; t++)
        {
            for (int c = 0; c < 64; c++)
            {
                double diff = Math.Abs(actual[t][c] - expected[t][c]);
                maxAbsDiff = Math.Max(maxAbsDiff, diff);
                sumAbsDiff += diff;
                sumAbsExpected += Math.Abs(expected[t][c]);
                count++;
            }
        }
        double meanAbsDiff = sumAbsDiff / count;
        double relError = sumAbsDiff / sumAbsExpected;

        // Measured relative error against the real reference is ~7e-6 (essentially F32-rounding-
        // level agreement over 24 real transformer layers) -- this tolerance has generous headroom
        // above that measured value while still being tight enough that a real bug (like the
        // missing condition_embedder this test caught, which produced order-1 relative error) fails
        // loudly rather than slipping through.
        Assert.True(relError < 0.001, $"relative error {relError:F6} exceeds tolerance -- real numeric mismatch against diffusers reference (maxAbsDiff={maxAbsDiff:F6}, meanAbsDiff={meanAbsDiff:F6})");
    }
}
