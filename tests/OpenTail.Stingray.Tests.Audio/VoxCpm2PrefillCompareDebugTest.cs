using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: prints this port's own real prefill output (lmHidden/residualHidden
/// right after the text-only prefill loop, BEFORE any CFM patch generation) at the same sample
/// indices the vendored reference's new `voxcpm2.step.lm_hidden`/`voxcpm2.step.residual_hidden`
/// trace emits for its own real first step -- this is the one RNG-independent comparison point
/// (everything after step 0 depends on the CFM's own noise, which is a real, accepted non-bit-
/// exact-RNG gap, so comparing later steps wouldn't isolate a real formula bug from RNG drift).</summary>
public sealed class VoxCpm2PrefillCompareDebugTest : HeavyTestBase
{
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

    private const int NumLayers = 28, HiddenDim = 2048, NumHeads = 16, NumKvHeads = 2, HeadDim = 128;
    private const int FfDim = 6144, VocabSize = 73448;
    private const float RopeTheta = 10000f, RmsNormEps = 1e-5f;

    [Fact]
    public void PrintPrefillOutputAtReferenceSampleIndices()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        var tokenizer = VoxCpm2TextTokenizer.LoadFromPackedGguf(model);
        var textIds = tokenizer.Encode("Hello there, this is a real end to end test of speech synthesis.");
        var promptTokenIds = new int[textIds.Count + 1];
        textIds.CopyTo(promptTokenIds);
        promptTokenIds[^1] = tokenizer.AudioStartTokenId;
        Console.WriteLine($"[Ours] promptTokenIds=[{string.Join(",", promptTokenIds)}]");

        var llm = new VoxCpm2LlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, embeddingScale: 1f, residualScale: 1f, logitScale: 1f);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var residualLm = VoxCpm2ResidualLm.Load(source.GetTensor);
        var projWeights = VoxCpm2StepProjectionWeights.Load(source.GetTensor);

        // Replicate VoxCpm2Generator.Generate's own real text-only prefill loop exactly.
        var zeroHidden = new float[HiddenDim];
        float[] lastBaseHidden = zeroHidden;
        float[] lastResidualHidden = zeroHidden;
        for (int i = 0; i < promptTokenIds.Length; i++)
        {
            fwd.Prefill([promptTokenIds[i]], startPos: i);
            lastBaseHidden = fwd.LastHidden.ToArray();
            var concat = new float[HiddenDim * 2];
            Array.Copy(lastBaseHidden, concat, HiddenDim);
            var residualInput = Linear(concat, projWeights.FusionConcatProjWeight, projWeights.FusionConcatProjBias, HiddenDim * 2, HiddenDim);
            lastResidualHidden = residualLm.Step(residualInput);
        }

        // Real reference sample indices (from the vendored trace's own `sample_point_indices`,
        // captured verbatim from a real --log run of the reference CLI on the identical prompt).
        int[] indices = [0, 52, 104, 157, 209, 261, 314, 366, 419, 471, 523, 576, 628, 681, 682, 744, 806, 868, 930, 992, 1054, 1116, 1178, 1240, 1302, 1364, 1365, 1417, 1469, 1522, 1574, 1627, 1679, 1732, 1784, 1837, 1889, 1942, 1994, 2047];

        var lmSb = new System.Text.StringBuilder("[Ours] lm_hidden samples=[");
        foreach (int idx in indices) lmSb.Append($"{idx}:{lastBaseHidden[idx]:G6},");
        lmSb.Append(']');
        Console.WriteLine(lmSb.ToString());

        var resSb = new System.Text.StringBuilder("[Ours] residual_hidden samples=[");
        foreach (int idx in indices) resSb.Append($"{idx}:{lastResidualHidden[idx]:G6},");
        resSb.Append(']');
        Console.WriteLine(resSb.ToString());
    }

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }
}
