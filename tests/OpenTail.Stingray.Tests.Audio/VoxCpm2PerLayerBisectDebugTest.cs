using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: bisects VoxCPM2's real LM-forward-pass divergence found in
/// <see cref="VoxCpm2PrefillCompareDebugTest"/> (token ids match the reference exactly after the
/// tokenizer fix, but the final prefill hidden state still diverges). Uses the existing
/// <see cref="IForwardPass.EnableHiddenTaps"/>/<see cref="IForwardPass.HiddenTapsAt"/> API (built
/// for speculative decoding, reused here as-is -- no production code changes needed) to capture
/// EVERY layer's hidden state at the LAST prefill position, printed at the same real reference
/// sample indices the vendored reference's new per-layer `voxcpm2.prefill.layer{i}.hidden` trace
/// emits (see `examples/audio.cpp/src/models/voxcpm2/minicpm.cpp`'s `VoxCPM2PromptPrefillRuntime::
/// Impl::run`, instrumented this session) -- finds the FIRST layer index where the two diverge.</summary>
public sealed class VoxCpm2PerLayerBisectDebugTest : HeavyTestBase
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
    public void PrintPerLayerHiddenAtReferenceSampleIndices()
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

        var llm = new VoxCpm2LlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, embeddingScale: 1f, residualScale: 1f, logitScale: 1f);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        Assert.True(fwd.SupportsHiddenTaps, "CPU ForwardPass must support hidden taps for this bisection.");
        fwd.EnableHiddenTaps([.. Enumerable.Range(0, NumLayers)]);

        fwd.Prefill(promptTokenIds, startPos: 0);

        int lastPos = promptTokenIds.Length - 1;
        var allLayers = fwd.HiddenTapsAt(lastPos);

        // Real reference sample indices (same 40 points VoxCpm2PrefillCompareDebugTest already
        // uses, captured verbatim from the vendored reference's own trace output).
        int[] indices = [0, 52, 104, 157, 209, 261, 314, 366, 419, 471, 523, 576, 628, 681, 682, 744, 806, 868, 930, 992, 1054, 1116, 1178, 1240, 1302, 1364, 1365, 1417, 1469, 1522, 1574, 1627, 1679, 1732, 1784, 1837, 1889, 1942, 1994, 2047];

        var lines = new List<string>();
        for (int layer = 0; layer < NumLayers; layer++)
        {
            var row = allLayers.Slice(layer * HiddenDim, HiddenDim);
            var sb = new System.Text.StringBuilder($"[Ours] layer{layer} hidden samples=[");
            foreach (int idx in indices) sb.Append($"{idx}:{row[idx]:G6},");
            sb.Append(']');
            lines.Add(sb.ToString());
            Console.WriteLine(sb.ToString());
        }
        Directory.CreateDirectory("scratch");
        File.WriteAllLines("scratch/ours_trace.log", lines);
    }
}
