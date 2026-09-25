
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="SileroVad"/>'s rewritten 16kHz forward pass
/// (see docs/audio-review-progress.md's Silero VAD section) -- runs the real
/// `models/silero_vad.onnx` via onnxruntime (`scratch-llamacpp-ref/silero_golden_input.txt` +
/// the golden probability captured alongside it) and checks this C# port's output against it,
/// not just shape/finite checks. The golden input is a seeded-random (numpy `default_rng(42)`)
/// synthetic 512-sample frame -- content doesn't matter for this check, only that the SAME
/// exact samples feed both the real ONNX graph and this C# port with a fresh (all-zero) LSTM
/// state on both sides.
/// </summary>
public sealed class SileroVadWeightsTests : HeavyTestBase
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

    [Fact]
    public void ProcessFrame_RealWeights_MatchesOnnxGoldenProbability()
    {
        string? onnxPath = FindRepoFile("models/silero_vad.onnx");
        Assert.SkipUnless(onnxPath != null, "models/silero_vad.onnx not found");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/silero_golden_input.txt");
        Assert.SkipUnless(goldenPath != null, "scratch-llamacpp-ref/silero_golden_input.txt not found (re-run the golden dump script)");

        var parts = File.ReadAllText(goldenPath!).Trim().Split(',');
        Assert.Equal(512, parts.Length);
        var frame = new float[512];
        for (int i = 0; i < 512; i++) frame[i] = float.Parse(parts[i]);

        using var vad = SileroVad.Load(onnxPath!);
        float prob = vad.ProcessFrame(frame);

        // Oracle computed live with onnxruntime on the OFFICIAL v5 input: 64 samples of context (all
        // zero for a fresh stream) + the 512-sample frame, fresh zero state. The old hard-coded golden
        // (0.025505661964416504) came from feeding onnxruntime the bare 512-sample frame, the same
        // missing-context convention the native port had until 2026-09-25, so it agreed with the bug.
        using var ort = OpenTail.Stingray.Core.OnnxModelSession.TryLoad(onnxPath);
        Assert.SkipUnless(ort is { IsAvailable: true }, "onnxruntime not available");
        var x = new float[576];
        frame.CopyTo(x, 64);
        var outs = ort!.Run(("input", x, new[] { 1, 576 }), ("state", new float[256], new[] { 2, 1, 128 }), ("sr", new long[] { 16000 }, Array.Empty<int>()));
        float goldenProb = outs["output"][0];
        Assert.True(MathF.Abs(prob - goldenProb) < 0.0001f,
            $"prob={prob} vs onnxruntime={goldenProb}, diff={MathF.Abs(prob - goldenProb)}");
    }
}
