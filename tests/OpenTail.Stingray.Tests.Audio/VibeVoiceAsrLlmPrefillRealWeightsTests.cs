using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight test for the fix that closes VibeVoice ASR's documented real RAM-ceiling blocker
/// (docs/audio-review-progress.md's VibeVoice ASR entries): `VibeVoiceLlmTensorSource` now passes
/// per-layer weight matrices through under their REAL on-disk `DType` (zero-copy, `ForwardPass`
/// dequantizes on the fly) instead of eagerly materializing a full FP32 copy of every weight in
/// this 7B-class checkpoint. Constructs the real `ForwardPass` and runs a real, small `Prefill`
/// call -- exactly the operation that previously threw `OutOfMemoryException` inside the FP32
/// dequant-at-load path.
/// </summary>
public sealed class VibeVoiceAsrLlmPrefillRealWeightsTests : HeavyTestBase
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

    // Real decoder_config values (Qwen2-family), confirmed against the real checkpoint this
    // session (docs/audio-review-progress.md's VibeVoice ASR "checkpoint download finished" entry).
    private const int NumLayers = 28, HiddenDim = 3584, NumHeads = 28, NumKvHeads = 4, HeadDim = 128;
    private const int FfDim = 18944, VocabSize = 152064;
    private const float RopeTheta = 1000000f, RmsNormEps = 1e-6f;

    [Fact]
    public void Prefill_OnRealCheckpoint_DoesNotOom()
    {
        string? path = FindRepoFile("models/_models/vibevoice_asr/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-asr-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        var llm = new VibeVoiceLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var logits = fwd.Prefill([1, 2, 3, 4, 5], startPos: 0);
        Assert.Equal(VocabSize, logits.Length);
        Assert.All(logits.ToArray(), l => Assert.True(float.IsFinite(l)));
    }
}
