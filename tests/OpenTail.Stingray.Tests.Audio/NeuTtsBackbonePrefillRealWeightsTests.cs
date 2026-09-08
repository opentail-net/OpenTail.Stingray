using OpenTail.Stingray.Audio.NeuTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt smoke test for the newly-ported NeuTTS backbone LM (a standard Qwen3-family
/// AR decoder, bridged into `OpenTail.Stingray.Engine`'s existing `ForwardPass` via
/// `NeuTtsBackboneTensorSource` -- same technique as `HiggsLlmTensorSource`/
/// `VibeVoiceLlmTensorSource`, no bespoke transformer math needed). Constructs the real
/// `ForwardPass` over the real downloaded checkpoint and runs a real `Prefill` call. This is
/// backbone-only per the session's real scoping note (docs/audio-review-progress.md's NeuTTS
/// entry) -- the FSQ acoustic-decoder codec is a separate, not-yet-started task.
/// </summary>
public sealed class NeuTtsBackbonePrefillRealWeightsTests : HeavyTestBase
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

    [Fact]
    public void Prefill_OnRealCheckpoint_ProducesFiniteLogits()
    {
        string? path = FindRepoFile("models/_models/neutts/NeuTTS-2E-GGUF/neutts-2e-orig.gguf");
        Assert.SkipUnless(path != null, "neutts-2e-orig.gguf not found");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        using var llm = new NeuTtsBackboneTensorSource(source);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var logits = fwd.Prefill([1, 2, 3, 4, 5], startPos: 0);
        sw.Stop();

        Assert.Equal(NeuTtsBackboneTensorSource.VocabSize, logits.Length);
        Assert.All(logits.ToArray(), l => Assert.True(float.IsFinite(l)));
        Console.WriteLine($"[NeuTtsBackbone] prefill of 5 tokens: {sw.ElapsedMilliseconds}ms, vocab={logits.Length}");
    }
}
