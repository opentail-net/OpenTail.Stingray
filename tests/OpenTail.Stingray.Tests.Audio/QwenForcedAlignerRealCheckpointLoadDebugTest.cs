
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: validates whether QwenAsrWeights.LoadFromSafetensors -- written for
/// the base Qwen/Qwen3-ASR-0.6B checkpoint -- also loads the real Qwen/Qwen3-ForcedAligner-0.6B
/// checkpoint cleanly, since both share the same thinker_config.audio_config/text_config schema
/// and (per a direct tensor dump) identical thinker.audio_tower.*/thinker.model.* tensor names.</summary>
public sealed class QwenForcedAlignerRealCheckpointLoadDebugTest : HeavyTestBase
{
    private static string? FindRepoDir(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void LoadForcedAlignerCheckpoint_ViaQwenAsrWeights()
    {
        string? dir = FindRepoDir("models/qwen3-forcedaligner");
        Assert.SkipUnless(dir != null, "models/qwen3-forcedaligner not found");

        using var w = OpenTail.Stingray.Audio.QwenASR.QwenAsrWeights.LoadFromSafetensors(dir!);
        Console.Error.WriteLine($"[FA-Load] AudioLayers={w.AudioLayers} AudioDim={w.AudioDim} AudioHeads={w.AudioHeads} AudioFfDim={w.AudioFfDim} AudioConvChannels={w.AudioConvChannels} AudioProjDim={w.AudioProjDim}");
        Console.Error.WriteLine($"[FA-Load] LlmLayers={w.LlmLayers} LlmDim={w.LlmDim} LlmHeads={w.LlmHeads} LlmKvHeads={w.LlmKvHeads} LlmHeadDim={w.LlmHeadDim} LlmFfDim={w.LlmFfDim} LlmVocabSize={w.LlmVocabSize}");
        Console.Error.WriteLine($"[FA-Load] AudioStart={w.AudioStartTokenId} AudioEnd={w.AudioEndTokenId} AudioPad={w.AudioPadTokenId}");
        Assert.Equal(24, w.AudioLayers);
        Assert.Equal(28, w.LlmLayers);

        // Also confirm the swapped classify head tensor is reachable by its real name.
        var classifyHead = w.GetTensor("thinker.lm_head.weight");
        Console.Error.WriteLine($"[FA-Load] classifyHead.Length={classifyHead.Length} (expect 5000*1024={5000 * 1024})");
        Assert.Equal(5000 * 1024, classifyHead.Length);
    }
}
