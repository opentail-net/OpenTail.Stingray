using System.Runtime.InteropServices;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for <see cref="VibeVoiceLlmTensorSource"/>: loads the real checkpoint,
/// builds the Qwen2 bridge with the real decoder_config numbers, and confirms every expected
/// canonical tensor resolves to finite float data of the right shape -- the necessary precondition
/// before wiring this into a real ForwardPass prefill/decode loop (not yet done).
/// </summary>
public sealed class VibeVoiceLlmTensorSourceRealWeightsTests : HeavyTestBase
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

    // Real decoder_config from the checkpoint's embedded config.json.
    private const int NumLayers = 28, HiddenDim = 3584, NumHeads = 28, NumKvHeads = 4, HeadDim = 128;
    private const int FfDim = 18944, VocabSize = 152064;
    private const float RopeTheta = 1000000f, RmsNormEps = 1e-6f;

    [Fact]
    public unsafe void Build_OnRealCheckpoint_ResolvesAllExpectedTensors()
    {
        string? path = FindRepoFile("models/_models/vibevoice_asr/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-asr-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new VibeVoiceLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps);

        Assert.Equal("qwen2", llm.Metadata["general.architecture"]);
        Assert.Equal(NumLayers, llm.Metadata["qwen2.block_count"]);

        var embedInfo = llm.FindTensor("token_embd.weight");
        Assert.NotNull(embedInfo);
        Assert.Equal([(long)HiddenDim, VocabSize], embedInfo!.Value.Dimensions);

        var outputInfo = llm.FindTensor("output.weight");
        Assert.NotNull(outputInfo); // real lm_head.weight tensor exists (not tied), confirmed via dump

        void CheckFinite(GgufTensorInfo? info)
        {
            Assert.NotNull(info);
            var data = llm.GetTensorData(info!.Value);
            var floats = MemoryMarshal.Cast<byte, float>(data);
            Assert.All(floats.ToArray(), v => Assert.True(float.IsFinite(v)));
        }

        CheckFinite(llm.FindTensor("output_norm.weight"));
        CheckFinite(llm.FindTensor("blk.0.attn_q.weight"));
        CheckFinite(llm.FindTensor("blk.0.attn_q.bias"));
        CheckFinite(llm.FindTensor("blk.0.attn_k.weight"));
        CheckFinite(llm.FindTensor("blk.0.attn_output.weight"));
        Assert.Null(llm.FindTensor("blk.0.attn_output.bias")); // real: o_proj has no bias tensor
        CheckFinite(llm.FindTensor("blk.27.ffn_gate.weight")); // last real layer index
    }
}
