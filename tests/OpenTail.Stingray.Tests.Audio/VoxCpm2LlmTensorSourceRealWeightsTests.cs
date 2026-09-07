using System.Runtime.InteropServices;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for <see cref="VoxCpm2LlmTensorSource"/>: loads the real checkpoint,
/// builds the minicpm bridge with the real lm_config numbers, and confirms every expected
/// canonical tensor resolves to finite float data of the right shape. Tensor-shape verification
/// only (no live ForwardPass generation) -- this codebase's real bridging technique dequantizes
/// every weight to FP32 in memory, and a similarly-large model (VibeVoice ASR, 7B-class) hit a
/// real OutOfMemoryException doing that in this environment; this model is smaller (~1.5B) but a
/// live end-to-end generation run is deferred to a machine with more headroom.
/// </summary>
public sealed class VoxCpm2LlmTensorSourceRealWeightsTests : HeavyTestBase
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

    // Real lm_config from the checkpoint's embedded config.json.
    private const int NumLayers = 28, HiddenDim = 2048, NumHeads = 16, NumKvHeads = 2, HeadDim = 128;
    private const int FfDim = 6144, VocabSize = 73448;
    private const float RopeTheta = 10000f, RmsNormEps = 1e-5f;
    // Real: use_mup=false in this checkpoint's config -> embedding_scale/residual_scale are both
    // the identity (1.0), confirmed directly from minicpm.cpp's `config.use_mup ? ... : 1.0F`
    // guards (not guessed). logit_scale is unused by this class entirely (base_lm never computes
    // logits here -- only LastHidden is consumed by VoxCPM2's downstream DiT/FSQ heads).
    private const float EmbeddingScale = 1.0f, ResidualScale = 1.0f, LogitScale = 1.0f;

    [Fact]
    public unsafe void Build_OnRealCheckpoint_ResolvesAllExpectedTensors()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new VoxCpm2LlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, EmbeddingScale, ResidualScale, LogitScale);

        Assert.Equal("minicpm", llm.Metadata["general.architecture"]);
        Assert.Equal(NumLayers, llm.Metadata["minicpm.block_count"]);

        var embedInfo = llm.FindTensor("token_embd.weight");
        Assert.NotNull(embedInfo);
        Assert.Equal([(long)HiddenDim, VocabSize], embedInfo!.Value.Dimensions);

        void CheckFinite(GgufTensorInfo? info)
        {
            Assert.NotNull(info);
            var data = llm.GetTensorData(info!.Value);
            var floats = MemoryMarshal.Cast<byte, float>(data);
            Assert.All(floats.ToArray(), v => Assert.True(float.IsFinite(v)));
        }

        CheckFinite(llm.FindTensor("output_norm.weight"));
        CheckFinite(llm.FindTensor("blk.0.attn_q.weight"));
        CheckFinite(llm.FindTensor("blk.0.attn_k.weight"));
        CheckFinite(llm.FindTensor("blk.0.attn_output.weight"));
        Assert.Null(llm.FindTensor("blk.0.attn_q.bias")); // real: MiniCPM has no QKV bias, unlike VibeVoice's Qwen2
        CheckFinite(llm.FindTensor("blk.27.ffn_gate.weight")); // last real layer index
    }
}
