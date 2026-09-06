
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real load validation for OmniVoiceLlmTensorSource against the real downloaded
/// k2-fsa/OmniVoice checkpoint (models/_models/omnivoice/model.safetensors).</summary>
public sealed class OmniVoiceLlmTensorSourceLoadTests : HeavyTestBase
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
    public void Load_RealCheckpoint_AllLayersPresentWithFiniteValues()
    {
        string? path = FindRepoFile("models/_models/omnivoice/model.safetensors");
        Assert.SkipUnless(path != null, "omnivoice model.safetensors not found");

        // Real config.json: llm_config -- 28 layers, hidden=1024, 16 heads, 8 kv-heads,
        // head_dim=128, intermediate=3072, vocab=151676, rope_theta=1e6, rms_norm_eps=1e-6.
        using var source = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceLlmTensorSource(
            path!, numLayers: 28, hiddenDim: 1024, numHeads: 16, numKvHeads: 8, headDim: 128,
            ffDim: 3072, vocabSize: 151676, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

        Assert.NotNull(source.FindTensor("token_embd.weight"));
        Assert.NotNull(source.FindTensor("output_norm.weight"));
        Assert.Null(source.FindTensor("output.weight")); // tied embeddings, no separate lm_head

        for (int i = 0; i < 28; i++)
        {
            foreach (var name in new[]
            {
                $"blk.{i}.attn_norm.weight", $"blk.{i}.attn_q.weight", $"blk.{i}.attn_k.weight",
                $"blk.{i}.attn_v.weight", $"blk.{i}.attn_output.weight", $"blk.{i}.attn_q_norm.weight",
                $"blk.{i}.attn_k_norm.weight", $"blk.{i}.ffn_norm.weight", $"blk.{i}.ffn_gate.weight",
                $"blk.{i}.ffn_up.weight", $"blk.{i}.ffn_down.weight",
            })
            {
                var info = source.FindTensor(name);
                Assert.True(info is not null, $"missing tensor {name}");
                var bytes = source.GetTensorData(info!.Value);
                var floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes);
                foreach (var v in floats) Assert.True(float.IsFinite(v), $"{name} contains a non-finite value");
            }
        }
    }
}
