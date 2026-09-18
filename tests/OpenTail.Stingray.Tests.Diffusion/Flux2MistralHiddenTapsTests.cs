namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real smoke test for FLUX.2's text-conditioning extraction: confirms
/// <see cref="Engine.ForwardPass.EnableHiddenTaps"/> (already-existing PR #413 mechanism, docs/087)
/// runs end-to-end against the real Mistral-Small-3.2-24B checkpoint and produces the real
/// 15360-wide (3x5120) concatenated hidden-state vector FLUX.2's recipe needs. NOT a numeric
/// golden-parity check against a real independent Mistral reference (docs/087 explicitly calls
/// for that as a separate, follow-up differential test before trusting this for FLUX.2 image
/// quality) -- this only confirms the real mechanism wires up correctly end-to-end and produces
/// finite, non-degenerate output.
/// </summary>
public sealed class Flux2MistralHiddenTapsTests
{
    private const string ModelFileName = "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf";

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Mistral_RealWeights_HiddenTaps_ProduceReal15360DimConditioning()
    {
        string? modelPath = FindModelPath(ModelFileName);
        Assert.SkipUnless(modelPath != null, "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf not found");

        using var model = GgufModel.Open(modelPath!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        Assert.True(fwd.SupportsHiddenTaps, "ForwardPass must support hidden taps for FLUX.2 text conditioning");

        // Real FLUX.2 recipe (docs/087): HF's literal hidden_states[10,20,30] = this codebase's
        // layer-output convention [9,19,29] ("layer i's tap is HF's hidden_states[i+1]").
        int[] tapLayers = [9, 19, 29];
        fwd.EnableHiddenTaps(tapLayers);
        Assert.Equal(3 * hp.EmbeddingDim, fwd.HiddenTapDim);
        Assert.Equal(15360, fwd.HiddenTapDim); // real FLUX.2 ContextInDim

        var tokens = tokenizer.Encode("a red apple on a wooden table").ToList();
        Assert.True(tokens.Count > 0);

        fwd.Prefill(tokens);

        for (int p = 0; p < tokens.Count; p++)
        {
            var tap = fwd.HiddenTapsAt(p);
            Assert.Equal(15360, tap.Length);
            bool allZero = true;
            foreach (var v in tap)
            {
                Assert.True(float.IsFinite(v), $"hidden tap at position {p} contains NaN/Inf");
                if (v != 0f) allZero = false;
            }
            Assert.False(allZero, $"hidden tap at position {p} is all-zero -- likely a wiring bug");
        }
    }
}
