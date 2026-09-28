namespace OpenTail.Stingray.Tests.ForwardPass;

// Scratch (untracked): per-layer hidden states of the LAST prompt token, first 3 / last 3 values, in the shape
// llama-eval-callback prints, for a layer-by-layer diff against llama.cpp. ZZ_DUMP=1, ZZ_MODEL, ZZ_IDS_FILE, ZZ_OUT.
public sealed class ZzLayerDumpTmp
{
    [Fact]
    public void Dump()
    {
        if (Environment.GetEnvironmentVariable("ZZ_DUMP") != "1") return;
        using var model = GgufModel.Open(Environment.GetEnvironmentVariable("ZZ_MODEL")!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var ids = File.ReadAllText(Environment.GetEnvironmentVariable("ZZ_IDS_FILE")!).Split(',').Select(int.Parse).ToList();

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024);
        if (ids.Count > 1) fwd.Prefill(ids.Take(ids.Count - 1).ToList());

        Engine.StageCapture.Records.Clear();
        Engine.StageCapture.Enabled = true;
        fwd.Forward(ids[^1], ids.Count - 1);
        Engine.StageCapture.Enabled = false;

        using var w = new StreamWriter(Environment.GetEnvironmentVariable("ZZ_OUT")!);
        foreach (var (_, layer, stage, data) in Engine.StageCapture.Records)
        {
            if (stage is not ("post_attn_resid" or "post_ffn_resid" or "attn_norm" or "attn_out" or "o_proj")) continue;
            int n = data.Length;
            w.WriteLine($"{layer} {stage} {data[0]:F4} {data[1]:F4} {data[2]:F4} ... {data[n - 3]:F4} {data[n - 2]:F4} {data[n - 1]:F4}");
        }
        Engine.StageCapture.Reset();
    }
}
