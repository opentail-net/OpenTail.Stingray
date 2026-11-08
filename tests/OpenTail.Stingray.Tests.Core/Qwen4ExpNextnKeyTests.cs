namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Regression for the MTP head key. The pinned llama.cpp reference and the Qwen 3.8 MTP head files use
/// <c>{arch}.nextn_predict_layers</c>; Stingray read <c>{arch}.nextn.layer_count</c> (found 2026-11-08).
/// </summary>
public sealed class Qwen4ExpNextnKeyTests
{
    [Fact]
    public void NextnCount_ReadsNextnPredictLayersKey()
    {
        var meta = new Dictionary<string, object>
        {
            ["qwen4exp.full_attention_interval"] = 4,
            ["qwen4exp.nextn_predict_layers"] = 1,
        };
        var hp = OpenTail.Stingray.Engine.Qwen4ExpHyperparams.FromMetadata(meta, totalLayers: 48);
        Assert.Equal(1, hp.NumLayerNextn);
    }

    [Fact]
    public void NextnCount_IsZeroWithoutTheKey()
    {
        var meta = new Dictionary<string, object> { ["qwen4exp.full_attention_interval"] = 4 };
        var hp = OpenTail.Stingray.Engine.Qwen4ExpHyperparams.FromMetadata(meta, totalLayers: 48);
        Assert.Equal(0, hp.NumLayerNextn);
    }
}
