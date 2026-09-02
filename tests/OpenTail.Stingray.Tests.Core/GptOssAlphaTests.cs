using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// ALPHA/UNTESTED-AGAINST-REAL-WEIGHTS tests for GptOssGraph (GptOssAlpha.cs). Synthetic
/// invariant/formula checks only -- see docs/060-gpt-oss-implementation-plan.md.
/// </summary>
public class GptOssAlphaTests
{
    [Fact]
    public void SoftmaxWithSink_NoSink_SumsToOne()
    {
        float[] scores = [1f, 2f, 3f];
        GptOssGraph.SoftmaxWithSink(scores, null);

        float sum = scores[0] + scores[1] + scores[2];
        Assert.Equal(1f, sum, 5);
    }

    [Fact]
    public void SoftmaxWithSink_WithSink_SumsToLessThanOne()
    {
        // A large positive sink should absorb a meaningful share of the softmax mass, leaving
        // the real keys' weights summing to strictly less than 1.
        float[] scores = [1f, 2f, 3f];
        GptOssGraph.SoftmaxWithSink(scores, sink: 5f);

        float sum = scores[0] + scores[1] + scores[2];
        Assert.True(sum < 1f, $"expected sum < 1 with a dominant sink, got {sum}");
        Assert.True(sum > 0f);
    }

    [Fact]
    public void SoftmaxWithSink_VanishinglySmallSink_ApproachesSumOne()
    {
        // A very negative (irrelevant) sink should contribute ~nothing to the denominator.
        float[] scores = [1f, 2f, 3f];
        GptOssGraph.SoftmaxWithSink(scores, sink: -1000f);

        float sum = scores[0] + scores[1] + scores[2];
        Assert.True(sum > 0.999f, $"expected sum ~1 with a negligible sink, got {sum}");
    }

    [Fact]
    public void SoftmaxWithSink_PreservesRelativeOrderOfRealScores()
    {
        float[] scores = [1f, 5f, 2f];
        GptOssGraph.SoftmaxWithSink(scores, sink: 3f);

        Assert.True(scores[1] > scores[2]);
        Assert.True(scores[2] > scores[0]);
    }

    [Fact]
    public void SwigluOai_MatchesHandComputedValue()
    {
        // alpha=1.702, limit=7 (reference defaults). gate=1, up=1 (no clamping triggered).
        float[] gate = [1f];
        float[] up = [1f];
        var output = new float[1];
        GptOssGraph.SwigluOai(gate, up, output);

        float expectedSwish = 1f / (1f + MathF.Exp(-1.702f * 1f));
        float expected = expectedSwish * (1f + 1f);
        Assert.Equal(expected, output[0], 5);
    }

    [Fact]
    public void SwigluOai_ClampsGateAboveLimit()
    {
        // gate way above the limit (7.0) should behave as if clamped to exactly 7.0.
        float[] gateHigh = [100f];
        float[] gateAtLimit = [7f];
        float[] up = [0f];
        var outHigh = new float[1];
        var outAtLimit = new float[1];
        GptOssGraph.SwigluOai(gateHigh, up, outHigh);
        GptOssGraph.SwigluOai(gateAtLimit, up, outAtLimit);

        Assert.Equal(outAtLimit[0], outHigh[0], 5);
    }

    [Fact]
    public void SwigluOai_ClampsUpToSymmetricRange()
    {
        float[] gate = [1f];
        float[] upHigh = [100f];
        float[] upAtLimit = [7f];
        float[] upLow = [-100f];
        float[] upAtNegLimit = [-7f];
        var outHigh = new float[1];
        var outAtLimit = new float[1];
        var outLow = new float[1];
        var outAtNegLimit = new float[1];
        GptOssGraph.SwigluOai(gate, upHigh, outHigh);
        GptOssGraph.SwigluOai(gate, upAtLimit, outAtLimit);
        GptOssGraph.SwigluOai(gate, upLow, outLow);
        GptOssGraph.SwigluOai(gate, upAtNegLimit, outAtNegLimit);

        Assert.Equal(outAtLimit[0], outHigh[0], 5);
        Assert.Equal(outAtNegLimit[0], outLow[0], 5);
    }

    [Fact]
    public void SwigluOai_IsAdditiveNotMultiplicative()
    {
        // The whole point of this formula (vs. standard SwiGLU) is swish*(1+up), not swish*up --
        // pin that up=0 does NOT zero the output (unlike standard gate*up SwiGLU, where up=0
        // always gives 0 regardless of gate).
        float[] gate = [2f];
        float[] upZero = [0f];
        var output = new float[1];
        GptOssGraph.SwigluOai(gate, upZero, output);

        Assert.NotEqual(0f, output[0]);
    }

    [Fact]
    public void SelectThenSoftmaxGate_SelectsTopKByRawLogit()
    {
        float[] logits = [0.1f, 5.0f, -3.0f, 4.9f, 0.5f];
        var indices = new int[2];
        var weights = new float[2];
        GptOssGraph.SelectThenSoftmaxGate(logits, topK: 2, indices, weights);

        Assert.Contains(1, indices); // logit 5.0
        Assert.Contains(3, indices); // logit 4.9
    }

    [Fact]
    public void SelectThenSoftmaxGate_WeightsSumToOne()
    {
        float[] logits = [1f, 2f, 3f, 4f];
        var indices = new int[3];
        var weights = new float[3];
        GptOssGraph.SelectThenSoftmaxGate(logits, topK: 3, indices, weights);

        float sum = weights[0] + weights[1] + weights[2];
        Assert.Equal(1f, sum, 5);
    }

    [Fact]
    public void SelectThenSoftmaxGate_ExcludedLogitDoesNotAffectSelectedWeights()
    {
        // Softmax over {5, 3} should be identical whether or not a much-lower excluded logit
        // (e.g. -100) exists in the full vector -- proving the softmax runs ONLY over the
        // selected subset, not the full expert set (the defining property of this gating
        // function vs. every other MoE architecture's softmax-then-select).
        float[] withExcluded = [5f, 3f, -100f];
        float[] withoutExcluded = [5f, 3f];
        var idx1 = new int[2]; var w1 = new float[2];
        var idx2 = new int[2]; var w2 = new float[2];
        GptOssGraph.SelectThenSoftmaxGate(withExcluded, 2, idx1, w1);
        GptOssGraph.SelectThenSoftmaxGate(withoutExcluded, 2, idx2, w2);

        Array.Sort(w1);
        Array.Sort(w2);
        Assert.Equal(w2[0], w1[0], 5);
        Assert.Equal(w2[1], w1[1], 5);
    }

    [Fact]
    public void FromGgufMetadata_ReadsGptOssKeys()
    {
        var metadata = new Dictionary<string, object>
        {
            ["gpt-oss.attention.layer_norm_rms_epsilon"] = 1e-5f,
            ["gpt-oss.expert_feed_forward_length"] = (ulong)2880,
            ["gpt-oss.attention.sliding_window"] = (ulong)128,
            ["gpt-oss.attention.sliding_window_pattern"] = (ulong)2,
            ["gpt-oss.rope.freq_base"] = 150000f,
            ["gpt-oss.rope.freq_base_swa"] = 150000f,
        };

        var hp = GptOssHyperparams.FromGgufMetadata(
            metadata, "gpt-oss", numLayer: 24, embedDim: 2880, numHeads: 64, numHeadsKv: 8,
            headDim: 64, numExperts: 32, numExpertsUsed: 4, vocabSize: 201088);

        Assert.Equal(2880, hp.ExpertFeedForwardLength);
        Assert.Equal(128, hp.SlidingWindow);
        Assert.Equal(2, hp.SwaPeriod);
        Assert.Equal(150000f, hp.RopeFreqBase);
    }

    [Fact]
    public void IsSwaLayer_AlternatesStrictly1To1StartingWithSwa()
    {
        var hp = new GptOssHyperparams { SwaPeriod = 2 };

        Assert.True(hp.IsSwaLayer(0));
        Assert.False(hp.IsSwaLayer(1));
        Assert.True(hp.IsSwaLayer(2));
        Assert.False(hp.IsSwaLayer(3));
    }

    [Fact]
    public void RopeFreqBaseSwa_DefaultsToRopeFreqBase_WhenNotDeclared()
    {
        var metadata = new Dictionary<string, object>
        {
            ["gpt-oss.rope.freq_base"] = 150000f,
        };
        var hp = GptOssHyperparams.FromGgufMetadata(
            metadata, "gpt-oss", numLayer: 24, embedDim: 2880, numHeads: 64, numHeadsKv: 8,
            headDim: 64, numExperts: 32, numExpertsUsed: 4, vocabSize: 201088);

        Assert.Equal(150000f, hp.RopeFreqBaseSwa);
    }
}
