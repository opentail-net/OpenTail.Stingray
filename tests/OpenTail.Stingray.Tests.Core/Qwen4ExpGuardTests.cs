using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

public sealed class Qwen4ExpGuardTests
{
    [Fact]
    public void ForwardPass_AcceptsRealConfigs_WithQsaIndexer()
    {
        // Real qwen4exp checkpoints declare indexer_top_k; the QSA mixer now implements IMRoPE, indexer-K caching,
        // K-pool selection and the PLE n-gram table. Construction should succeed without throwing NotSupportedException.
        var hp = new Qwen4ExpHyperparams
        {
            IndexerTopK = 2048,
            IndexerKPool = 4,
            EmbedDim = 128,
            NumHeads = 4,
            NumHeadsKv = 2,
            HeadDim = 32,
            NumLayer = 1
        };
        // Verify no refusal exception is thrown on construction with non-zero IndexerTopK
        var ex = Record.Exception(() => new Qwen4ExpForwardPass(null!, hp, null!));
        Assert.IsNotType<NotSupportedException>(ex);
    }
}
