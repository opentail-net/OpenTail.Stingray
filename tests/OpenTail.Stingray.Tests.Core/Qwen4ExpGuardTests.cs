using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

public sealed class Qwen4ExpGuardTests
{
    [Fact]
    public void ForwardPass_RefusesRealConfigs_WithQsaIndexer()
    {
        // A real qwen4exp checkpoint declares indexer_top_k; the QSA mixer lacks RoPE, indexer selection and the
        // PLE n-gram table, so construction must refuse before any tensor is read.
        var hp = new Qwen4ExpHyperparams { IndexerTopK = 2048 };
        var ex = Assert.Throws<NotSupportedException>(() => new Qwen4ExpForwardPass(null!, hp, null!));
        Assert.Contains("QSA", ex.Message);
    }
}
