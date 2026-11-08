using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Job 3 (Qwen 3.8 plan): the per-token byte figure equals the rows the forward pass writes per token, including the raw and
/// pooled indexer keys and the PLE history that the old figure left out. Synthetic: rows are simulated with the sizes
/// <c>QsaAppend</c> uses.
/// </summary>
public sealed class Qwen4ExpPerTokenBytesTests
{
    [Theory]
    [InlineData(4, 4, 4, 2, 8, 4, true)]    // every attention layer has an indexer, PLE present
    [InlineData(4, 2, 4, 2, 8, 4, false)]   // only some attention layers have an indexer, no PLE
    [InlineData(3, 0, 2, 4, 6, 3, true)]    // no indexers at all
    [InlineData(2, 2, 1, 8, 16, 8, false)]
    public void PerTokenBytes_EqualsTheRowsTheForwardPassWrites(
        int attentionLayers, int indexerLayers, int numHeadsKv, int headDim, int indexerKeyLength, int indexerKPool, bool hasPle)
    {
        // Simulate n tokens exactly as QsaAppend writes them: K and V rows for each attention layer; a raw key per token and
        // one pooled key per block for each layer with an indexer; one PLE history entry per token.
        int tokens = indexerKPool * 3;
        long written = 0;
        for (int t = 0; t < tokens; t++)
        {
            for (int a = 0; a < attentionLayers; a++)
            {
                written += 2L * numHeadsKv * headDim * sizeof(float);
                if (a < indexerLayers)
                {
                    written += (long)indexerKeyLength * sizeof(float);            // raw key, this token
                    if ((t + 1) % indexerKPool == 0)
                        written += (long)indexerKeyLength * sizeof(float);        // pooled key, once per block
                }
            }
            if (hasPle) written += sizeof(int);
        }

        long perToken = Qwen4ExpSequenceState.PerTokenBytes(
            attentionLayers, indexerLayers, numHeadsKv, headDim, indexerKeyLength, indexerKPool, hasPle);

        Assert.Equal(written, tokens * perToken);
    }

    [Fact]
    public void PooledKey_IsAmortisedAndRoundedUp()
    {
        // 3 floats = 12 bytes per key; one pooled key per 4 tokens. Rounded up, a token is charged 3 bytes for the pooled key.
        long withPooled = Qwen4ExpSequenceState.PerTokenBytes(1, 1, 1, 1, 3, 4, false);
        long withoutPooled = 2L * 1 * 1 * sizeof(float) + 3 * sizeof(float);
        Assert.Equal(withoutPooled + 3, withPooled);
    }

    [Fact]
    public void PleHistory_CostsOneIntPerToken_OnlyWhenPresent()
    {
        long without = Qwen4ExpSequenceState.PerTokenBytes(2, 0, 1, 1, 0, 1, hasPle: false);
        long with = Qwen4ExpSequenceState.PerTokenBytes(2, 0, 1, 1, 0, 1, hasPle: true);
        Assert.Equal(sizeof(int), with - without);
    }
}
