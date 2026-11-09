using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

using static OpenTail.Stingray.Tests.Core.Qwen4ExpTestSupport;

/// <summary>
/// Qwen 3.8 plan, Phase 3 (Job 3): on a real qwen4exp cache, a second turn that starts after earlier positions keeps a turn-start
/// snapshot, and the snapshot's held bytes are measured. Real checkpoint; skipped when absent.
/// </summary>
public sealed class Qwen4ExpTurnSnapshotTests
{
    private const string FirstTurn = "The capital of France is";
    private const string SecondTurn = " Paris. The capital of Italy is";

    private static string? FindModel() => ModelLocator.FindDeep(Qwen4ExpTestSupport.TrunkFileName);


    [Fact]
    public void SecondTurn_KeepsATurnStartSnapshot_WithMeasuredHeldBytes()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var batched = new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] first = tok.Encode(FirstTurn).ToArray();
        int[] second = tok.Encode(SecondTurn).ToArray();
        var cache = batched.CreateCache();
        var held = (ISnapshotAccountedSequenceKvCache)cache;

        // The first turn prefills, then generates a few tokens. A new turn starts after those generated positions, which is
        // what makes the cache keep a turn-start snapshot (a prefill that continues the last prefill does not).
        var logits = batched.PrefillWithCache(first, cache, 0).ToArray();
        int pos = first.Length;
        for (int s = 0; s < 3; s++)
        {
            int t = Argmax(logits);
            logits = batched.BatchForwardMulti(new[] { t }, new[] { pos }, new[] { cache })[0];
            pos++;
        }
        long heldAfterFirst = held.HeldSnapshotBytes;
        batched.PrefillWithCache(second, cache, pos);
        long heldAfterSecond = held.HeldSnapshotBytes;

        long fixedBytes = batched.FixedBytesPerSequence;
        long perToken = batched.KvBytesPerToken;
        Console.WriteLine($"TURN_SNAPSHOT first={first.Length} generated=3 second={second.Length} held_after_first={heldAfterFirst} held_after_second={heldAfterSecond} fixed={fixedBytes} per_token={perToken}");

        Assert.Equal(0, heldAfterFirst);
        Assert.True(heldAfterSecond > 0, "the second turn must keep a turn-start snapshot");
        Assert.True(heldAfterSecond >= fixedBytes,
            $"the snapshot holds {heldAfterSecond} bytes, less than the fixed state {fixedBytes}");
        cache.Dispose();
    }
}
