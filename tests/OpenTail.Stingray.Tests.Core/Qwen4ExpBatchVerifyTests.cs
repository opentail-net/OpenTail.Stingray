using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 4 (batched verify, replay form): on the real qwen4exp checkpoint, <c>BatchVerify</c> returns exactly the
/// logits that sequential <c>Forward</c> calls give, and a rollback to a position inside the block, followed by a different
/// token, matches a fresh sequential run exactly. Real trunk; skipped when absent.
/// </summary>
public sealed class Qwen4ExpBatchVerifyTests
{
    private const string Text = "The printing press was one of the most important inventions in history.";

    private static string? FindModel() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    [Fact]
    public void BatchVerify_MatchesSequentialForward_AndRollbackReplaysExactly()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);
        int[] prompt = tok.Encode(Text).ToArray();
        int P = prompt.Length;

        // Sequential reference: the greedy block, and the logits after each block token.
        pass.ResetCache();
        ReadOnlySpan<float> lg = default;
        int pos = 0;
        foreach (int id in prompt) lg = pass.Forward(id, pos++);
        var block = new int[3];
        var reference = new float[3][];
        for (int i = 0; i < 3; i++)
        {
            block[i] = Argmax(lg);
            lg = pass.Forward(block[i], pos++);
            reference[i] = lg.ToArray();
        }

        // Batched verify over the same block from the same prompt state.
        pass.ResetCache();
        pos = 0;
        foreach (int id in prompt) lg = pass.Forward(id, pos++);
        pass.BatchVerifyEnabled = true;
        Assert.True(pass.SupportsBatchVerify);
        float[][] verified = pass.BatchVerify(block, P);
        for (int i = 0; i < 3; i++)
            Assert.True(reference[i].AsSpan().SequenceEqual(verified[i]), $"BatchVerify logits differ from sequential Forward at block token {i}.");

        // Rollback inside the block: keep block[0], then feed a different token. Compare with a fresh sequential run of [prompt, block[0], alt].
        int alt = block[2];
        pass.RestoreBatchSnapshot(P + 1);
        ReadOnlySpan<float> afterRollback = pass.Forward(alt, P + 1);
        float[] rolled = afterRollback.ToArray();

        pass.ResetCache();
        pos = 0;
        foreach (int id in prompt) pass.Forward(id, pos++);
        pass.Forward(block[0], pos++);
        float[] fresh = pass.Forward(alt, pos).ToArray();
        Assert.True(fresh.AsSpan().SequenceEqual(rolled), "rollback then a different token differs from a fresh sequential run.");

        // BatchForward2 is the same contract for two tokens.
        pass.ResetCache();
        pos = 0;
        foreach (int id in prompt) pass.Forward(id, pos++);
        pass.BatchForward2(block[0], block[1], P, out ReadOnlySpan<float> l1, out ReadOnlySpan<float> l2);
        Assert.True(reference[0].AsSpan().SequenceEqual(l1), "BatchForward2 first logits differ from sequential Forward.");
        Assert.True(reference[1].AsSpan().SequenceEqual(l2), "BatchForward2 second logits differ from sequential Forward.");

        Console.WriteLine($"BATCH_VERIFY block={block.Length} prompt={P} exact=true rollback_exact=true");
    }

    private const string DecodeText =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type in Mainz.";

    private const int Generate = 24;

    /// <summary>The MTP decoder on the batched replay path (opted in) emits the plain greedy tokens, rollbacks included.</summary>
    [Fact]
    public void MtpDecoder_BatchedReplay_EmitsThePlainGreedyTokens()
    {
        string? trunkPath = FindModel();
        string? mtpPath = ModelLocator.FindDeep("mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf");
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        Assert.SkipUnless(mtpPath is not null, "Qwen3.8 MTP shared Q4_K_M head is required (1.9 GB).");

        using var trunk = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(trunk.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(trunk.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(trunk, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(trunk, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(trunk);
        using var mtp = GgufModel.Open(mtpPath!);
        pass.AttachMtpHead(Qwen4ExpMtpHead.Load(mtp, tensors, blockIndex: totalLayers, embedDim: hp.EmbedDim, hc: hp.HyperConnectionCount));
        int[] prompt = tok.Encode(DecodeText).ToArray();

        var plain = new List<int>();
        {
            ReadOnlySpan<float> lg = default;
            int pos = 0;
            foreach (int id in prompt) lg = pass.Forward(id, pos++);
            for (int s = 0; s < Generate; s++)
            {
                int t = Argmax(lg);
                plain.Add(t);
                lg = pass.Forward(t, pos++);
            }
        }

        pass.ResetCache();
        float[] lastLogits;
        {
            ReadOnlySpan<float> last = default;
            int pos = 0;
            foreach (int id in prompt) last = pass.Forward(id, pos++);
            lastLogits = last.ToArray();
        }
        pass.PrefillMtp(prompt, 0);
        pass.BatchVerifyEnabled = true;

        var decoder = new MtpDecoder(pass);
        decoder.Initialize(prompt.Length, lastLogits);
        var emitted = new List<int>();
        decoder.Decode(Generate, ReadOnlySpan<int>.Empty, t => emitted.Add(t), pMin: 1f, draftN: 1);

        Console.WriteLine($"BATCH_DECODER emitted={emitted.Count} drafts={decoder.TotalDraftsEmitted} accepted={decoder.TotalDraftsAccepted}");
        Assert.Equal(plain.ToArray(), emitted.Take(Generate).ToArray());
    }
}
