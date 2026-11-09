using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

using static OpenTail.Stingray.Tests.Core.Qwen4ExpTestSupport;

/// <summary>
/// Qwen 3.8 plan, Phase 4 (rollback, lossless): greedy speculative decoding with the shared MTP head must give exactly the plain
/// greedy tokens. A chain of K drafts is verified by the trunk; on rejection the trunk and head state are restored from a
/// snapshot and the accepted prefix is re-fed. Test-side loop only, no product change. Real trunk and head; skipped when absent.
/// </summary>
public sealed class Qwen4ExpSpeculativeLosslessTests
{
    private const string Text =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type in Mainz.";

    private const int Generate = 24;
    private const int K = 3;

    private static string? FindTrunk() => ModelLocator.FindDeep(Qwen4ExpTestSupport.TrunkFileName);

    private static string? FindMtp() => ModelLocator.FindDeep("mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf");


    [Fact]
    public void SpeculativeGreedy_WithRollback_EqualsPlainGreedy()
    {
        string? trunkPath = FindTrunk();
        string? mtpPath = FindMtp();
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

        int hcDim = hp.HyperConnectionCount * hp.EmbedDim;
        int[] prompt = tok.Encode(Text).ToArray();
        int target = prompt.Length + Generate;

        // Plain greedy reference: trunk only.
        var plain = prompt.ToList();
        {
            var s0 = pass.CreateSequenceState();
            ReadOnlySpan<float> lg = default;
            for (int q = 0; q < prompt.Length; q++) lg = pass.Forward(plain[q], q, s0);
            for (int s = 0; s < Generate; s++)
            {
                int t = Argmax(lg);
                plain.Add(t);
                lg = pass.Forward(t, plain.Count - 1, s0);
            }
        }

        // Speculative greedy with rollback.
        var S = pass.CreateSequenceState();
        var seq = prompt.ToList();
        var hid = new float[target + K + 4][];
        var hOut = new float[hcDim];
        int proc = 0;
        float[]? lastLogits = null;
        int drafts = 0, accepted = 0, rollbacks = 0;

        // Feeds seq tokens [proc, upto) through the trunk, refreshing the head's cache from the same positions.
        void Feed(int upto)
        {
            while (proc < upto)
            {
                lastLogits = pass.Forward(seq[proc], proc, S).ToArray();
                hid[proc] = pass.LastHcResidual.ToArray();
                if (proc >= 1) pass.ForwardMtp(seq[proc], hid[proc - 1], proc, S, hOut);
                proc++;
            }
        }

        Feed(prompt.Length);
        while (seq.Count < target)
        {
            int t = Argmax(lastLogits!);
            seq.Add(t);
            int q = seq.Count - 1; // position of t; trunk has processed [0, q)
            var snap = S.Snapshot();

            // Draft a chain of K tokens with the head, starting from t and the hidden state of position q-1.
            var d = new int[K];
            int cur = t;
            var hcur = hid[q - 1];
            var hm = new float[hcDim];
            for (int i = 0; i < K; i++)
            {
                d[i] = Argmax(pass.ForwardMtp(cur, hcur, q + i, S, hm));
                hcur = hm.ToArray();
                cur = d[i];
            }
            drafts += K;

            // Verify: the trunk takes t, d[0], ..., d[K-2] at q, q+1, ..., q+K-1.
            var inputs = new int[K];
            inputs[0] = t;
            for (int i = 1; i < K; i++) inputs[i] = d[i - 1];
            var vlog = new float[K][];
            var vhid = new float[K][];
            for (int i = 0; i < K; i++)
            {
                vlog[i] = pass.Forward(inputs[i], q + i, S).ToArray();
                vhid[i] = pass.LastHcResidual.ToArray();
            }

            int a = 0;
            while (a < K && Argmax(vlog[a]) == d[a]) a++;
            accepted += a;
            for (int i = 0; i < a; i++) seq.Add(d[i]);

            if (a >= K - 1)
            {
                // The trunk processed exactly the accepted tokens (inputs[0..K-1] are all accepted when a >= K-1).
                for (int i = 0; i < K; i++) hid[q + i] = vhid[i];
                proc = q + K;
                if (a == K - 1) lastLogits = vlog[K - 1];
                else Feed(seq.Count); // all K accepted: feed the last accepted draft for its logits
            }
            else
            {
                // A rejection inside the chain: the trunk and head processed rejected drafts. Restore and re-feed.
                rollbacks++;
                S.Restore(snap);
                proc = q;
                Feed(seq.Count);
            }
        }

        var spec = seq.Take(target).ToArray();
        var plainTarget = plain.Take(target).ToArray();
        Console.WriteLine($"SPEC drafts={drafts} accepted={accepted} rate={(double)accepted / drafts:F4} rollbacks={rollbacks}");
        Console.WriteLine($"SPEC_TEXT {tok.Decode(spec.Skip(prompt.Length).ToArray())}");
        Assert.Equal(plainTarget, spec);
    }
}
