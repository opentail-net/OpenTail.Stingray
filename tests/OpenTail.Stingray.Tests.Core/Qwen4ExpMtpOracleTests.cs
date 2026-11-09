using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 4 (per-step oracle): dumps the MTP head's inputs and Stingray's head logits for the speed-test prompt
/// and 24 greedy steps, then compares them with the reference head logits written by the local driver
/// (<c>llama-mtp-check</c>, built from the unsloth fork, not part of this repository). Skipped unless
/// <c>STINGRAY_MTP_ORACLE_DIR</c> is set, the checkpoint is present, and the reference file exists. The verdict uses the
/// token criterion of the plan (D1): the reference top-1 must be Stingray's top-1, or within 0.12 logits of it in Stingray's top-5.
/// </summary>
public sealed class Qwen4ExpMtpOracleTests
{
    private const string Text =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type in Mainz.";

    private const int Generate = 24;
    private const float Tolerance = 0.12f;

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMtp() => ModelLocator.FindDeep("mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf");

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    [Fact]
    public void MtpHead_PerStep_MatchesTheReferenceWithinTolerance()
    {
        string? dir = Environment.GetEnvironmentVariable("STINGRAY_MTP_ORACLE_DIR");
        Assert.SkipUnless(!string.IsNullOrWhiteSpace(dir), "Set STINGRAY_MTP_ORACLE_DIR to the oracle work folder.");
        string? trunkPath = FindTrunk();
        string? mtpPath = FindMtp();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        Assert.SkipUnless(mtpPath is not null, "Qwen3.8 MTP shared Q4_K_M head is required (1.9 GB).");
        Directory.CreateDirectory(dir!);

        using var trunk = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(trunk.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(trunk.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(trunk, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(trunk, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(trunk);
        using var mtp = GgufModel.Open(mtpPath!);
        pass.AttachMtpHead(Qwen4ExpMtpHead.Load(mtp, tensors, blockIndex: totalLayers, embedDim: hp.EmbedDim, hc: hp.HyperConnectionCount));

        // Trunk pass: the token at each position, then its residual hidden state (the MTP input for the next position).
        var seq = tok.Encode(Text).ToList();
        int promptLength = seq.Count;
        var steps = new List<(int Token, int Pos, float[] Hidden, float[] Logits)>();
        pass.ResetCache();
        ReadOnlySpan<float> lg = default;
        for (int p = 0; p < promptLength + Generate - 1; p++)
        {
            if (p >= seq.Count) seq.Add(Argmax(lg));
            lg = pass.Forward(seq[p], p);
            int next = p + 1 < promptLength ? seq[p + 1] : Argmax(lg);
            if (p + 1 >= seq.Count) seq.Add(next);
            var hPrev = pass.LastHidden.ToArray();
            ReadOnlySpan<float> mtpLogits = pass.MtpForward(seq[p + 1], p + 1, hPrev);
            steps.Add((seq[p + 1], p + 1, hPrev, mtpLogits.ToArray()));
        }

        // Inputs for the reference driver: int32 n_steps, n_hidden, n_vocab, then per step int32 token, int32 pos, float hidden.
        string stepsPath = Path.Combine(dir!, "mtp_steps.bin");
        using (var w = new BinaryWriter(File.Create(stepsPath)))
        {
            w.Write(steps.Count);
            w.Write(steps[0].Hidden.Length);
            w.Write(steps[0].Logits.Length);
            foreach (var s in steps)
            {
                w.Write(s.Token);
                w.Write(s.Pos);
                foreach (float f in s.Hidden) w.Write(f);
            }
        }
        using (var w = new BinaryWriter(File.Create(Path.Combine(dir!, "mtp_steps_stingray.bin"))))
        {
            foreach (var s in steps) foreach (float f in s.Logits) w.Write(f);
        }
        Console.WriteLine($"MTP_ORACLE wrote {steps.Count} steps to {stepsPath}");

        string refPath = Path.Combine(dir!, "mtp_steps_ref.bin");
        Assert.SkipUnless(File.Exists(refPath), $"reference logits not found: {refPath} (run llama-mtp-check on mtp_steps.bin).");

        int vocab = steps[0].Logits.Length;
        var refFlat = new float[steps.Count * vocab];
        using (var r = new BinaryReader(File.OpenRead(refPath)))
        {
            for (int i = 0; i < refFlat.Length; i++) refFlat[i] = r.ReadSingle();
        }

        int failures = 0;
        double worstCos = 1.0;
        float worstAbs = 0f;
        for (int s = 0; s < steps.Count; s++)
        {
            ReadOnlySpan<float> ours = steps[s].Logits;
            ReadOnlySpan<float> theirs = refFlat.AsSpan(s * vocab, vocab);
            int top = Argmax(ours);
            int refTop = Argmax(theirs);
            bool ok = refTop == top || (Top5Contains(ours, refTop) && ours[top] - ours[refTop] <= Tolerance);
            if (!ok) failures++;

            double dot = 0, na = 0, nb = 0;
            float maxAbs = 0f;
            for (int i = 0; i < vocab; i++)
            {
                dot += ours[i] * theirs[i];
                na += ours[i] * ours[i];
                nb += theirs[i] * theirs[i];
                maxAbs = Math.Max(maxAbs, Math.Abs(ours[i] - theirs[i]));
            }
            double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb));
            worstCos = Math.Min(worstCos, cos);
            worstAbs = Math.Max(worstAbs, maxAbs);
            Console.WriteLine($"MTP_ORACLE step={s} pos={steps[s].Pos} stingray_top={top} ref_top={refTop} ok={ok} cos={cos:F6} max_abs={maxAbs:F4}");
        }
        Console.WriteLine($"MTP_ORACLE_SUMMARY steps={steps.Count} failures={failures} worst_cos={worstCos:F6} worst_max_abs={worstAbs:F4}");
        Assert.Equal(0, failures);
    }

    private static bool Top5Contains(ReadOnlySpan<float> logits, int id)
    {
        var top = new int[5];
        for (int k = 0; k < 5; k++)
        {
            int best = -1;
            for (int i = 0; i < logits.Length; i++)
            {
                if (Array.IndexOf(top, i, 0, k) >= 0) continue;
                if (best < 0 || logits[i] > logits[best]) best = i;
            }
            top[k] = best;
        }
        return Array.IndexOf(top, id) >= 0;
    }
}
