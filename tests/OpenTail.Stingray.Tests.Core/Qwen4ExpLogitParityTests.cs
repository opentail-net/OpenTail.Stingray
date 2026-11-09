using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 6.4 (logit distributions): teacher-forces the pinned llama.cpp reference's greedy tokens through the
/// Stingray trunk and compares the full distribution at each step. The reference is the local <c>llama-logit-dump</c> driver on
/// <c>bed0a8566</c>, which is not part of this repository. Skipped unless <c>STINGRAY_LOGIT_PARITY_DIR</c> is set and the
/// checkpoint is present. The first run writes <c>logit_prompt.bin</c>; after the reference writes <c>logit_ref.bin</c> and
/// <c>logit_tokens_ref.bin</c>, the second run compares. Verdict: the D1 token criterion (the reference top-1 must be Stingray's
/// top-1, or within 0.12 logits of it in Stingray's top-5), with total variation and cosine reported per step.
/// </summary>
public sealed class Qwen4ExpLogitParityTests
{
    private const string Prompt =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type in Mainz.";

    private const int Generate = 24;
    private const float Tolerance = 0.12f;

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
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

    private static double[] Softmax(ReadOnlySpan<float> logits)
    {
        float max = logits[Argmax(logits)];
        var p = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) { p[i] = Math.Exp(logits[i] - max); sum += p[i]; }
        for (int i = 0; i < p.Length; i++) p[i] /= sum;
        return p;
    }

    [Fact]
    public void TrunkLogits_TeacherForced_MatchTheReferenceWithinTolerance()
    {
        string? dir = Environment.GetEnvironmentVariable("STINGRAY_LOGIT_PARITY_DIR");
        Assert.SkipUnless(!string.IsNullOrWhiteSpace(dir), "Set STINGRAY_LOGIT_PARITY_DIR to the parity work folder.");
        string? trunkPath = FindTrunk();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        Directory.CreateDirectory(dir!);

        using var model = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);

        int[] prompt = tok.Encode(Prompt).ToArray();
        string promptPath = Path.Combine(dir!, "logit_prompt.bin");
        using (var w = new BinaryWriter(File.Create(promptPath)))
        {
            w.Write(prompt.Length);
            w.Write(Generate);
            foreach (int id in prompt) w.Write(id);
        }
        Console.WriteLine($"LOGIT_PARITY wrote {prompt.Length} prompt tokens, {Generate} steps to {promptPath}");

        string refLogits = Path.Combine(dir!, "logit_ref.bin");
        string refTokens = Path.Combine(dir!, "logit_tokens_ref.bin");
        Assert.SkipUnless(File.Exists(refLogits) && File.Exists(refTokens),
            $"reference output not found in {dir} (run llama-logit-dump on logit_prompt.bin).");

        // Reference greedy tokens, teacher-forced below so each step compares the same context.
        int[] forced = new int[Generate];
        using (var r = new BinaryReader(File.OpenRead(refTokens)))
        {
            for (int s = 0; s < Generate; s++) forced[s] = r.ReadInt32();
        }

        pass.ResetCache();
        ReadOnlySpan<float> lg = default;
        for (int p = 0; p < prompt.Length; p++) lg = pass.Forward(prompt[p], p);

        int vocab = lg.Length;
        var refFlat = new float[Generate * vocab];
        using (var r = new BinaryReader(File.OpenRead(refLogits)))
        {
            for (int i = 0; i < refFlat.Length; i++) refFlat[i] = r.ReadSingle();
        }

        int failures = 0;
        double worstCos = 1.0, worstTv = 0.0;
        for (int s = 0; s < Generate; s++)
        {
            if (s > 0) lg = pass.Forward(forced[s - 1], prompt.Length + s - 1);
            ReadOnlySpan<float> ours = lg;
            ReadOnlySpan<float> theirs = refFlat.AsSpan(s * vocab, vocab);

            int top = Argmax(ours);
            int refTop = Argmax(theirs);
            bool ok = refTop == top || (Top5Contains(ours, refTop) && ours[top] - ours[refTop] <= Tolerance);
            if (!ok) failures++;

            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < vocab; i++)
            {
                dot += ours[i] * theirs[i];
                na += ours[i] * ours[i];
                nb += theirs[i] * theirs[i];
            }
            double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb));
            var pOurs = Softmax(ours);
            var pTheirs = Softmax(theirs);
            double tv = 0;
            for (int i = 0; i < vocab; i++) tv += Math.Abs(pOurs[i] - pTheirs[i]);
            tv /= 2;
            worstCos = Math.Min(worstCos, cos);
            worstTv = Math.Max(worstTv, tv);
            Console.WriteLine($"LOGIT_PARITY step={s} forced={forced[s]} stingray_top={top} ref_top={refTop} ok={ok} cos={cos:F6} tv={tv:F6}");
        }
        Console.WriteLine($"LOGIT_PARITY_SUMMARY steps={Generate} failures={failures} worst_cos={worstCos:F6} worst_tv={worstTv:F6}");
        Assert.Equal(0, failures);
    }
}
