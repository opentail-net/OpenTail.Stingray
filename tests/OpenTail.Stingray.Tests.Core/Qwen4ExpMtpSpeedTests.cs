using System.Diagnostics;
using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 4 (speed): wall-clock decode time for plain greedy against the product <see cref="MtpDecoder"/>, on the same
/// prompt, three repetitions each. Measurement only: the numbers are printed, and the test asserts that both paths produce the same
/// tokens. Real trunk and shared head; skipped when absent.
/// </summary>
public sealed class Qwen4ExpMtpSpeedTests
{
    private const string Text =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type in Mainz.";

    private const int Generate = 24;
    private const int Repetitions = 3;

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMtp() => ModelLocator.FindDeep("mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf");

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    [Fact]
    public void PlainAndMtpDecode_SameTokens_TimedOverRepetitions()
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
        int[] prompt = tok.Encode(Text).ToArray();

        int[]? reference = null;
        for (int rep = 0; rep < Repetitions; rep++)
        {
            // Plain greedy: prompt, then Generate tokens, timed from the first decode step.
            pass.ResetCache();
            var plain = new List<int>();
            ReadOnlySpan<float> lg = default;
            int pos = 0;
            foreach (int id in prompt) lg = pass.Forward(id, pos++);
            var sw = Stopwatch.StartNew();
            for (int s = 0; s < Generate; s++)
            {
                int t = Argmax(lg);
                plain.Add(t);
                lg = pass.Forward(t, pos++);
            }
            sw.Stop();
            double plainTps = Generate / sw.Elapsed.TotalSeconds;

            // Product MTP decoder from the same prompt state.
            pass.ResetCache();
            float[] lastLogits;
            {
                ReadOnlySpan<float> last = default;
                int p2 = 0;
                foreach (int id in prompt) last = pass.Forward(id, p2++);
                lastLogits = last.ToArray();
            }
            pass.PrefillMtp(prompt, 0);
            var decoder = new MtpDecoder(pass);
            decoder.Initialize(prompt.Length, lastLogits);
            var emitted = new List<int>();
            var sw2 = Stopwatch.StartNew();
            decoder.Decode(Generate, ReadOnlySpan<int>.Empty, t => emitted.Add(t), pMin: 1f, draftN: 1);
            sw2.Stop();
            double mtpTps = Generate / sw2.Elapsed.TotalSeconds;

            Console.WriteLine($"SPEED rep={rep} plain_tok_s={plainTps:F3} mtp_tok_s={mtpTps:F3} accept={decoder.AcceptanceRate:F4} drafts={decoder.TotalDraftsEmitted}");
            Assert.Equal(plain.ToArray(), emitted.Take(Generate).ToArray());
            reference ??= plain.ToArray();
        }
        Assert.NotNull(reference);
    }
}
