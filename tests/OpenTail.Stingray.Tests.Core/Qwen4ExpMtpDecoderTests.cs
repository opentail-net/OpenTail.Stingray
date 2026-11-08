using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 4 (product path): the product <see cref="MtpDecoder"/>, driving the real qwen4exp forward pass through
/// <see cref="IForwardPass"/>, emits exactly the plain greedy tokens. Real trunk and shared head; skipped when absent.
/// </summary>
public sealed class Qwen4ExpMtpDecoderTests
{
    private const string Text =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type in Mainz.";

    private const int Generate = 24;

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMtp() => ModelLocator.FindDeep("mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf");

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    [Fact]
    public void MtpDecoder_OnTheQwenPass_EmitsThePlainGreedyTokens()
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

        // The head is attached before the prompt, so the trunk hiddens of the prompt positions are recorded.
        using var mtp = GgufModel.Open(mtpPath!);
        pass.AttachMtpHead(Qwen4ExpMtpHead.Load(mtp, tensors, blockIndex: totalLayers, embedDim: hp.EmbedDim, hc: hp.HyperConnectionCount));
        Assert.True(pass.HasMtpHead);

        int[] prompt = tok.Encode(Text).ToArray();

        // Plain greedy reference through the same interface the decoder uses.
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

        // Product decoder: prompt forward, prompt replay into the head's cache, then MtpDecoder from that state.
        pass.ResetCache();
        float[] lastLogits;
        {
            ReadOnlySpan<float> last = default;
            int pos = 0;
            foreach (int id in prompt) last = pass.Forward(id, pos++);
            lastLogits = last.ToArray(); // copied before PrefillMtp reuses the logits buffer
        }
        pass.PrefillMtp(prompt, 0);

        var decoder = new MtpDecoder(pass);
        decoder.Initialize(prompt.Length, lastLogits);
        var emitted = new List<int>();
        decoder.Decode(Generate, ReadOnlySpan<int>.Empty, t => emitted.Add(t), pMin: 1f, draftN: 1);

        Console.WriteLine($"MTP_DECODER emitted={emitted.Count} drafts={decoder.TotalDraftsEmitted} accepted={decoder.TotalDraftsAccepted} rate={decoder.AcceptanceRate:F4}");
        Console.WriteLine($"MTP_DECODER_TEXT {tok.Decode(emitted.ToArray())}");
        Assert.Equal(plain.Take(Generate).ToArray(), emitted.Take(Generate).ToArray());
    }
}
