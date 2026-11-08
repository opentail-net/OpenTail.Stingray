using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Qwen 3.8 plan, Phase 4 (acceptance rate, real trunk and shared MTP head): at each greedy position, the head's draft for the
/// next token is compared with the trunk's own next greedy token. Nothing is rolled back and no output changes, so this measures
/// head quality only. The draft convention matches <see cref="Qwen4ExpForwardPass.ForwardMtp"/>: the token at position p with the
/// trunk's hidden state at p-1 predicts the token at p+1. Skipped when the trunk or the head is absent.
/// </summary>
public sealed class Qwen4ExpMtpAcceptanceTests
{
    private const string Text =
        "The printing press was one of the most important inventions in history. Johannes Gutenberg developed movable type in Mainz.";

    private const int DecodeSteps = 40;

    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMtp() => ModelLocator.FindDeep("mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf");

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    [Fact]
    public void MtpDraft_AcceptanceRate_OnTheRealTrunkAndHead()
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
        var head = Qwen4ExpMtpHead.Load(mtp, tensors, blockIndex: totalLayers, embedDim: hp.EmbedDim, hc: hp.HyperConnectionCount);
        pass.AttachMtpHead(head);

        int hcDim = hp.HyperConnectionCount * hp.EmbedDim;
        var hOut = new float[hcDim];
        var state = pass.CreateSequenceState();

        var tokens = tok.Encode(Text).ToList();
        int prompt = tokens.Count;
        Assert.True(prompt >= 16, $"the prompt is {prompt} tokens");

        // Prompt: trunk forward at every position, and the head's cache filled with its drafts (the drafts are not scored).
        float[]? hPrev = null;
        ReadOnlySpan<float> logits = default;
        for (int q = 0; q < prompt; q++)
        {
            if (q >= 1) pass.ForwardMtp(tokens[q], hPrev!, q, state, hOut);
            logits = pass.Forward(tokens[q], q, state);
            hPrev = pass.LastHcResidual.ToArray();
        }

        // Decode: the trunk's greedy next token is scored against the draft made one step earlier.
        int next = Argmax(logits);
        int measured = 0, accepted = 0;
        var generated = new System.Text.StringBuilder();
        for (int step = 0; step < DecodeSteps; step++)
        {
            int q = tokens.Count;
            tokens.Add(next);
            int draft = Argmax(pass.ForwardMtp(next, hPrev!, q, state, hOut));
            logits = pass.Forward(next, q, state);
            hPrev = pass.LastHcResidual.ToArray();
            next = Argmax(logits);
            measured++;
            if (draft == next) accepted++;
        }

        for (int i = prompt; i < tokens.Count; i++) generated.Append(tok.Decode(new[] { tokens[i] }));
        double rate = (double)accepted / measured;
        Console.WriteLine($"MTP_ACCEPT prompt_tokens={prompt} measured={measured} accepted={accepted} rate={rate:F4}");
        Console.WriteLine($"MTP_TEXT {generated}");
        Assert.True(rate > 0.05, $"acceptance {rate:F4} is no better than a weak head; see the notes in the plan");
    }
}
