namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// First real-weight execution attempt for qwen4exp (Qwen 3.8 Flash Next), per docs/2-coverage/
/// 2026-10-02-tensorsharp-takeaways-plan.md's "Pending: Level 4 paged real-weight execution" item.
/// Bypasses the engine's admission gate deliberately (qwen4exp is NotAdmitted) by constructing
/// Qwen4ExpForwardPass directly, the same way Qwen4ExpAlphaTests does with synthetic tensors --
/// here with the real GGUF instead. Not a parity test: there is no reference output pinned yet.
///
/// <para>Found 2026-11-07: tensor loading had two wrong tensor names (fixed, see
/// Qwen4ExpTensorSet.cs's "output_hc_norm/_down/_up" comment) and metadata/loader worked once that
/// was fixed. Then a REAL crash: <see cref="Qwen4ExpForwardPass"/>'s <c>AsFloatSpan</c> helper cast
/// every weight tensor as Float32 with no dtype check -- this checkpoint stores the hyper-connection
/// weights as Q8_0, so the cast massively overran the real (quantized, much smaller) buffer and
/// crashed the whole process with a native AccessViolationException. Fixed to fail loudly with a
/// NotSupportedException instead (AsFloatSpan's own comment has the detail) -- dequantizing these
/// tensors is real, not-yet-done follow-up work, so this test currently documents that exact,
/// confirmed blocker rather than asserting a result that can't be produced yet.</para>
/// </summary>
public sealed class Qwen4ExpRealWeightsSmokeTests
{
    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void Prefill_OnRealCheckpoint_RefusesQuantizedHcWeightsInsteadOfCorruptingMemory()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = OpenTail.Stingray.Engine.Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = OpenTail.Stingray.Engine.Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var forwardPass = new OpenTail.Stingray.Engine.Qwen4ExpForwardPass(model, hp, tensors);

        var tokenizer = OpenTail.Stingray.Core.GgufTokenizer.FromGgufModel(model);
        var ids = new List<int>();
        if (tokenizer.AddBosToken) ids.Add(tokenizer.BosTokenId);
        ids.AddRange(tokenizer.Encode("The capital of France is"));

        var ex = Assert.Throws<NotSupportedException>(() => forwardPass.Prefill(ids.ToArray()));
        Assert.Contains("Q8_0", ex.Message);
    }
}
