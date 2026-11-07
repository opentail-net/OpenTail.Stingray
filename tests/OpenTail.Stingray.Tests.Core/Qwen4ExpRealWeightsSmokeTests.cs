namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// First real-weight execution attempt for qwen4exp (Qwen 3.8 Flash Next), per docs/2-coverage/
/// 2026-10-02-tensorsharp-takeaways-plan.md's "Pending: Level 4 paged real-weight execution" item.
/// Bypasses the engine's admission gate deliberately (qwen4exp is NotAdmitted) by constructing
/// Qwen4ExpForwardPass directly, the same way Qwen4ExpAlphaTests does with synthetic tensors --
/// here with the real GGUF instead. Not a parity test: there is no reference output pinned yet.
///
/// <para>Found 2026-11-07: tensor loading had two wrong tensor names (fixed, see
/// Qwen4ExpTensorSet.cs's "output_hc_norm/_down/_up" comment). Then a REAL crash: <see
/// cref="Qwen4ExpForwardPass"/>'s <c>AsFloatSpan</c> helper cast every weight tensor as Float32
/// with no dtype check -- this checkpoint stores the hyper-connection weights as Q8_0, so the cast
/// massively overran the real (quantized, much smaller) buffer and crashed the whole process with
/// a native AccessViolationException. First fixed to fail loudly with a NotSupportedException
/// instead of corrupting memory, then (same day) actually fixed by dequantizing via the engine's
/// existing Dequantize.ToFloat32, cached per tensor pointer (AsFloatSpan's own comment has the
/// detail) -- these are all small per-layer control tensors, not the large MoE expert matrices, so
/// caching them as F32 is bounded. This test now checks prefill runs to completion with finite
/// logits -- still not a parity test (no reference output pinned), just "doesn't crash or produce
/// NaN/Inf" on a real checkpoint.</para>
/// </summary>
public sealed class Qwen4ExpRealWeightsSmokeTests
{
    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void Prefill_OnRealCheckpoint_ProducesFiniteLogits()
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

        var logits = forwardPass.Prefill(ids.ToArray());

        Assert.True(logits.Length > 0);
        Assert.All(logits.ToArray(), l => Assert.True(float.IsFinite(l)));

        int best = 0; float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        var decodeLogits = forwardPass.Forward(best, ids.Count);
        Assert.True(decodeLogits.Length > 0);
        Assert.All(decodeLogits.ToArray(), l => Assert.True(float.IsFinite(l)));
    }

    /// <summary>
    /// Numeric cross-check against llama.cpp's own eval-callback dump on the identical real
    /// checkpoint and prompt (2026-11-07 diagnostic): llama.cpp reports "number of input tokens
    /// = 5" (tokens 760, 6511, 314, 9338, 369, no BOS) and its "model.input_embed" node sum =
    /// 2.627300. If our tokenization or embedding-table dequant disagrees with llama.cpp's,
    /// nothing downstream can be right -- so this checks that FIRST, before any layer math.
    /// </summary>
    [Fact]
    public unsafe void Tokenization_And_Embedding_MatchLlamaCppDump()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = OpenTail.Stingray.Engine.Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = OpenTail.Stingray.Engine.Qwen4ExpTensorSet.Load(model, hp, totalLayers);

        var tokenizer = OpenTail.Stingray.Core.GgufTokenizer.FromGgufModel(model);
        var ids = tokenizer.Encode("The capital of France is");

        Assert.False(tokenizer.AddBosToken, "llama.cpp saw 5 input tokens for this prompt (no BOS); AddBosToken disagreeing means our tokenizer config is the divergence, not layer math.");
        Assert.Equal(new[] { 760, 6511, 314, 9338, 369 }, ids);

        int embedDim = hp.EmbedDim;
        var info = tensors.TokEmbd.Info;
        int bytesPerRow = (embedDim / OpenTail.Stingray.Core.DTypeInfo.BlockSize(info.DType)) * OpenTail.Stingray.Core.DTypeInfo.BytesPerBlock(info.DType);

        double sum = 0;
        var row = new float[embedDim];
        foreach (int token in ids)
        {
            byte* rowPtr = tensors.TokEmbd.DataPtr + (long)token * bytesPerRow;
            fixed (float* destPtr = row)
            {
                OpenTail.Stingray.Cpu.SimdKernels.DequantRow(rowPtr, destPtr, embedDim, info.DType);
            }
            foreach (float v in row) sum += v;
        }

        // llama.cpp's cb() dump prints "sum = 2.627300" for model.input_embed on this exact
        // checkpoint and prompt. Loose tolerance: it sums in a different order/precision path.
        Assert.True(Math.Abs(sum - 2.627300) < 0.01, $"embedding sum {sum:F6} vs llama.cpp's 2.627300 -- tokenization/embedding dequant divergence, not layer math.");
    }
}
