using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Phase 6.2 (Qwen 3.8 plan): the PLE gather reads only the head rows a token selects from <c>per_layer_token_embd</c>,
/// not the table. Counted by <see cref="Qwen4ExpForwardPass.PleGatherRows"/> and <see cref="Qwen4ExpForwardPass.PleGatherBytes"/>
/// (logical rows and bytes, not OS pages). Real checkpoint, opt-in: skipped visibly when the UD-IQ1_S trunk is absent.
/// </summary>
public sealed class Qwen4ExpPleTouchTests
{
    private static string? FindTrunk() => ModelLocator.FindDeep("Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void DecodeToken_TouchesOnlyItsGatheredPleRows()
    {
        string? trunkPath = FindTrunk();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tok = GgufTokenizer.FromGgufModel(model);
        long tableBytes = pass.PleTableBytes;
        Assert.True(tableBytes > 0, "per_layer_token_embd.weight must be present for this test.");

        int[] prompt = tok.Encode("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n").ToArray();
        var state = pass.CreateSequenceState();
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in prompt) logits = pass.Forward(id, pos++, state);
        long prefillRows = pass.PleGatherRows;

        var rowsPerDecodeToken = new List<long>();
        var bytesPerDecodeToken = new List<long>();
        for (int s = 0; s < 5; s++)
        {
            long rows0 = pass.PleGatherRows, bytes0 = pass.PleGatherBytes;
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            logits = pass.Forward(best, pos++, state);
            rowsPerDecodeToken.Add(pass.PleGatherRows - rows0);
            bytesPerDecodeToken.Add(pass.PleGatherBytes - bytes0);
        }

        Console.WriteLine($"PLE table bytes={tableBytes} prompt_tokens={prompt.Length} prefill_rows={prefillRows}");
        Console.WriteLine($"PLE decode rows per token={string.Join(",", rowsPerDecodeToken)} bytes per token={string.Join(",", bytesPerDecodeToken)}");

        // Every decode token gathers the same head rows (layers with PLE x heads), and those bytes are a sliver of the table.
        Assert.True(rowsPerDecodeToken.Distinct().Count() == 1, "decode tokens gathered different row counts");
        Assert.True(rowsPerDecodeToken[0] > 0, "a decode token must gather PLE rows");
        Assert.True(bytesPerDecodeToken[0] * 100 < tableBytes,
            $"one decode token reads {bytesPerDecodeToken[0]} bytes of a {tableBytes}-byte table (expected under 1%)");
    }
}
