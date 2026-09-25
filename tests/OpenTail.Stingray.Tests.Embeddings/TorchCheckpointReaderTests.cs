namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>
/// <see cref="TorchCheckpointReader"/> against the same checkpoint's own safetensors: timm's
/// mobilenetv3_small_100.lamb_in1k ships both <c>pytorch_model.bin</c> (zip pickle) and <c>model.safetensors</c>, so
/// every tensor must match exactly (same F32 values, same shape, same names).
/// </summary>
public sealed class TorchCheckpointReaderTests
{
    [Fact]
    public void PytorchBin_MatchesSafetensors_BitForBit()
    {
        string? dir = BertEncoderOnnxParityTests.FindRepoDir("models/_models/hf/timm__mobilenetv3_small_100.lamb_in1k");
        Assert.SkipUnless(dir != null && File.Exists(Path.Combine(dir, "pytorch_model.bin")), "timm mobilenetv3_small_100 pytorch_model.bin not found");

        var torch = TorchCheckpointReader.Read(Path.Combine(dir!, "pytorch_model.bin"));
        using var st = SafetensorsLoader.Open(Path.Combine(dir!, "model.safetensors"));
        var stNames = st.TensorNames.ToHashSet();
        int compared = 0;
        foreach (var (name, t) in torch)
        {
            if (name.EndsWith("num_batches_tracked", StringComparison.Ordinal)) continue; // int64 BN counters: not weights
            Assert.Contains(name, stNames);
            Assert.Equal(st.GetShape(name), t.Shape);
            Assert.Equal(st.ReadF32(name), t.Data);
            compared++;
        }
        Console.WriteLine($"[TorchReader] {compared} tensors identical to model.safetensors ({torch.Count} read)");
        Assert.True(compared > 100);
    }
}
