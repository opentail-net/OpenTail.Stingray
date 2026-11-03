namespace OpenTail.Stingray.Tests.Audio;

/// <summary>A truncated whisper.cpp ggml file must fail to load, not yield a partly zero-filled tensor.</summary>
public sealed class WhisperGgmlTruncationTests
{
    private static byte[] Header()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(0x67676d6cu);
        for (int i = 0; i < 11; i++) w.Write(1);  // hparams
        w.Write(0); w.Write(0);                   // filterbank: 0 x 0
        w.Write(0);                               // vocab count
        w.Flush();
        return ms.ToArray();
    }

    private static byte[] TensorHeader(int ggmlType, int elements)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(1); w.Write(1); w.Write(ggmlType); w.Write(elements); w.Write((byte)'w');
        w.Flush();
        return ms.ToArray();
    }

    private static WhisperGgmlModel Load(params byte[][] parts)
    {
        string path = Path.Combine(Path.GetTempPath(), $"ggml-trunc-{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, parts.SelectMany(p => p).ToArray());
            return WhisperGgmlModel.Load(path);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void CompleteF32Tensor_Loads()
    {
        var model = Load(Header(), TensorHeader(0, 2), new byte[8]);
        Assert.Equal(2, model.GetTensor("w").Length);
    }

    [Fact]
    public void TruncatedF32Tensor_Throws() =>
        Assert.Throws<InvalidDataException>(() => Load(Header(), TensorHeader(0, 4), new byte[8]));

    [Fact]
    public void TruncatedF16Tensor_Throws() =>
        Assert.Throws<InvalidDataException>(() => Load(Header(), TensorHeader(1, 4), new byte[4]));

    [Fact]
    public void IncompleteTrailingTensorHeader_Throws() =>
        Assert.Throws<InvalidDataException>(() => Load(Header(), TensorHeader(0, 1), new byte[4], new byte[5]));
}
