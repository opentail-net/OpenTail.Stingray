using System.Text;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// `gguf-transplant` copies a base GGUF and takes regex-matched tensors from a donor. These build tiny synthetic
/// GGUFs (no model files needed, so nothing can silently skip) and pin: the replaced tensor carries the donor's
/// dtype and bytes, every other tensor and all metadata survive byte-for-byte, offsets stay aligned, and the
/// failure modes (dimension mismatch, no match, unsafe output path) fail cleanly without writing a file.
/// </summary>
public sealed class GgufTransplantCommandTests : IDisposable
{
    private const int GgmlF32 = 0, GgmlF16 = 1;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ot-transplant-" + Guid.NewGuid().ToString("N"));

    public GgufTransplantCommandTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } }

    private sealed record T(string Name, int Type, long[] Dims, byte[] Data);

    /// <summary>Minimal GGUF v3 writer: three KV entries (string, uint32, uint32) and the given tensors, 32-byte aligned.</summary>
    private static void WriteGguf(string path, IReadOnlyList<T> tensors, uint answer = 42)
    {
        const int alignment = 32;
        using var ms = new MemoryStream();
        var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        void Str(string s) { var b = Encoding.UTF8.GetBytes(s); w.Write((ulong)b.Length); w.Write(b); }

        w.Write(Encoding.ASCII.GetBytes("GGUF"));
        w.Write(3u);
        w.Write((ulong)tensors.Count);
        w.Write(3UL);
        Str("general.architecture"); w.Write(8u); Str("test");
        Str("general.alignment"); w.Write(4u); w.Write((uint)alignment);
        Str("test.answer"); w.Write(4u); w.Write(answer);

        var offsets = new long[tensors.Count];
        long cursor = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = (cursor + alignment - 1) / alignment * alignment;
            cursor = offsets[i] + tensors[i].Data.Length;
        }
        for (int i = 0; i < tensors.Count; i++)
        {
            Str(tensors[i].Name);
            w.Write((uint)tensors[i].Dims.Length);
            foreach (long d in tensors[i].Dims) w.Write((ulong)d);
            w.Write((uint)tensors[i].Type);
            w.Write((ulong)offsets[i]);
        }
        while (ms.Position % alignment != 0) w.Write((byte)0);
        long dataStart = ms.Position;
        for (int i = 0; i < tensors.Count; i++)
        {
            while (ms.Position < dataStart + offsets[i]) w.Write((byte)0);
            w.Write(tensors[i].Data);
        }
        w.Flush();
        File.WriteAllBytes(path, ms.ToArray());
    }

    private static byte[] Bytes(int length, int seed) => Enumerable.Range(0, length).Select(i => (byte)(seed * 31 + i * 7)).ToArray();

    // base: a (F32 4x2, 32 B), b (F32 8, 32 B), c (F16 5, 10 B -> exercises padding between tensors)
    private (string Base, string Donor) MakePair(long[]? donorADims = null)
    {
        string b = Path.Combine(_dir, "base.gguf"), d = Path.Combine(_dir, "donor.gguf");
        WriteGguf(b, [new("a", GgmlF32, [4, 2], Bytes(32, 1)), new("b", GgmlF32, [8], Bytes(32, 2)), new("c", GgmlF16, [5], Bytes(10, 3))]);
        // donor: a as F16 (same dims -> 16 B), b with different bytes; c absent is allowed because it is never selected.
        WriteGguf(d, [new("a", GgmlF16, donorADims ?? [4, 2], Bytes(16, 9)), new("b", GgmlF32, [8], Bytes(32, 8))], answer: 7);
        return (b, d);
    }

    private static (int Exit, string Out, string Err) Run(params string[] args)
    {
        var command = (ICommand)new GgufTransplantCommand();
        var originalOut = Console.Out;
        var originalErr = Console.Error;
        var stdout = new StringWriter();
        var stderr = new StringWriter();
        Console.SetOut(stdout);
        Console.SetError(stderr);
        try { return (command.Run(args, CancellationToken.None), stdout.ToString(), stderr.ToString()); }
        finally { Console.SetOut(originalOut); Console.SetError(originalErr); }
    }

    [Fact]
    public void ReplacesOnlyMatchingTensors_AndKeepsEverythingElseByteForByte()
    {
        var (b, d) = MakePair();
        string o = Path.Combine(_dir, "out.gguf");

        var (exit, stdout, stderr) = Run("--base", b, "--donor", d, "--tensors", "^a$", "-o", o);

        Assert.True(exit == 0, $"exit {exit}\n{stdout}\n{stderr}");
        using var baseModel = GgufModel.Open(b);
        using var donor = GgufModel.Open(d);
        using var result = GgufModel.Open(o);

        Assert.Equal(3, result.Tensors.Count);
        Assert.Equal(["a", "b", "c"], result.Tensors.Select(t => t.Name));          // order preserved
        Assert.Equal("test", result.GetMetadata<string>("general.architecture"));     // metadata survives, incl.
        Assert.Equal(42u, result.GetMetadata<uint>("test.answer"));                   // the BASE's values, not the donor's

        var a = result.FindTensor("a")!.Value;
        Assert.Equal(DType.Float16, a.DType);                                          // donor dtype
        Assert.True(result.GetTensorData(a).SequenceEqual(donor.GetTensorData(donor.FindTensor("a")!.Value)));

        foreach (var name in new[] { "b", "c" })                                       // untouched tensors identical to base
        {
            var written = result.FindTensor(name)!.Value;
            var original = baseModel.FindTensor(name)!.Value;
            Assert.Equal(original.DType, written.DType);
            Assert.True(result.GetTensorData(written).SequenceEqual(baseModel.GetTensorData(original)), $"tensor {name} changed");
        }
        foreach (var t in result.Tensors) Assert.Equal(0UL, t.DataOffset % 32);        // alignment honoured
    }

    [Fact]
    public void DimensionMismatch_FailsWithoutWritingAFile()
    {
        var (b, d) = MakePair(donorADims: [2, 4]);
        string o = Path.Combine(_dir, "out.gguf");

        var (exit, _, stderr) = Run("--base", b, "--donor", d, "--tensors", "^a$", "-o", o);

        Assert.NotEqual(0, exit);
        Assert.Contains("dimension mismatch", stderr, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(o));
    }

    [Fact]
    public void PatternMatchingNothing_FailsWithoutWritingAFile()
    {
        var (b, d) = MakePair();
        string o = Path.Combine(_dir, "out.gguf");

        var (exit, _, _) = Run("--base", b, "--donor", d, "--tensors", "^nothing$", "-o", o);

        Assert.NotEqual(0, exit);
        Assert.False(File.Exists(o));
    }

    [Fact]
    public void DonorMissingASelectedTensor_Fails()
    {
        var (b, d) = MakePair();
        var (exit, _, _) = Run("--base", b, "--donor", d, "--tensors", "^c$", "-o", Path.Combine(_dir, "out.gguf"));   // donor has no 'c'
        Assert.NotEqual(0, exit);
        Assert.False(File.Exists(Path.Combine(_dir, "out.gguf")));
    }

    [Fact]
    public void RefusesToOverwriteBaseOrDonor_AndExistingOutputNeedsForce()
    {
        var (b, d) = MakePair();
        byte[] baseBefore = File.ReadAllBytes(b);

        Assert.NotEqual(0, Run("--base", b, "--donor", d, "--tensors", "^a$", "-o", b).Exit);
        Assert.NotEqual(0, Run("--base", b, "--donor", d, "--tensors", "^a$", "-o", d).Exit);
        Assert.Equal(baseBefore, File.ReadAllBytes(b));                                // base never modified

        string o = Path.Combine(_dir, "out.gguf");
        Assert.Equal(0, Run("--base", b, "--donor", d, "--tensors", "^a$", "-o", o).Exit);
        long firstLength = new FileInfo(o).Length;
        File.WriteAllBytes(o, [1, 2, 3]);
        Assert.NotEqual(0, Run("--base", b, "--donor", d, "--tensors", "^a$", "-o", o).Exit);       // exists, no --force
        Assert.Equal(3, new FileInfo(o).Length);                                                      // left alone
        Assert.Equal(0, Run("--base", b, "--donor", d, "--tensors", "^a$", "-o", o, "--force").Exit);
        Assert.Equal(firstLength, new FileInfo(o).Length);                                            // overwritten with a valid file
    }

    [Fact]
    public void DryRun_PrintsThePlanAndWritesNothing()
    {
        var (b, d) = MakePair();
        string o = Path.Combine(_dir, "out.gguf");

        var (exit, stdout, _) = Run("--base", b, "--donor", d, "--tensors", "^(a|b)$", "-o", o, "--dry-run");

        Assert.Equal(0, exit);
        Assert.Contains("2 of 3 tensors replaced", stdout, StringComparison.Ordinal);
        Assert.False(File.Exists(o));
    }

    [Fact]
    public void InvalidRegex_FailsCleanly()
    {
        var (b, d) = MakePair();
        var (exit, _, _) = Run("--base", b, "--donor", d, "--tensors", "(", "-o", Path.Combine(_dir, "out.gguf"));
        Assert.NotEqual(0, exit);
    }
}
