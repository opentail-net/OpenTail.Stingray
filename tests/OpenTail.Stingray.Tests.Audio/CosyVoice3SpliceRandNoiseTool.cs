using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF TOOL (not a regression test): splices the real reference's fixed
/// `flow/decoder.rand_noise` tensor ([15000, 80] Float32 -- CosyVoice3's frozen flow-matching ODE
/// starting noise, see docs/audio-review-progress.md) out of the CosyVoice3-GGUF Q8_0 reference
/// checkpoint and appends it, renamed `decoder.rand_noise` to match this repo's unprefixed naming
/// convention, into a NEW copy of our own CosyVoice3-2512_F16.gguf.
///
/// Byte-surgery strategy (avoids re-encoding the GGUF KV-metadata value format entirely):
///   [0, Shard0TensorInfoEndOffset)              copied verbatim (header+KV+existing tensor-infos)
///   + one new tensor-info entry for decoder.rand_noise (offset = old data blob's length)
///   + zero-padding out to the file's alignment
///   [Shard0DataStartOffset, EOF)                copied verbatim (existing tensor data, unmoved --
///                                                 every existing DataOffset is relative to this
///                                                 blob's start and therefore stays correct)
///   + the new tensor's raw bytes, appended last
///
/// The SOURCE file (CosyVoice3-2512_F16.gguf) is opened read-only and never written to; the output
/// goes to a sibling file with a distinct name.</summary>
public sealed class CosyVoice3SpliceRandNoiseTool : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Splice_RandNoise_Into_New_Checkpoint()
    {
        string? srcPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        string? donorPath = FindRepoFile("examples/audio.cpp/models/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf");
        Assert.SkipUnless(srcPath != null, "CosyVoice3-2512_F16.gguf not found");
        Assert.SkipUnless(donorPath != null, "cosyvoice3-q8_0.gguf reference checkpoint not found");

        string outPath = Path.Combine(Path.GetDirectoryName(srcPath!)!, "CosyVoice3-2512_F16-with-noise.gguf");

        byte[] noiseBytes;
        GgufTensorInfo donorTensor;
        using (var donor = GgufModel.Open(donorPath!))
        {
            donorTensor = donor.Tensors.First(t => t.Name == "flow/decoder.rand_noise");
            noiseBytes = donor.GetTensorData(donorTensor).ToArray();
        }
        Assert.Equal(15000L * 80 * sizeof(float), noiseBytes.Length);

        const int alignment = 32; // matches DefaultAlignment in GgufModel.cs
        using (var src = GgufModel.Open(srcPath!))
        {
            long tensorInfoEnd = src.Shard0TensorInfoEndOffset;
            long dataStart = src.Shard0DataStartOffset;
            long srcFileLen = new FileInfo(srcPath!).Length;
            long dataBlobLen = srcFileLen - dataStart;

            byte[] head = new byte[tensorInfoEnd];
            using (var fs = File.OpenRead(srcPath!))
                fs.ReadExactly(head, 0, head.Length);

            using var outFs = new FileStream(outPath, FileMode.Create, FileAccess.Write);

            // Header: bump tensor_count (offset 8, uint64 LE) by one, count unchanged otherwise.
            ulong oldTensorCount = BitConverter.ToUInt64(head, 8);
            BitConverter.GetBytes(oldTensorCount + 1).CopyTo(head, 8);
            outFs.Write(head, 0, head.Length);

            // New tensor-info entry: name, ndims, dims[], ggml type (Float32=0), data offset
            // (relative to the data blob start, i.e. right after the existing blob).
            WriteGgufString(outFs, "decoder.rand_noise");
            WriteU32(outFs, (uint)donorTensor.NDimensions);
            foreach (var d in donorTensor.Dimensions)
                WriteU64(outFs, (ulong)d);
            WriteU32(outFs, 0); // GGML_TYPE_F32
            WriteU64(outFs, (ulong)dataBlobLen);

            long posAfterEntry = outFs.Position;
            long padded = AlignUp(posAfterEntry, alignment);
            for (long i = posAfterEntry; i < padded; i++) outFs.WriteByte(0);

            using (var fs = File.OpenRead(srcPath!))
            {
                fs.Seek(dataStart, SeekOrigin.Begin);
                byte[] buf = new byte[1 << 20];
                long remaining = dataBlobLen;
                while (remaining > 0)
                {
                    int chunk = (int)Math.Min(buf.Length, remaining);
                    fs.ReadExactly(buf, 0, chunk);
                    outFs.Write(buf, 0, chunk);
                    remaining -= chunk;
                }
            }

            outFs.Write(noiseBytes, 0, noiseBytes.Length);
        }

        // Original must remain byte-for-byte untouched.
        Console.WriteLine($"Wrote {outPath} ({new FileInfo(outPath).Length} bytes); " +
                           $"original {srcPath} unchanged ({new FileInfo(srcPath!).Length} bytes)");

        using var verify = GgufModel.Open(outPath);
        var loaded = verify.Tensors.First(t => t.Name == "decoder.rand_noise");
        Assert.Equal(15000L, loaded.Dimensions[0]);
        Assert.Equal(80L, loaded.Dimensions[1]);
        var loadedBytes = verify.GetTensorData(loaded);
        Assert.True(loadedBytes.SequenceEqual(noiseBytes));
    }

    private static long AlignUp(long value, int alignment) =>
        (value + alignment - 1) / alignment * alignment;

    private static void WriteGgufString(Stream s, string str)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(str);
        WriteU64(s, (ulong)bytes.Length);
        s.Write(bytes, 0, bytes.Length);
    }

    private static void WriteU32(Stream s, uint v) => s.Write(BitConverter.GetBytes(v));
    private static void WriteU64(Stream s, ulong v) => s.Write(BitConverter.GetBytes(v));
}
