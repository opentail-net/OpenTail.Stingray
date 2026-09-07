
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps the real tensor names/shapes for MOSS-TTS-Nano's packed
/// moss-tts-nano-100m-q8_0.gguf (`audiocpp.tensor_names` metadata array, same packed-GGUF
/// convention as RVC -- see `RvcPackedTensorSource`). Real ground truth for writing the global
/// transformer / local transformer / audio codec weight loaders.</summary>
public sealed class MossTtsTensorNameDumpDebugTest : HeavyTestBase
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
    public void DumpAllTensorNames()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano-100m-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var namesObj = (object[])model.Metadata["audiocpp.tensor_names"];
        Assert.Equal(model.Tensors.Count, namesObj.Length);

        string? outDir = FindRepoFile("docs/audio-review-progress.md");
        string outPath = Path.Combine(Path.GetDirectoryName(outDir!)!, "..", "moss-tts-tensor-names.txt");
        outPath = Path.GetFullPath(outPath);

        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < namesObj.Length; i++)
        {
            string name = (string)namesObj[i];
            var t = model.Tensors[i];
            sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
        }
        File.WriteAllText(outPath, sb.ToString());
        Console.Error.WriteLine($"[MossTtsDump] wrote {outPath}, {namesObj.Length} tensors");
    }
}
