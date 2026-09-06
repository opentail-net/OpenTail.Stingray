
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps the real tensor names for RVC's support_hubert_base/support_rmvpe
/// components from the packed rvc-f16.gguf's `audiocpp.tensor_names` metadata array (real names,
/// paired by index with `Model.Tensors`, since `general.architecture=audiocpp`/`tensor_name_format
/// =native` means the tensors themselves carry opaque `_audiocpp.NNNN` names -- the real names
/// live only in this metadata array). Real ground truth for writing RvcHubertWeights/
/// RvcRmvpeWeights loaders.</summary>
public sealed class RvcTensorNameDumpDebugTest : HeavyTestBase
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
    public void DumpHubertAndRmvpeTensorNames()
    {
        string? path = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(path != null, "rvc-f16.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var namesObj = (object[])model.Metadata["audiocpp.tensor_names"];
        Assert.Equal(model.Tensors.Count, namesObj.Length);

        string? outDir = FindRepoFile("docs/audio-review-progress.md");
        string outPath = Path.Combine(Path.GetDirectoryName(outDir!)!, "..", "rvc-tensor-names.txt");
        outPath = Path.GetFullPath(outPath);

        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < namesObj.Length; i++)
        {
            string name = (string)namesObj[i];
            if (name.StartsWith("support_hubert_base/") || name.StartsWith("support_rmvpe/"))
            {
                var t = model.Tensors[i];
                sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
            }
        }
        File.WriteAllText(outPath, sb.ToString());
        Console.Error.WriteLine($"[RvcTensorDump] wrote {outPath}, {sb.ToString().Split('\n').Length} lines");
    }
}
