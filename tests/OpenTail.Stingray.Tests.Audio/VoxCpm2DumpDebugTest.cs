
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps config.json + audiovae tensor names/shapes for VoxCPM2's packed
/// GGUF. Real ground truth for writing the AudioVAE decoder loader.</summary>
public sealed class VoxCpm2DumpDebugTest : HeavyTestBase
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
    public void DumpConfigAndTensorNames()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        string? outDir = FindRepoFile("docs/00-current-work.md");
        string root = Path.GetDirectoryName(outDir!)!;

        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, "config.json", out var bytes))
        {
            string content = System.Text.Encoding.UTF8.GetString(bytes);
            File.WriteAllText(Path.Combine(root, "..", "voxcpm2-config.json"), content);
        }

        if (model.Metadata.TryGetValue("audiocpp.tensor_names", out var tnObj) && tnObj is object[] tensorNames)
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < tensorNames.Length; i++)
            {
                string name = (string)tensorNames[i];
                if (!name.StartsWith("audiovae_weights/")) continue;
                var t = model.Tensors[i];
                sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
            }
            File.WriteAllText(Path.Combine(root, "..", "voxcpm2-audiovae-tensor-names.txt"), sb.ToString());
        }
        Console.Error.WriteLine("[VoxCpm2Dump] wrote voxcpm2-config.json + voxcpm2-audiovae-tensor-names.txt");
    }
}
