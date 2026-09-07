
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps config.json + transformer.* tensor names for PersonaPlex's
/// packed GGUF. Real ground truth for its LM backbone's real config numbers.</summary>
public sealed class PersonaPlexDumpDebugTest : HeavyTestBase
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
    public void DumpConfigAndTransformerTensorNames()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        string? outDir = FindRepoFile("docs/audio-review-progress.md");
        string root = Path.GetDirectoryName(outDir!)!;

        if (model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) && namesObj is object[] names)
        {
            var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
            var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
            var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
            Console.Error.WriteLine($"[PersonaPlexDump] embedded files: {string.Join(" | ", names.Cast<string>())}");
            for (int i = 0; i < names.Length; i++)
            {
                string name = (string)names[i];
                if (name != "config.json") continue;
                long start = Convert.ToInt64(offsets[i]);
                long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
                string content = System.Text.Encoding.UTF8.GetString(bytes, (int)start, (int)(end - start));
                File.WriteAllText(Path.Combine(root, "..", "personaplex-config.json"), content);
            }
        }

        if (model.Metadata.TryGetValue("audiocpp.tensor_names", out var tnObj) && tnObj is object[] tensorNames)
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < tensorNames.Length; i++)
            {
                string name = (string)tensorNames[i];
                if (!name.StartsWith("transformer.", StringComparison.Ordinal)
                    && !name.Contains("depformer", StringComparison.OrdinalIgnoreCase)
                    && !name.Contains("embed", StringComparison.OrdinalIgnoreCase)
                    && !name.Contains("lm_head", StringComparison.OrdinalIgnoreCase))
                    continue;
                var t = model.Tensors[i];
                sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
            }
            File.WriteAllText(Path.Combine(root, "..", "personaplex-tensor-names.txt"), sb.ToString());
            Console.Error.WriteLine($"[PersonaPlexDump] wrote transformer tensor names, {tensorNames.Length} total tensors");
        }
    }
}
