using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: extracts VoxCPM2's real embedded tokenizer.json/tokenizer_config.json
/// to inspect the real pre_tokenizer/add_prefix_space configuration -- real, precise localization
/// of a confirmed first-token tokenization mismatch (see VoxCpm2PrefillCompareDebugTest).</summary>
public sealed class VoxCpm2TokenizerConfigDumpDebugTest : HeavyTestBase
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
    public void DumpTokenizerFiles()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names)
        {
            Console.WriteLine("No embedded_files metadata.");
            return;
        }
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        for (int i = 0; i < names.Length; i++)
        {
            string name = (string)names[i];
            if (name != "tokenizer.json" && name != "tokenizer_config.json" && name != "config.json") continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            string content = System.Text.Encoding.UTF8.GetString(bytes, (int)start, (int)(end - start));
            string outPath = Path.Combine(Path.GetTempPath(), $"voxcpm2-{name}");
            File.WriteAllText(outPath, content);
            Console.WriteLine($"Wrote {outPath} ({content.Length} chars)");
        }
    }
}
