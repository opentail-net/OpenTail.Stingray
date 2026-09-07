
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps GGUF metadata keys (config json) for moss-tts-nano checkpoint.</summary>
public sealed class MossTtsMetaDumpDebugTest : HeavyTestBase
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
    public void DumpMeta()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "gguf not found");
        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        string? outDir = FindRepoFile("docs/audio-review-progress.md");
        string outPath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(outDir!)!, "..", "moss-tts-meta.txt"));
        var sb = new System.Text.StringBuilder();
        foreach (var kv in model.Metadata)
        {
            if (kv.Key == "audiocpp.tensor_names") continue;
            if (kv.Value is object[] arr)
            {
                sb.AppendLine($"{kv.Key} = [{arr.Length}] {string.Join(" | ", arr.Take(50))}");
                continue;
            }
            sb.AppendLine($"{kv.Key} = {kv.Value}");
        }
        File.WriteAllText(outPath, sb.ToString());

        var names = (object[])model.Metadata["audiocpp.embedded_files.names"];
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)(long)Convert.ChangeType(o, typeof(long))).ToArray();
        string[] wantText = ["config.json", "audio_tokenizer/config.json", "tokenization_moss_tts_nano.py", "tokenizer_config.json", "special_tokens_map.json"];
        string[] wantBinary = ["tokenizer.model"];
        for (int i = 0; i < names.Length; i++)
        {
            string name = (string)names[i];
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            string cfgOutPath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(outDir!)!, "..", name.Replace('/', '_').Insert(0, "moss-tts-")));
            if (Array.IndexOf(wantText, name) >= 0)
            {
                string content = System.Text.Encoding.UTF8.GetString(bytes, (int)start, (int)(end - start));
                File.WriteAllText(cfgOutPath, content);
                Console.Error.WriteLine($"[MossTtsMeta] wrote {cfgOutPath}");
            }
            else if (Array.IndexOf(wantBinary, name) >= 0)
            {
                File.WriteAllBytes(cfgOutPath, bytes[(int)start..(int)end]);
                Console.Error.WriteLine($"[MossTtsMeta] wrote {cfgOutPath} ({end - start} bytes)");
            }
        }
        Console.Error.WriteLine($"[MossTtsMeta] wrote {outPath}");
    }
}
