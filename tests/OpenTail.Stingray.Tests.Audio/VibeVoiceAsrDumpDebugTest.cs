
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps config.json + real tensor names/shapes for VibeVoice ASR's
/// packed GGUF. Real ground truth for real-weight verification.</summary>
public sealed class VibeVoiceAsrDumpDebugTest : HeavyTestBase
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
        string? path = FindRepoFile("models/_models/vibevoice_asr/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-asr-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        string? outDir = FindRepoFile("docs/00-current-work.md");
        string root = Path.GetDirectoryName(outDir!)!;

        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, "config.json", out var configBytes))
        {
            string content = System.Text.Encoding.UTF8.GetString(configBytes);
            File.WriteAllText(Path.Combine(root, "..", "vibevoice-asr-config.json"), content);
        }
        Console.Error.WriteLine($"[VibeVoiceAsrDump] embedded files: {string.Join(" | ", OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.Names(model))}");

        if (model.Metadata.TryGetValue("audiocpp.tensor_names", out var tnObj) && tnObj is object[] tensorNames)
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < tensorNames.Length; i++)
            {
                string name = (string)tensorNames[i];
                var t = model.Tensors[i];
                sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
            }
            File.WriteAllText(Path.Combine(root, "..", "vibevoice-asr-tensor-names.txt"), sb.ToString());
            Console.Error.WriteLine($"[VibeVoiceAsrDump] wrote {tensorNames.Length} tensor names");
        }
    }
}
