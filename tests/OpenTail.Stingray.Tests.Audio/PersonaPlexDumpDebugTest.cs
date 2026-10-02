
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
        string? outDir = FindRepoFile("docs/00-current-work.md");
        string root = Path.GetDirectoryName(outDir!)!;

        Console.Error.WriteLine($"[PersonaPlexDump] embedded files: {string.Join(" | ", OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.Names(model))}");
        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, "config.json", out var configBytes))
        {
            string content = System.Text.Encoding.UTF8.GetString(configBytes);
            File.WriteAllText(Path.Combine(root, "..", "personaplex-config.json"), content);
        }
        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, "voices_safetensors/NATF0.safetensors", out var voiceBytes))
        {
            string voicePath = Path.Combine(root, "..", "personaplex-voice-natf0.safetensors");
            File.WriteAllBytes(voicePath, voiceBytes);
            Console.Error.WriteLine($"[PersonaPlexDump] wrote {voiceBytes.Length} bytes to {voicePath}");
        }

        if (model.Metadata.TryGetValue("audiocpp.tensor_names", out var tnObj) && tnObj is object[] tensorNames)
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < tensorNames.Length; i++)
            {
                string name = (string)tensorNames[i];
                var t = model.Tensors[i];
                sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
            }
            File.WriteAllText(Path.Combine(root, "..", "personaplex-tensor-names.txt"), sb.ToString());
            Console.Error.WriteLine($"[PersonaPlexDump] wrote transformer tensor names, {tensorNames.Length} total tensors");
        }
    }
}
