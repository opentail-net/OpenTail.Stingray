
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps config.json + body.*/tied.embedding.* tensor names for Higgs
/// Audio TTS's packed GGUF. Real ground truth for real-weight verification.</summary>
public sealed class HiggsAudioTtsDumpDebugTest : HeavyTestBase
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
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs checkpoint not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        string? outDir = FindRepoFile("docs/00-current-work.md");
        string root = Path.GetDirectoryName(outDir!)!;

        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, "config.json", out var configBytes))
        {
            string content = System.Text.Encoding.UTF8.GetString(configBytes);
            File.WriteAllText(Path.Combine(root, "..", "higgs-tts-config.json"), content);
            Console.Error.WriteLine($"[HiggsDump] embedded files: {string.Join(" | ", OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.Names(model))}");
        }

        if (model.Metadata.TryGetValue("audiocpp.tensor_names", out var tnObj) && tnObj is object[] tensorNames)
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < tensorNames.Length; i++)
            {
                string name = (string)tensorNames[i];
                if (!name.StartsWith("body.") && !name.StartsWith("tied.")) continue;
                var t = model.Tensors[i];
                sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
            }
            File.WriteAllText(Path.Combine(root, "..", "higgs-tts-tensor-names.txt"), sb.ToString());
            Console.Error.WriteLine($"[HiggsDump] wrote body/tied tensor names, {tensorNames.Length} total tensors");
        }
    }
}
