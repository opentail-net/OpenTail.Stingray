
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps real tensor names/shapes for MOSS-TTS-Nano's audio codec
/// decoder + quantizer (audio_tokenizer_weights/*) from the packed GGUF. Real ground truth for
/// writing the C# decoder port.</summary>
public sealed class MossTtsCodecTensorNameDumpDebugTest : HeavyTestBase
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
    public void DumpDecoderAndQuantizerTensorNames()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var namesObj = (object[])model.Metadata["audiocpp.tensor_names"];

        string? outDir = FindRepoFile("docs/audio-review-progress.md");
        string outPath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(outDir!)!, "..", "moss-tts-codec-tensor-names.txt"));

        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < namesObj.Length; i++)
        {
            string name = (string)namesObj[i];
            if (!name.StartsWith("audio_tokenizer_weights/decoder.") && !name.StartsWith("audio_tokenizer_weights/quantizer."))
                continue;
            var t = model.Tensors[i];
            sb.AppendLine($"{name}\t{string.Join(",", t.Dimensions)}\t{t.DType}");
        }
        File.WriteAllText(outPath, sb.ToString());
        Console.Error.WriteLine($"[MossTtsCodecDump] wrote {outPath}");
    }
}
