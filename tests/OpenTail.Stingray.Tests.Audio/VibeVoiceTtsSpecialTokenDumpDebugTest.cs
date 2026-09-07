using System.Text.Json;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: dumps real vocab ids for VibeVoice TTS's real repurposed vision
/// special tokens (speech_start/speech_end/speech_diffusion) plus eos.</summary>
public sealed class VibeVoiceTtsSpecialTokenDumpDebugTest : HeavyTestBase
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
    public void DumpSpecialTokenIds()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-1.5b-q8_0.gguf not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var namesObj = (object[])model.Metadata["audiocpp.embedded_files.names"];
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        string? tokenizerJson = null;
        for (int i = 0; i < namesObj.Length; i++)
        {
            if ((string)namesObj[i] != "tokenizer.json") continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            tokenizerJson = System.Text.Encoding.UTF8.GetString(bytes, (int)start, (int)(end - start));
        }
        Assert.NotNull(tokenizerJson);

        using var doc = JsonDocument.Parse(tokenizerJson!);
        var vocab = doc.RootElement.GetProperty("model").GetProperty("vocab");

        string[] wanted = ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|endoftext|>", "<|image_pad|>"];
        var found = new Dictionary<string, int>();
        foreach (string token in wanted)
            if (vocab.TryGetProperty(token, out var idEl)) found[token] = idEl.GetInt32();

        if (doc.RootElement.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
        {
            foreach (var entry in added.EnumerateArray())
            {
                if (!entry.TryGetProperty("content", out var contentEl)) continue;
                string content = contentEl.GetString() ?? "";
                if (Array.IndexOf(wanted, content) < 0) continue;
                found[content] = entry.GetProperty("id").GetInt32();
            }
        }

        foreach (string token in wanted)
            Console.Error.WriteLine(found.TryGetValue(token, out int id)
                ? $"[SpecialToken] {token} = {id}"
                : $"[SpecialToken] {token} = NOT FOUND");
    }
}
