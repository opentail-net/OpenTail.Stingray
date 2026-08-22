using System.IO;
using System.Linq;
using OpenTail.Stingray.Audio.QwenTTS;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real structural test for <see cref="QwenTtsTalkerPromptBuilder"/>'s explicit-language prompt
/// path (`[think, think_bos, languageId, think_eos, codec_pad]` codec prefix, 5 rows, vs. the
/// auto-language case's 4-row `[nothink, think_bos, think_eos, codec_pad]`), added this session
/// alongside the auto-only base prompt. Uses the real GGUF language table
/// (`qwen3-tts.codec.language_ids`/`language_names`).
/// </summary>
public sealed class QwenTtsTalkerLanguagePromptTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ReadLanguageTable_RealMetadata_ContainsEnglishAndChinese()
    {
        string? modelPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");

        using var rawModel = GgufModel.Open(modelPath!);
        var table = QwenTtsTalkerPromptBuilder.ReadLanguageTable(rawModel);

        Assert.Equal(10, table.Count);
        Assert.True(table.ContainsKey("english"));
        Assert.True(table.ContainsKey("chinese"));
    }

    [Fact]
    public void BuildBasePrompt_WithExplicitLanguage_ProducesOneMoreRowThanAuto()
    {
        string? modelPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");

        using var rawModel = GgufModel.Open(modelPath!);
        var weights = QwenTtsTalkerPromptBuilder.Weights.Load(rawModel);
        var tokenizer = GgufTokenizer.FromGgufModel(rawModel);
        var languageTable = QwenTtsTalkerPromptBuilder.ReadLanguageTable(rawModel);

        var (autoEmbed, autoRows) = QwenTtsTalkerPromptBuilder.BuildBasePrompt(weights, tokenizer, "Hello there.");
        var (langEmbed, langRows) = QwenTtsTalkerPromptBuilder.BuildBasePrompt(weights, tokenizer, "Hello there.", "english", languageTable);

        // Real layout difference: auto uses a 4-row codec prefix, explicit-language uses 5
        // (adds the real language-id row) -- everything else (role + text body) is identical.
        Assert.Equal(autoRows + 1, langRows);

        foreach (var v in autoEmbed) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        foreach (var v in langEmbed) Assert.False(float.IsNaN(v) || float.IsInfinity(v));

        Assert.Throws<System.ArgumentException>(() =>
            QwenTtsTalkerPromptBuilder.BuildBasePrompt(weights, tokenizer, "Hello there.", "klingon", languageTable));
    }
}
