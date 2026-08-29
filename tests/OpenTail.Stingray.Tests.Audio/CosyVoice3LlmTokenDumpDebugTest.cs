using System;
using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: dumps the raw speech-token ids CosyVoice3Llm actually generates
/// for a real prompt, to check for degenerate patterns (repetition collapse, near-empty output,
/// out-of-range ids) as a cheap first check before assuming the bug lives in the DiT/flow wiring
/// instead.</summary>
public sealed class CosyVoice3LlmTokenDumpDebugTest : HeavyTestBase
{
    private static string? FindModelPath(string relPath)
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
    public void Dump_SpeechTokens_ForRealPrompt()
    {
        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(ggufPath != null, "CosyVoice3 GGUF not found");

        using var rawModel = GgufModel.Open(ggufPath!);
        var llmSource = new CosyVoice3LlmTensorSource(rawModel);
        llmSource.EnableSpeechGenerationMode();

        var tokens = CosyVoice3Llm.GenerateSpeechTokens(rawModel, llmSource, "This is a test of voice synthesis.", 200);

        string msg = $"[TOKENS] count={tokens.Length} min={(tokens.Length > 0 ? tokens.AsSpan().ToArray().Min() : -1)} max={(tokens.Length > 0 ? tokens.AsSpan().ToArray().Max() : -1)} first20=[{string.Join(",", tokens.AsSpan(0, Math.Min(20, tokens.Length)).ToArray())}] last20=[{string.Join(",", tokens.AsSpan(Math.Max(0, tokens.Length - 20)).ToArray())}]";
        Console.WriteLine(msg);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "cosyvoice3_llm_tokens.txt"), msg);

        // Check for repetition collapse: count the longest run of the same token.
        int longestRun = 1, curRun = 1;
        for (int i = 1; i < tokens.Length; i++)
        {
            if (tokens[i] == tokens[i - 1]) { curRun++; longestRun = Math.Max(longestRun, curRun); }
            else curRun = 1;
        }
        File.AppendAllText(Path.Combine(Path.GetTempPath(), "cosyvoice3_llm_tokens.txt"), $"\nlongestRepeatRun={longestRun}");
    }
}
