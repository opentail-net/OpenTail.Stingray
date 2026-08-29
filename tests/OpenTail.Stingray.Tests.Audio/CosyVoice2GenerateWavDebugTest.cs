
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: generates a real CosyVoice2 wav end-to-end (no CLI wiring
/// exists for CosyVoice2 yet -- "cosyvoice"/"cosy" both route to CosyVoice3Pipeline). This is a
/// real listening test, not a numeric one -- the earlier claim that "CosyVoice2's LLM was already
/// verified working" was based only on CosyVoiceLlmTensorSourceTests, which only checks finite/
/// non-degenerate logits, NOT a real numeric oracle comparison (examples/cosyvoice.cpp doesn't
/// even implement the base cosyvoice_model::llm_job path CosyVoice2 would need -- only
/// cosyvoice_model_3's). So actually listening to real output is the only real check available
/// for CosyVoice2 right now.</summary>
public sealed class CosyVoice2GenerateWavDebugTest : HeavyTestBase
{
    private static string? FindModelPath(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_RealCosyVoice2Wav()
    {
        string? llmPath = FindModelPath("models/cosyvoice2_llm.safetensors");
        string? tokDir = FindModelPath("models/cosyvoice2_tokenizer");
        string? flowPath = FindModelPath("models/cosyvoice2_flow.safetensors");
        string? hiftPath = FindModelPath("models/cosyvoice2_hift.safetensors");
        Assert.SkipUnless(llmPath != null && tokDir != null && flowPath != null && hiftPath != null,
            "CosyVoice2 model files not found");

        using var pipeline = CosyVoice2Pipeline.Load(llmPath!, tokDir!, flowPath!, hiftPath!);
        var wav = pipeline.Generate("This is a test of voice synthesis.", seed: 42);

        Assert.True(wav.Length > 0, "CosyVoice2 produced empty audio");

        string repoRoot = Directory.GetParent(Path.GetDirectoryName(llmPath!)!)!.FullName;
        var result = new AudioGenerationResult(wav, pipeline.SampleRate);
        string outPath = Path.Combine(repoRoot, "docs", "audio-samples", "cosyvoice2-real-check.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {wav.Length} samples, {wav.Length / (double)pipeline.SampleRate:F2}s");
    }
}
