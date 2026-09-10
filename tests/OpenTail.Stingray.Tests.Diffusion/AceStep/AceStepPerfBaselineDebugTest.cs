using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Text;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using OpenTail.Stingray.Diffusion.AceStep.Vae;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for ACE-Step Turbo for PerformanceLeague.md backfill.
/// No numeric golden reference exists for the full pipeline yet (see
/// AceStepPipelineEndToEndTests), so this is a timing-only, non-degeneracy-checked measurement --
/// OT-only new coverage, no ratio. Not part of the permanent suite, delete after use.
/// </summary>
public sealed class AceStepPerfBaselineDebugTest
{
    private const int Runs = 3;

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
    public void Bench_Generate_2sAudio()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        string? vaePath = FindRepoFile("models/acestep-v15/vae.safetensors");
        string? ggufPath = FindRepoFile("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        Assert.SkipUnless(turboPath != null && vaePath != null && ggufPath != null,
            "models/acestep-v15 or models/qwen3-embedding-0.6b weights not found");

        using var turboLoader = SafetensorsLoader.Open(turboPath!);
        var ditWeights = AceStepDiTWeights.Load(turboLoader);
        var conditionWeights = AceStepConditionEncoderWeights.Load(turboLoader);
        var timbreWeights = AceStepTimbreEncoderWeights.Load(turboLoader);

        using var vaeLoader = SafetensorsLoader.Open(vaePath!);
        var vaeWeights = AceStepOobleckDecoderWeights.Load(vaeLoader);
        var vaeEncoderWeights = AceStepOobleckEncoderWeights.Load(vaeLoader);

        using var textEncoder = new AceStepQwen3TextEncoder(ggufPath!);

        var model = new AceStepModel
        {
            Transformer = ditWeights,
            Vae = vaeWeights,
            VaeEncoder = vaeEncoderWeights,
            TextEncoder = textEncoder,
            ConditionEncoder = conditionWeights,
            TimbreEncoder = timbreWeights,
        };
        var pipeline = new AceStepPipeline(model);

        var genParams = new AceStepGenerationParams
        {
            Prompt = "A cinematic orchestral soundtrack with deep drums",
            Lyrics = "",
            Instrumental = true,
            DurationSeconds = 2f,
            Seed = 1234,
        };

        var warm = pipeline.Generate(genParams);
        Assert.True(warm.SampleCount > 0, "generated zero samples on warmup");

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        int sr = AceStepConfig.VaeSampleRate;
        for (int i = 0; i < Runs; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = pipeline.Generate(genParams);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = result.SampleCount;
            sr = result.SampleRate;
        }

        double audioSec = sampleCount / (double)sr;
        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSec;
        string msg = $"[ACE-Step-Turbo] prompt=\"A cinematic orchestral soundtrack with deep drums\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[ACE-Step-Turbo] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime; 8-step Turbo, non-degeneracy checked only, no numeric golden reference exists yet)";
        Console.Error.WriteLine(msg);
        string? docsDir = FindRepoFile("docs");
        if (docsDir != null)
            File.AppendAllText(Path.Combine(docsDir, "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
