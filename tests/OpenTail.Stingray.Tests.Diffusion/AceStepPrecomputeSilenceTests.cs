using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Vae;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class AceStepPrecomputeSilenceTests
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void PrecomputeAndSave_SilenceLatentAndTimbre()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        string? vaePath = FindRepoFile("models/acestep-v15/vae.safetensors");
        Assert.SkipUnless(turboPath != null, "turbo.safetensors not found");
        Assert.SkipUnless(vaePath != null, "vae.safetensors not found");

        using var turboLoader = SafetensorsLoader.Open(turboPath!);
        var timbreWeights = AceStepTimbreEncoderWeights.Load(turboLoader);

        using var vaeLoader = SafetensorsLoader.Open(vaePath!);
        var vaeEncoderWeights = AceStepOobleckEncoderWeights.Load(vaeLoader);

        const int frames = 750;
        int hopLength = AceStepConfig.VaeDownsamplingRatios.Aggregate(1, (a, b) => a * b);
        int sampleCount = frames * hopLength;
        var zeroPcm = new float[AceStepConfig.VaeAudioChannels * sampleCount];

        Console.WriteLine($"[Precompute] Encoding {sampleCount} silence samples through VAE encoder...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var flat = AceStepOobleckEncoder.EncodeMode(vaeEncoderWeights, zeroPcm, AceStepConfig.VaeAudioChannels, sampleCount);
        Console.WriteLine($"[Precompute] VAE Encode completed in {sw.ElapsedMilliseconds} ms ({flat.Length} floats)");

        // Convert to rows: [frames, 64]
        int latentDim = AceStepConfig.VaeDecoderInputChannels;
        var rows = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var row = new float[latentDim];
            for (int c = 0; c < latentDim; c++) row[c] = flat[c * frames + t];
            rows[t] = row;
        }

        sw.Restart();
        var timbreRow = AceStepTimbreEncoder.Forward(timbreWeights, rows);
        Console.WriteLine($"[Precompute] Timbre Forward completed in {sw.ElapsedMilliseconds} ms ({timbreRow.Length} floats)");

        // Save silence_latent.bin
        string acestepDir = Path.GetDirectoryName(turboPath!)!;
        string latentOutPath = Path.Combine(acestepDir, "silence_latent.bin");
        using (var fs = File.Create(latentOutPath))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write(frames);
            bw.Write(latentDim);
            for (int t = 0; t < frames; t++)
                for (int c = 0; c < latentDim; c++)
                    bw.Write(rows[t][c]);
        }
        Console.WriteLine($"[Precompute] Saved {latentOutPath} ({new FileInfo(latentOutPath).Length} bytes)");

        // Save silence_timbre.bin
        string timbreOutPath = Path.Combine(acestepDir, "silence_timbre.bin");
        using (var fs = File.Create(timbreOutPath))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write(timbreRow.Length);
            for (int i = 0; i < timbreRow.Length; i++)
                bw.Write(timbreRow[i]);
        }
        Console.WriteLine($"[Precompute] Saved {timbreOutPath} ({new FileInfo(timbreOutPath).Length} bytes)");

        // Also copy directly into src/OpenTail.Stingray.Diffusion/AceStep/ for internalization
        string? srcAceStepDir = FindRepoFile("src/OpenTail.Stingray.Diffusion/AceStep/AceStepPipeline.cs");
        if (srcAceStepDir != null)
        {
            string destDir = Path.GetDirectoryName(srcAceStepDir)!;
            File.Copy(latentOutPath, Path.Combine(destDir, "silence_latent.bin"), overwrite: true);
            File.Copy(timbreOutPath, Path.Combine(destDir, "silence_timbre.bin"), overwrite: true);
            Console.WriteLine($"[Precompute] Internalized into {destDir}");
        }

        Assert.True(File.Exists(latentOutPath));
        Assert.True(File.Exists(timbreOutPath));
    }
}
