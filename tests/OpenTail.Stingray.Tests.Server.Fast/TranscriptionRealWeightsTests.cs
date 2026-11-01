using System.Diagnostics;
using System.Text.Json;
using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// <c>/v1/audio/transcriptions</c> with the real installed model: it used to build a weightless placeholder pipeline, so a 200 meant nothing.
/// This posts a real speech recording whose words are known (<c>docs/audio-samples/piper-lessac-sample.wav</c>, "Hello from Piper on OpenTail
/// Stingray.") and requires the first words back ("hello from piper"; whisper-base mishears the brand name on this synthetic voice, and
/// <c>stingray transcribe</c> prints the same text from the same weights), then sends two requests at once, since each request gets its own pipeline
/// over the cached weights.
/// Skipped, visibly, unless <c>STINGRAY_RUN_HEAVY_TESTS=1</c> and the catalog's <c>ggml-base.bin</c> (matching size) is found: a run that took
/// milliseconds did not run real weights (CLAUDE.md rule 12), so the elapsed time is part of the assertion.
/// </summary>
public sealed class TranscriptionRealWeightsTests
{
    private static string? FindUpward(string relative)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    /// <summary>The folder holding the installed catalog model: <c>models/</c> or <c>models/_models/</c> above the test, else a <c>_models</c> folder at the root of any fixed drive.</summary>
    private static string? FindModelFolder(CatalogFile file)
    {
        foreach (string relative in new[] { Path.Combine("models", file.FileName), Path.Combine("models", "_models", file.FileName) })
            if (FindUpward(relative) is { } found && new FileInfo(found).Length == file.Size) return Path.GetDirectoryName(found);
        foreach (var drive in DriveInfo.GetDrives().Where(d => d.DriveType == DriveType.Fixed && d.IsReady))
        {
            string candidate = Path.Combine(drive.RootDirectory.FullName, "_models", file.FileName);
            if (File.Exists(candidate) && new FileInfo(candidate).Length == file.Size) return Path.GetDirectoryName(candidate);
        }
        return null;
    }

    [Fact]
    public async Task RealWhisperBase_TranscribesTheKnownSample_AndTwoConcurrentRequestsBothSucceed()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "real-weight test: set STINGRAY_RUN_HEAVY_TESTS=1");
        var entry = ModelCatalog.DefaultFor("transcribe")!;
        string? folder = FindModelFolder(entry.MainFile);
        Assert.SkipUnless(folder is not null, $"{entry.MainFile.FileName} ({entry.MainFile.Size} bytes) is required for this test");
        string? sample = FindUpward(Path.Combine("docs", "audio-samples", "piper-lessac-sample.wav"));
        Assert.SkipUnless(sample is not null, "docs/audio-samples/piper-lessac-sample.wav not found");

        var provider = new InstalledWhisperPipelineProvider(() => new ModelHome(folder!));
        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s =>
            {
                s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"));
                s.AddSingleton<IWhisperPipelineProvider>(provider);
            }));
        using var client = factory.CreateClient();
        byte[] wav = await File.ReadAllBytesAsync(sample!);

        async Task<string> TranscribeAsync()
        {
            using var form = new MultipartFormDataContent
            {
                { new ByteArrayContent(wav), "file", "sample.wav" }, { new StringContent("whisper-base"), "model" },
                { new StringContent("en"), "language" },
            };
            using var response = await client.PostAsync("/v1/audio/transcriptions", form);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            return doc.RootElement.GetProperty("text").GetString()!;
        }

        var clock = Stopwatch.StartNew();
        string first = await TranscribeAsync();
        TimeSpan firstTook = clock.Elapsed;

        // The words of the recording, not just any text: a placeholder pipeline produces none of them.
        string normalized = new string(first.ToLowerInvariant().Where(c => char.IsLetter(c) || c == ' ').ToArray());
        // whisper-base mishears the brand name on this synthetic voice ("Piperon Penial Tingad"); `stingray transcribe` gives the same text from the same
        // weights. The first words are what proves real recognition: a placeholder pipeline yields none of them.
        Assert.Contains("hello from piper", normalized);
        Assert.True(firstTook > TimeSpan.FromSeconds(0.5), $"the first request took {firstTook.TotalMilliseconds:F0} ms; real weights take seconds");

        clock.Restart();
        var both = await Task.WhenAll(TranscribeAsync(), TranscribeAsync());
        Assert.Equal(first, both[0]);   // same audio, same weights, greedy: the same words, concurrent or not
        Assert.Equal(first, both[1]);
        Console.Error.WriteLine($"[TranscriptionRealWeightsTests] first request {firstTook.TotalSeconds:F1} s (load included); two concurrent {clock.Elapsed.TotalSeconds:F1} s; text: {first}");
    }
}
