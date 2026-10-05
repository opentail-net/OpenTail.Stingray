using OpenTail.Stingray.Cli;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class SttCommandRoutingTests
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
    public void ResolveVoxtralPath_FindsDirectGgufFile()
    {
        string? ggufPath = FindRepoFile("models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf");
        Assert.SkipUnless(ggufPath is not null, "Voxtral GGUF not present in this environment");

        string? resolved = SttCommand.ResolveVoxtralPath(ggufPath);
        Assert.NotNull(resolved);
        Assert.True(File.Exists(resolved));
        Assert.EndsWith(".gguf", resolved, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ResolveVoxtralPath_DiscoversDefaultGgufWhenOmitted()
    {
        string? resolved = SttCommand.ResolveVoxtralPath(null);
        Assert.SkipUnless(resolved is not null, "no default Voxtral GGUF present in this environment");
        Assert.True(File.Exists(resolved) || Directory.Exists(resolved));
    }

    [Fact]
    public void SttCommand_FailsGracefully_WhenInputFileDoesNotExist()
    {
        var cmd = new SttCommand();
        int exitCode = cmd.ExecuteInternal(new SttCommand.Settings
        {
            Model = "voxtral",
            InputPath = "nonexistent_audio_file_12345.wav"
        });
        Assert.Equal(1, exitCode);
    }

    [Fact]
    public void SttCommand_ExecutesVoxtralGguf_Successfully()
    {
        string? ggufPath = FindRepoFile("models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf");
        string? wavPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(ggufPath is not null, "Voxtral GGUF not present in this environment");
        Assert.SkipUnless(wavPath is not null, "sample audio not present in this environment");

        var cmd = new SttCommand();
        int exitCode = cmd.ExecuteInternal(new SttCommand.Settings
        {
            Model = "voxtral",
            ModelFile = ggufPath,
            InputPath = wavPath
        });
        Assert.Equal(0, exitCode);
    }
}
