namespace OpenTail.Stingray.Tests.Cli;

public sealed class TtsEnginesTests
{
    [Fact]
    public void TheEngineOptionDescription_MatchesTheSharedEngineList()
    {
        // The attribute must stay a plain literal (the option-inventory generator reads quoted text), so this keeps it
        // in lockstep with TtsEngines.Names, which also feeds the unknown-engine error.
        var attr = typeof(TtsCommand.Settings).GetProperty(nameof(TtsCommand.Settings.Engine))!
            .GetCustomAttributes(typeof(System.ComponentModel.DescriptionAttribute), false)
            .Cast<System.ComponentModel.DescriptionAttribute>().Single();
        Assert.Equal("TTS engine: " + TtsEngines.Names + ".", attr.Description);
    }

    [Fact]
    public void EveryCanonicalEngineIsNamedInTheHelpText_AndTheHelpNamesNothingUnknown()
    {
        // The help list was once stale (5 engines documented, 12 accepted). The help text is generated from Names; this
        // pins that Names and the alias table agree in both directions.
        var helped = TtsEngines.Names.Split(',').Select(p => p.Trim().Split(' ')[0]).ToList();
        Assert.Equal(TtsEngines.CanonicalNames.OrderBy(x => x), helped.OrderBy(x => x));
    }

    [Theory]
    [InlineData("fish", "fishspeech")]
    [InlineData("S2-Pro", "fishspeech")]
    [InlineData("f5", "f5tts")]
    [InlineData("QWEN-TALKER", "qwentts")]
    [InlineData("xttsv2", "xtts")]
    [InlineData(" kokoro ", "kokoro")]
    public void Aliases_MapToTheCanonicalName_CaseInsensitively(string alias, string canonical) =>
        Assert.Equal(canonical, TtsEngines.Canonical(alias));

    [Fact]
    public void UnknownEngine_HasNoCanonicalName() => Assert.Null(TtsEngines.Canonical("nope"));

    [Fact]
    public void OnlyEnginesWithASamplingStepHonourTheSeed()
    {
        foreach (var e in new[] { "fishspeech", "parler", "qwentts", "xtts", "mms", "cosyvoice" })
            Assert.True(TtsEngines.HonorsSeed(e), e);
        foreach (var e in new[] { "kokoro", "piper", "f5tts", "chatterbox", "melo", "orpheus" })
            Assert.False(TtsEngines.HonorsSeed(e), e);
    }
}

public sealed class TtsCommandTests
{
    private static (int Exit, string Out, string Err) Run(params string[] args)
    {
        var command = (ICommand)new TtsCommand();
        var originalOut = Console.Out;
        var originalErr = Console.Error;
        var stdout = new StringWriter();
        var stderr = new StringWriter();
        Console.SetOut(stdout);
        Console.SetError(stderr);
        try { return (command.Run(args, CancellationToken.None), stdout.ToString(), stderr.ToString()); }
        finally { Console.SetOut(originalOut); Console.SetError(originalErr); }
    }

    [Fact]
    public void MissingText_IsAUsageError()
    {
        var (exit, _, err) = Run("-e", "kokoro");
        Assert.Equal(ExitCodesForTests.Usage, exit);
        Assert.Contains("--text", err, StringComparison.Ordinal);
    }

    [Fact]
    public void UnknownEngine_IsAUsageErrorThatListsEveryEngine()
    {
        var (exit, _, err) = Run("-t", "hello", "-e", "nope");
        Assert.Equal(ExitCodesForTests.Usage, exit);
        Assert.Contains("fishspeech", err, StringComparison.Ordinal);
        Assert.Contains("xtts", err, StringComparison.Ordinal);
    }

    [Fact]
    public void EngineThatNeedsAModelPath_ReportsAUsageErrorWithoutGenerating()
    {
        var (exit, _, err) = Run("-t", "hello", "-e", "piper");   // piper has no default location
        Assert.Equal(ExitCodesForTests.Usage, exit);
        Assert.Contains("--model", err, StringComparison.Ordinal);
    }

    [Fact]
    public void SeedIsAnAcceptedOption()
    {
        // Parsing only: a bad engine fails fast, so no model is touched. The point is that --seed binds (it did not exist).
        var (exit, _, err) = Run("-t", "hello", "-e", "nope", "--seed", "7");
        Assert.Equal(ExitCodesForTests.Usage, exit);
        Assert.DoesNotContain("Unrecognized", err, StringComparison.OrdinalIgnoreCase);
    }

    // ExitCodes is internal to the CLI assembly; mirror the documented value so a silent renumbering is caught here.
    private static class ExitCodesForTests { public const int Usage = 64; }
}
