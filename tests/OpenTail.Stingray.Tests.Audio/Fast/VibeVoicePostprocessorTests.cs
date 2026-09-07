using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural tests for <see cref="VibeVoicePostprocessor"/>.</summary>
public sealed class VibeVoicePostprocessorTests
{
    [Fact]
    public void Decode_ParsesJsonArraySegments()
    {
        string raw = """
        Sure, here is the transcript:
        [{"Start time": 0.0, "End time": 1.5, "Speaker ID": "1", "Content": "Hello there."},
         {"Start time": 1.5, "End time": 3.0, "Speaker ID": "2", "Content": "Hi!"}]
        """;

        var decoded = VibeVoicePostprocessor.Decode(raw);

        Assert.Equal(2, decoded.Segments.Count);
        Assert.Equal("Hello there.", decoded.Segments[0].Text);
        Assert.Equal("1", decoded.Segments[0].SpeakerId);
        Assert.Equal(0.0, decoded.Segments[0].StartTime);
        Assert.Equal(1.5, decoded.Segments[0].EndTime);
        Assert.Equal("Hi!", decoded.Segments[1].Text);
        Assert.Equal("Hello there. Hi!", decoded.Text);
    }

    [Fact]
    public void Decode_AcceptsFallbackKeyAliases()
    {
        string raw = """[{"Start": 0, "End": 2, "Speaker": "A", "text": "fallback keys"}]""";
        var decoded = VibeVoicePostprocessor.Decode(raw);
        Assert.Single(decoded.Segments);
        Assert.Equal("fallback keys", decoded.Segments[0].Text);
        Assert.Equal("A", decoded.Segments[0].SpeakerId);
    }

    [Fact]
    public void Decode_NoJsonPayload_FallsBackToTrimmedRawText()
    {
        var decoded = VibeVoicePostprocessor.Decode("  just plain text, no json  ");
        Assert.Empty(decoded.Segments);
        Assert.Equal("just plain text, no json", decoded.Text);
    }

    [Fact]
    public void Decode_MalformedJson_FallsBackToTrimmedRawText()
    {
        var decoded = VibeVoicePostprocessor.Decode("[{\"Content\": \"unterminated");
        Assert.Empty(decoded.Segments);
        Assert.Equal("[{\"Content\": \"unterminated", decoded.Text);
    }
}
