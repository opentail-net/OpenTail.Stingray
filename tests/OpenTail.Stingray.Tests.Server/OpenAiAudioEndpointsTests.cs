using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Server.Endpoints;
using Xunit;

namespace OpenTail.Stingray.Tests.Server;

public sealed class OpenAiAudioEndpointsTests
{
    [Fact]
    public void FormatSrt_ProducesStandardSubRipFormatting()
    {
        var segments = new List<SpeechSegment>
        {
            new()
            {
                Id = 1,
                Start = TimeSpan.FromSeconds(1.234),
                End = TimeSpan.FromSeconds(4.567),
                Text = "Hello world!"
            },
            new()
            {
                Id = 2,
                Start = TimeSpan.FromSeconds(5.000),
                End = TimeSpan.FromSeconds(8.123),
                Text = "This is a subtitle test."
            }
        };

        string srt = OpenAiAudioEndpoints.FormatSrt(segments);

        Assert.Contains("1\r\n00:00:01,234 --> 00:00:04,567\r\nHello world!", srt.Replace("\n", "\r\n").Replace("\r\r\n", "\r\n"));
        Assert.Contains("2\r\n00:00:05,000 --> 00:00:08,123\r\nThis is a subtitle test.", srt.Replace("\n", "\r\n").Replace("\r\r\n", "\r\n"));
    }

    [Fact]
    public void FormatVtt_ProducesStandardWebVttFormatting()
    {
        var segments = new List<SpeechSegment>
        {
            new()
            {
                Id = 1,
                Start = TimeSpan.FromSeconds(0.000),
                End = TimeSpan.FromSeconds(2.500),
                Text = "Welcome to Stingray."
            }
        };

        string vtt = OpenAiAudioEndpoints.FormatVtt(segments);

        Assert.StartsWith("WEBVTT", vtt);
        Assert.Contains("00:00:00.000 --> 00:00:02.500", vtt);
        Assert.Contains("Welcome to Stingray.", vtt);
    }
}
