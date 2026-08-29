using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Server.Endpoints;
using Xunit;

namespace OpenTail.Stingray.Tests.Server;

public sealed class OpenAiImageEndpointsTests
{
    [Fact]
    public void ImageGenerationRequest_DeserializesCorrectly()
    {
        string json = """
        {
            "prompt": "A scenic landscape at sunrise",
            "n": 2,
            "size": "1024x1024",
            "response_format": "b64_json",
            "quality": "hd"
        }
        """;

        var req = JsonSerializer.Deserialize<OpenAiImageEndpoints.ImageGenerationRequest>(json);
        Assert.NotNull(req);
        Assert.Equal("A scenic landscape at sunrise", req.Prompt);
        Assert.Equal(2, req.N);
        Assert.Equal("1024x1024", req.Size);
        Assert.Equal("b64_json", req.ResponseFormat);
        Assert.Equal("hd", req.Quality);
    }

    [Fact]
    public void ImageApiResponse_SerializesToStandardOpenAiJson()
    {
        var response = new OpenAiImageEndpoints.ImageApiResponse
        {
            Created = 1710000000,
            Data =
            [
                new OpenAiImageEndpoints.ImageResponseItem
                {
                    B64Json = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    RevisedPrompt = "A scenic landscape at sunrise"
                }
            ]
        };

        string json = JsonSerializer.Serialize(response);
        Assert.Contains("\"created\":1710000000", json);
        Assert.Contains("\"b64_json\":", json);
        Assert.Contains("\"revised_prompt\":", json);
        Assert.DoesNotContain("\"url\":", json);
    }

    [Fact]
    public void PngWriter_EncodesValidPngBytes()
    {
        int w = 16;
        int h = 16;
        var rgb = new float[w * h * 3];
        Array.Fill(rgb, 0.5f);

        string tempPath = Path.Combine(Path.GetTempPath(), $"test_png_{Guid.NewGuid():N}.png");
        try
        {
            PngWriter.Write(tempPath, rgb, w, h);
            Assert.True(File.Exists(tempPath));

            byte[] bytes = File.ReadAllBytes(tempPath);
            Assert.True(bytes.Length > 8);
            // Verify PNG magic number
            Assert.Equal(0x89, bytes[0]);
            Assert.Equal((byte)'P', bytes[1]);
            Assert.Equal((byte)'N', bytes[2]);
            Assert.Equal((byte)'G', bytes[3]);
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }
}
