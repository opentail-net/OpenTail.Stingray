using System.Text.Json;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 5 E3: the <c>model</c> field of <c>/v1/audio/transcriptions</c> and <c>/v1/audio/translations</c> is resolved against the
/// catalog's transcribe entries and the aliases <c>whisper-1</c> and <c>transcribe</c>. An unknown id is a 400 that names the field and
/// lists the valid ids; an omitted field keeps the route's default pipeline. Resolution does not yet change which weights run: there
/// is one transcribe entry, and the route builds its default pipeline either way.
/// </summary>
public sealed class TranscriptionModelFieldTests
{
    private static byte[] Wav(int samples = 8000)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        int dataBytes = samples * 2;
        w.Write("RIFF"u8); w.Write(36 + dataBytes); w.Write("WAVE"u8);
        w.Write("fmt "u8); w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(16000); w.Write(32000); w.Write((short)2); w.Write((short)16);
        w.Write("data"u8); w.Write(dataBytes); w.Write(new byte[dataBytes]);
        w.Flush();
        return ms.ToArray();
    }

    private static async Task<HttpResponseMessage> PostAsync(string route, string? model)
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"))));
        using var client = factory.CreateClient();
        using var form = new MultipartFormDataContent { { new ByteArrayContent(Wav()), "file", "a.wav" } };
        if (model is not null) form.Add(new StringContent(model), "model");
        var response = await client.PostAsync(route, form);
        factory.Dispose();
        return response;
    }

    [Theory]
    [InlineData("/v1/audio/transcriptions")]
    [InlineData("/v1/audio/translations")]
    public async Task UnknownModel_IsA400_NamingTheField_AndListingTheValidIds(string route)
    {
        using var response = await PostAsync(route, "no-such-model");
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("type").GetString());
        string message = doc.RootElement.GetProperty("message").GetString()!;
        Assert.Contains("model", message);
        Assert.Contains("no-such-model", message);
        foreach (string id in new[] { "whisper-base", "whisper-1", "transcribe" }) Assert.Contains(id, message);
        Assert.Contains("model", doc.RootElement.GetProperty("error").GetProperty("message").GetString());
    }

    [Theory]
    [InlineData("/v1/audio/transcriptions", "whisper-base")]
    [InlineData("/v1/audio/transcriptions", "whisper-1")]
    [InlineData("/v1/audio/translations", "whisper-base")]
    [InlineData("/v1/audio/transcriptions", null)]
    public async Task KnownAliasOrOmittedModel_IsAccepted(string route, string? model)
    {
        using var response = await PostAsync(route, model);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public void ValidateTranscriptionModel_ReturnsNullForOmittedAndKnown_AndAMessageOtherwise()
    {
        Assert.Null(OpenAiAudioEndpoints.ValidateTranscriptionModel(null));
        Assert.Null(OpenAiAudioEndpoints.ValidateTranscriptionModel("  "));
        Assert.Null(OpenAiAudioEndpoints.ValidateTranscriptionModel("whisper-base"));
        Assert.Null(OpenAiAudioEndpoints.ValidateTranscriptionModel("Whisper-Base"));
        Assert.Null(OpenAiAudioEndpoints.ValidateTranscriptionModel("whisper-1"));
        Assert.Null(OpenAiAudioEndpoints.ValidateTranscriptionModel("transcribe"));
        Assert.Null(OpenAiAudioEndpoints.ValidateTranscriptionModel(""));
        Assert.NotNull(OpenAiAudioEndpoints.ValidateTranscriptionModel("qwen2.5-0.5b"));   // a chat entry is not a transcribe entry
    }
}
