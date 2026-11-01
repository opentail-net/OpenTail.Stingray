using System.Text.Json;
using OpenTail.Stingray.Audio.Whisper;
using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// The <c>model</c> field of <c>/v1/audio/transcriptions</c> and <c>/v1/audio/translations</c> (plan 5 E3, then the owner's decision of 2026-11-01 to run
/// real weights): it is resolved against the catalog's transcribe entries and the aliases <c>whisper-1</c> and <c>transcribe</c>; an unknown id is a 400
/// that names the field and lists the valid ids; the resolved model is what the route asks the pipeline provider for; and a valid model whose weights are
/// not installed is a 503 that says how to install it, never a transcript from a weightless placeholder. These tests need no weights: the provider is
/// replaced, or pointed at an empty model home. The real-weights run is <c>TranscriptionRealWeightsTests</c> in Tests.Server.
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

    /// <summary>Hands out the structure-only placeholder pipeline and records which model it was asked for.</summary>
    private sealed class RecordingProvider : IWhisperPipelineProvider
    {
        public List<string?> Requested { get; } = [];
        public WhisperPipeline Create(string? model) { Requested.Add(model); return new WhisperPipeline(); }
    }

    private static async Task<HttpResponseMessage> PostAsync(string route, string? model, IWhisperPipelineProvider provider)
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s =>
            {
                s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"));
                s.AddSingleton(provider);
            }));
        using var client = factory.CreateClient();
        using var form = new MultipartFormDataContent { { new ByteArrayContent(Wav()), "file", "a.wav" } };
        if (model is not null) form.Add(new StringContent(model), "model");
        var response = await client.PostAsync(route, form);
        factory.Dispose();
        return response;
    }

    private static string EmptyModelHome() => Directory.CreateTempSubdirectory("opentail_empty_model_home_").FullName;

    [Theory]
    [InlineData("/v1/audio/transcriptions")]
    [InlineData("/v1/audio/translations")]
    public async Task UnknownModel_IsA400_NamingTheField_AndListingTheValidIds_AndNeverReachesTheProvider(string route)
    {
        var provider = new RecordingProvider();
        using var response = await PostAsync(route, "no-such-model", provider);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("type").GetString());
        string message = doc.RootElement.GetProperty("message").GetString()!;
        Assert.Contains("model", message);
        Assert.Contains("no-such-model", message);
        foreach (string id in new[] { "whisper-base", "whisper-1", "transcribe" }) Assert.Contains(id, message);
        Assert.Contains("model", doc.RootElement.GetProperty("error").GetProperty("message").GetString());
        Assert.Empty(provider.Requested);
    }

    [Theory]
    [InlineData("/v1/audio/transcriptions", "whisper-base")]
    [InlineData("/v1/audio/transcriptions", "whisper-1")]
    [InlineData("/v1/audio/translations", "whisper-base")]
    [InlineData("/v1/audio/transcriptions", null)]
    public async Task KnownAliasOrOmittedModel_IsAccepted_AndTheRequestedModelReachesTheProvider(string route, string? model)
    {
        var provider = new RecordingProvider();
        using var response = await PostAsync(route, model, provider);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal([model], provider.Requested);
    }

    [Theory]
    [InlineData("/v1/audio/transcriptions", "whisper-base")]
    [InlineData("/v1/audio/transcriptions", "whisper-1")]
    [InlineData("/v1/audio/transcriptions", null)]
    [InlineData("/v1/audio/translations", "transcribe")]
    public async Task ValidModelThatIsNotInstalled_Is503_WithTheSetupHint_NotAPlaceholderTranscript(string route, string? model)
    {
        string home = EmptyModelHome();
        try
        {
            using var response = await PostAsync(route, model, new InstalledWhisperPipelineProvider(() => new ModelHome(home)));
            Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
            using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            Assert.Equal("model_not_installed", doc.RootElement.GetProperty("type").GetString());
            string message = doc.RootElement.GetProperty("message").GetString()!;
            Assert.Contains("whisper-base", message);
            Assert.Contains("stingray setup transcribe", message);
            Assert.Contains("not installed", message);
            Assert.Contains("stingray setup transcribe", doc.RootElement.GetProperty("error").GetProperty("message").GetString());
        }
        finally { Directory.Delete(home, recursive: true); }
    }

    [Fact]
    public async Task ModelFileOfTheWrongSize_IsReportedAsIncompleteOrDamaged()
    {
        string home = EmptyModelHome();
        try
        {
            var entry = ModelCatalog.DefaultFor("transcribe")!;
            await File.WriteAllBytesAsync(Path.Combine(home, entry.MainFile.FileName), new byte[1024]);
            using var response = await PostAsync("/v1/audio/transcriptions", "whisper-base",
                new InstalledWhisperPipelineProvider(() => new ModelHome(home)));
            Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
            using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            Assert.Contains("incomplete or damaged", doc.RootElement.GetProperty("message").GetString());
        }
        finally { Directory.Delete(home, recursive: true); }
    }

    [Fact]
    public void InstalledProvider_RefusesAnIdThatIsNotATranscribeModel()
    {
        var provider = new InstalledWhisperPipelineProvider(() => new ModelHome(EmptyModelHome()));
        Assert.Throws<ArgumentException>(() => provider.Create("qwen2.5-0.5b"));   // a chat entry
        Assert.Throws<ArgumentException>(() => provider.Create("no-such-model"));
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
