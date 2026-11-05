using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// /v1/responses in multi-model mode: the unknown-model (404) and insufficient-resources (503) answers carry the same JSON
/// error envelope as every other error on the endpoint, not an empty body.
/// </summary>
public sealed class ResponsesMultiModelErrorTests
{
    private const string KnownAlias = "known";

    [Fact]
    public async Task UnknownModel_Is404_WithTheJsonErrorEnvelope()
    {
        using var factory = CreateFactory(new FakeMultiModelService(insufficient: false));
        using var client = factory.CreateClient();

        using var response = await PostResponsesAsync(client, "nope");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        var body = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(body);
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("type").GetString());
        Assert.Contains("nope", doc.RootElement.GetProperty("message").GetString());
    }

    [Fact]
    public async Task InsufficientResources_Is503_WithTheJsonErrorEnvelope()
    {
        using var factory = CreateFactory(new FakeMultiModelService(insufficient: true));
        using var client = factory.CreateClient();

        using var response = await PostResponsesAsync(client, KnownAlias);

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        var body = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(body);
        Assert.Equal("insufficient_resources", doc.RootElement.GetProperty("type").GetString());
        Assert.False(string.IsNullOrWhiteSpace(doc.RootElement.GetProperty("message").GetString()));
    }

    private static Task<HttpResponseMessage> PostResponsesAsync(HttpClient client, string model)
    {
        var json = JsonSerializer.Serialize(new { model, input = "hi", max_output_tokens = 4 });
        return client.PostAsync("/v1/responses", new StringContent(json, Encoding.UTF8, "application/json"));
    }

    private static WebApplicationFactory<Program> CreateFactory(IInferenceService service) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceService>(service)));

    private sealed class FakeMultiModelService(bool insufficient) : IInferenceService
    {
        public bool IsMultiModel => true;

        public IReadOnlyList<string> AvailableModelAliases { get; } = [KnownAlias];

        public ModelId ResolveModel(string? requestedModel)
        {
            if (requestedModel is null || !string.Equals(requestedModel, KnownAlias, StringComparison.OrdinalIgnoreCase))
                throw new ModelNotFoundException(requestedModel ?? "", AvailableModelAliases);
            return new ModelId(KnownAlias);
        }

        public ValueTask<ModelRuntimeHandle> AcquireAsync(ModelId model, CancellationToken ct = default)
        {
            if (!insufficient) throw new InvalidOperationException("the fake only supports the insufficient-resources path");
            throw new InsufficientResourcesException(
                model, 1, 0, ResourceAdmission.InsufficientHostMemory,
                new ResourceSnapshot(1, 0, null, 0, 0));
        }
    }
}
