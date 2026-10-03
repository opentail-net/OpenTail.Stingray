using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using OpenTail.Stingray.Server.Endpoints;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Wire-format tests for <c>/v1/embeddings</c> (OpenAI) and <c>/api/embed</c> (Ollama). None needs an encoder model:
/// validation happens before model resolution, and a valid request with no model configured answers 404 rather than
/// producing synthetic vectors. Numerical correctness of the encoders is covered by the Embeddings test project.
/// </summary>
public sealed class EmbeddingEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public EmbeddingEndpointTests(WebApplicationFactory<Program> factory) => _factory = factory;

    private static JsonElement Json(string s) => JsonDocument.Parse(s).RootElement.Clone();

    // ── The parser, with no HTTP ─────────────────────────────────────────────

    [Theory]
    [InlineData("\"one\"", new[] { "one" })]
    [InlineData("[\"one\",\"two\"]", new[] { "one", "two" })]
    public void TryParseInput_AcceptsStringOrStringArray(string json, string[] expected)
    {
        Assert.True(OpenAiEmbeddingEndpoints.TryParseInput(Json(json), out var texts, out var error), error);
        Assert.Equal(expected, texts);
    }

    [Theory]
    [InlineData("[]")]                // empty array
    [InlineData("\"\"")]              // empty string
    [InlineData("[\"ok\",\"\"]")]     // empty member
    [InlineData("[1,2,3]")]           // token ids are not supported
    [InlineData("[\"ok\",3]")]        // mixed
    [InlineData("42")]                // wrong type
    [InlineData("null")]
    public void TryParseInput_RejectsInvalid_InsteadOfDroppingSilently(string json)
    {
        Assert.False(OpenAiEmbeddingEndpoints.TryParseInput(Json(json), out _, out var error));
        Assert.False(string.IsNullOrEmpty(error));
    }

    // ── HTTP ─────────────────────────────────────────────────────────────────

    private async Task<(HttpStatusCode Status, string Body)> Post(string route, string body)
    {
        Environment.SetEnvironmentVariable("STINGRAY_EMBEDDING_MODEL", null); // no default encoder for these tests
        var client = _factory.CreateClient();
        using var content = new StringContent(body, Encoding.UTF8, "application/json");
        var resp = await client.PostAsync(route, content);
        return (resp.StatusCode, await resp.Content.ReadAsStringAsync());
    }

    [Theory]
    [InlineData("/v1/embeddings")]
    [InlineData("/api/embed")]
    public async Task ArrayInput_IsAcceptedByValidation_AndReportsMissingModelAs404(string route)
    {
        // Before the fix the standard {"input": [...]} form threw a JsonException and came back as a 400 "Invalid JSON".
        var (status, body) = await Post(route, "{\"model\":\"does-not-exist\",\"input\":[\"one\",\"two\"]}");
        Assert.Equal(HttpStatusCode.NotFound, status);
        Assert.Contains("does-not-exist", body);
    }

    [Theory]
    [InlineData("/v1/embeddings", "{\"model\":\"m\"}")]
    [InlineData("/v1/embeddings", "{\"model\":\"m\",\"input\":[]}")]
    [InlineData("/v1/embeddings", "{\"model\":\"m\",\"input\":\"\"}")]
    [InlineData("/v1/embeddings", "{\"model\":\"m\",\"input\":[1,2]}")]
    [InlineData("/v1/embeddings", "{not json")]
    [InlineData("/api/embed", "{\"model\":\"m\"}")]
    [InlineData("/api/embed", "{\"model\":\"m\",\"input\":[\"a\",\"\"]}")]
    [InlineData("/api/embed", "{not json")]
    public async Task InvalidRequests_AreRejectedWith400(string route, string body)
    {
        var (status, _) = await Post(route, body);
        Assert.Equal(HttpStatusCode.BadRequest, status);
    }

    [Fact]
    public async Task ErrorShapes_FollowEachApiConvention()
    {
        var (_, openAi) = await Post("/v1/embeddings", "{\"model\":\"m\"}");
        using (var doc = JsonDocument.Parse(openAi))
        {
            var err = doc.RootElement.GetProperty("error");
            Assert.Equal("invalid_request_error", err.GetProperty("type").GetString());
            Assert.True(err.TryGetProperty("message", out _));
        }

        var (_, ollama) = await Post("/api/embed", "{\"model\":\"m\"}");
        using (var doc = JsonDocument.Parse(ollama))
        {
            Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("error").ValueKind); // Ollama: {"error": "..."}
        }
    }
}
