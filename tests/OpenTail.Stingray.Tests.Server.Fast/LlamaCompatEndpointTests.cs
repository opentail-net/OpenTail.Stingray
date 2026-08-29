
namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>Wire contracts for the small llama-server compatibility endpoints. These endpoints
/// must remain useful without becoming a second inference protocol.</summary>
public sealed class LlamaCompatEndpointTests : IDisposable
{
    private readonly List<WebApplicationFactory<Program>> _factories = new();

    public void Dispose()
    {
        foreach (var factory in _factories) factory.Dispose();
    }

    private HttpClient CreateClient()
    {
        var factory = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(builder => builder.ConfigureServices(services =>
            {
                services.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"));
                services.AddSingleton<ITokenizer>(new TestTokenizer());
            }));
        _factories.Add(factory);
        return factory.CreateClient();
    }

    [Fact]
    public async Task Tokenize_EncodesAndOptionallyPrependsBos()
    {
        var response = await CreateClient().PostAsJsonAsync("/tokenize", new { content = "ab", add_special = true });
        response.EnsureSuccessStatusCode();
        using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal(3, json.RootElement.GetProperty("n_tokens").GetInt32());
        Assert.Equal([1, 10, 11], json.RootElement.GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32()));
    }

    [Fact]
    public async Task Detokenize_DecodesTokenIds()
    {
        var response = await CreateClient().PostAsJsonAsync("/detokenize", new { tokens = new[] { 10, 11 } });
        response.EnsureSuccessStatusCode();
        using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("ab", json.RootElement.GetProperty("content").GetString());
    }

    [Theory]
    [InlineData("/tokenize")]
    [InlineData("/detokenize")]
    public async Task TokenEndpoints_RejectMissingRequiredPayload(string path)
    {
        var response = await CreateClient().PostAsJsonAsync(path, new { });
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, response.StatusCode);
        using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("invalid_request_error", json.RootElement.GetProperty("type").GetString());
    }

    [Fact]
    public async Task Props_ReportsSafeModelFacts()
    {
        var response = await CreateClient().GetAsync("/props");
        response.EnsureSuccessStatusCode();
        using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("test-model", json.RootElement.GetProperty("model").GetString());
        Assert.Equal(64, json.RootElement.GetProperty("vocab_size").GetInt32());
        Assert.True(json.RootElement.GetProperty("thinking_enabled").GetBoolean());
    }

    private sealed class TestTokenizer : ITokenizer
    {
        public IReadOnlyList<int> Encode(string text) => text.Select(c => c switch { 'a' => 10, 'b' => 11, _ => 0 }).ToArray();
        public string Decode(IEnumerable<int> tokens) => new(tokens.Select(i => i switch { 10 => 'a', 11 => 'b', _ => '?' }).ToArray());
        public byte[] DecodeBytes(int token) => [(byte)Decode([token])[0]];
        public int VocabSize => 64;
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public ImmutableArray<int> EogTokenIds => [EosTokenId];
        public bool AddBosToken => true;
    }
}
