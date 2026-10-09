using System.Text.Json;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class PullCommandListingTests
{
    // Shape copied from a real https://huggingface.co/api/models/<repo>?blobs=true response (2026-10-09).
    private const string WithBlobs = """
        {"id":"owner/repo","siblings":[
          {"rfilename":".gitattributes","blobId":"abc","size":1519},
          {"rfilename":"Model-Q4_K_M.gguf","blobId":"66e0eb4f","size":397808192,
           "lfs":{"sha256":"6EB923E7D26E9CEA28811E1A8E852009B21242FB157B26149D3B188F3A8C8653","size":397808192,"pointerSize":134}},
          {"rfilename":"Model-Q8_0.gguf","blobId":"77f0","size":531000000,
           "lfs":{"sha256":"not-a-64-char-hash","size":531000000,"pointerSize":134}},
          {"rfilename":"tiny.gguf","size":10}
        ]}
        """;

    [Fact]
    public void Listing_ReadsGgufNamesSizesAndPublishedSha256()
    {
        var hashes = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        using var doc = JsonDocument.Parse(WithBlobs);
        var files = PullCommand.ParseGgufListing(doc.RootElement, hashes);

        Assert.Equal(["Model-Q4_K_M.gguf", "Model-Q8_0.gguf", "tiny.gguf"], files.Select(f => f.Name).ToArray());
        Assert.Equal(397808192L, files[0].Size);
        Assert.Equal("6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653", hashes["Model-Q4_K_M.gguf"]);   // lowercased
        Assert.False(hashes.ContainsKey("Model-Q8_0.gguf"));       // a malformed hash is not recorded, so it can never "verify" anything
        Assert.False(hashes.ContainsKey("tiny.gguf"));             // not an LFS file: no hash published
    }

    [Fact]
    public void Listing_WithoutBlobsInfo_StillListsFiles_WithNoHashes()
    {
        using var doc = JsonDocument.Parse("""{"siblings":[{"rfilename":"Model-Q4_K_M.gguf"},{"rfilename":"README.md"}]}""");
        var hashes = new Dictionary<string, string>();
        var files = PullCommand.ParseGgufListing(doc.RootElement, hashes);
        Assert.Single(files);
        Assert.Null(files[0].Size);
        Assert.Empty(hashes);
    }

    [Fact]
    public void Listing_WithCommitSha_ParsesGgufFilesAndHashes()
    {
        const string withSha = """
            {
              "id": "owner/repo",
              "sha": "abcdef1234567890abcdef1234567890abcdef12",
              "siblings": [
                {"rfilename": "Model-Q4_K_M.gguf", "size": 12345, "lfs": {"sha256": "6EB923E7D26E9CEA28811E1A8E852009B21242FB157B26149D3B188F3A8C8653"}},
                {"rfilename": "other.txt", "size": 100}
              ]
            }
            """;
        using var doc = JsonDocument.Parse(withSha);
        var hashes = new Dictionary<string, string>();
        var files = PullCommand.ParseGgufListing(doc.RootElement, hashes);
        Assert.Single(files);
        Assert.Equal("Model-Q4_K_M.gguf", files[0].Name);
        Assert.Equal(12345L, files[0].Size);
        Assert.Equal("6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653", hashes["Model-Q4_K_M.gguf"]);
    }

    [Fact]
    public void Listing_WithoutSha_SucceedsWithoutThrowing()
    {
        // HubClient.ParseRepo throws InvalidDataException if 'sha' is missing; ParseGgufListing must tolerate this.
        const string noSha = """
            {
              "id": "owner/repo",
              "siblings": [
                {"rfilename": "Model-Q5_K_M.gguf", "size": 55555, "lfs": {"sha256": "AABBCCDDEEFF00112233445566778899AABBCCDDEEFF00112233445566778899"}}
              ]
            }
            """;
        using var doc = JsonDocument.Parse(noSha);
        var hashes = new Dictionary<string, string>();
        var files = PullCommand.ParseGgufListing(doc.RootElement, hashes);
        Assert.Single(files);
        Assert.Equal("Model-Q5_K_M.gguf", files[0].Name);
        Assert.Equal(55555L, files[0].Size);
        Assert.Equal("aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899", hashes["Model-Q5_K_M.gguf"]);
    }

    [Fact]
    public void Listing_WithoutSiblings_ReturnsEmptyList()
    {
        using var doc = JsonDocument.Parse("""{"id": "owner/repo"}""");
        var hashes = new Dictionary<string, string>();
        var files = PullCommand.ParseGgufListing(doc.RootElement, hashes);
        Assert.Empty(files);
        Assert.Empty(hashes);
    }

    [Fact]
    public void Listing_WhenRootIsNotAnObject_ReturnsEmptyList()
    {
        using var doc = JsonDocument.Parse("[]");
        var hashes = new Dictionary<string, string>();
        var files = PullCommand.ParseGgufListing(doc.RootElement, hashes);
        Assert.Empty(files);
        Assert.Empty(hashes);
    }

    [Fact]
    public void Listing_WithMalformedSibling_SkipsMalformedAndKeepsValid()
    {
        const string json = """
            {
              "siblings": [
                {"size": 100},
                {"rfilename": null},
                {"rfilename": "valid.gguf", "size": 2048}
              ]
            }
            """;
        using var doc = JsonDocument.Parse(json);
        var files = PullCommand.ParseGgufListing(doc.RootElement);
        Assert.Single(files);
        Assert.Equal("valid.gguf", files[0].Name);
        Assert.Equal(2048L, files[0].Size);
    }
}

