using System.IO;
using Xunit;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// E3: the listing that pull and scout read from the Hugging Face API, when the response is not a usable listing. Each failure must
/// name what happened, so the user knows whether to wait, check the name, or fetch a local copy.
/// </summary>
public sealed class PullListingFailureTests
{
    [Fact]
    public void A_NonJsonResponse_IsReportedAsAListingProblem_NotAParserError()
    {
        // A rate-limit or proxy page arrives as HTML with a 200 status; the raw parser message would say nothing useful.
        var ex = Assert.Throws<InvalidDataException>(() =>
            PullCommand.ParseListingJson("<html><body>Too many requests</body></html>", null, out _));
        Assert.Contains("did not return a model listing", ex.Message);
        Assert.Contains("Try again later", ex.Message);
    }

    [Fact]
    public void AnEmptyObject_IsAListingWithNoGgufFiles_AndNoRevision()
    {
        var files = PullCommand.ParseListingJson("{}", null, out var revision);
        Assert.Empty(files);
        Assert.Null(revision);
    }

    [Fact]
    public void AListingWithoutSiblings_IsEmpty_NotAnError()
    {
        var files = PullCommand.ParseListingJson("{\"sha\":\"abcdef1234567890\",\"id\":\"o/r\"}", null, out var revision);
        Assert.Empty(files);
        Assert.Equal("abcdef1234567890", revision);
    }

    [Fact]
    public void AListingWithAGgufAndItsHash_ReturnsTheFileAndRecordsTheHash()
    {
        const string json = "{\"sha\":\"abcdef1234567890\",\"siblings\":[{\"rfilename\":\"m-Q4_K_M.gguf\",\"size\":42,"
            + "\"lfs\":{\"sha256\":\"6EB923E7D26E9CEA28811E1A8E852009B21242FB157B26149D3B188F3A8C8653\",\"size\":42}}]}";
        var hashes = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var files = PullCommand.ParseListingJson(json, hashes, out var revision);
        Assert.Equal("m-Q4_K_M.gguf", Assert.Single(files).Name);
        Assert.Equal("abcdef1234567890", revision);
        Assert.Equal("6EB923E7D26E9CEA28811E1A8E852009B21242FB157B26149D3B188F3A8C8653", hashes["m-Q4_K_M.gguf"], ignoreCase: true);
    }
}
