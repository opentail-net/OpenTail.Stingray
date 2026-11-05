using System.IO;
using OpenTail.Stingray.Core.Net;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// E3: a Hub model-API body that is not usable (not JSON, or JSON without the commit SHA scout needs). Scout reads these through
/// <see cref="HubClient.ParseRepoJson"/>; each failure must say what happened and what to do.
/// </summary>
public sealed class HubListingFailureTests
{
    [Fact]
    public void A_NonJsonBody_IsReportedAsAListingProblem_NotAParserError()
    {
        var ex = Assert.Throws<InvalidDataException>(() =>
            HubClient.ParseRepoJson("<html><body>Rate limit reached</body></html>", "owner/name"));
        Assert.Contains("did not return a model listing for 'owner/name'", ex.Message);
        Assert.Contains("Try again later", ex.Message);
    }

    [Fact]
    public void JsonWithoutACommitSha_IsRefusedWithAClearReason()
    {
        var ex = Assert.Throws<InvalidDataException>(() => HubClient.ParseRepoJson("{\"id\":\"owner/name\"}", "owner/name"));
        Assert.Contains("no commit SHA", ex.Message);
    }
}
