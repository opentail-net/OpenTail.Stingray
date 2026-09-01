using Xunit;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// `list-env` prints setting VALUES, so it is the one place in this surface where a credential
/// could be echoed to a terminal, a screenshot, or a pasted support report. The QoL plan's §10
/// privacy tests require that secrets never serialize; these pin that.
///
/// <para>Redaction keys off the NAME, not the value, because a value's shape proves nothing — an
/// access token is indistinguishable from any other opaque string.</para>
/// </summary>
public sealed class ListEnvRedactionTests
{
    [Theory]
    [InlineData("STINGRAY_HF_TOKEN")]
    [InlineData("STINGRAY_API_KEY")]
    [InlineData("STINGRAY_CLIENT_SECRET")]
    [InlineData("STINGRAY_DB_PASSWORD")]
    [InlineData("STINGRAY_AZURE_CREDENTIAL")]
    public void CredentialShapedNamesAreRedacted(string name) =>
        Assert.True(ListEnvCommand.IsSensitive(name), $"{name} must be redacted but was not");

    /// <summary>Matching is on a substring, so a suffixed or prefixed variant is still caught.</summary>
    [Theory]
    [InlineData("STINGRAY_HF_TOKEN_FILE")]
    [InlineData("STINGRAY_EXTRA_API_KEY_PATH")]
    public void CredentialFragmentsAreCaughtAnywhereInTheName(string name) =>
        Assert.True(ListEnvCommand.IsSensitive(name), $"{name} must be redacted but was not");

    /// <summary>
    /// Over-redaction is a real cost too: the command exists to show values, and hiding ordinary
    /// settings would make it useless. These must NOT be masked.
    /// </summary>
    [Theory]
    [InlineData("STINGRAY_CPU_THREADS")]
    [InlineData("STINGRAY_KV_DTYPE")]
    [InlineData("STINGRAY_MODEL")]
    [InlineData("STINGRAY_MAX_BATCH")]
    public void OrdinarySettingsAreNotRedacted(string name) =>
        Assert.False(ListEnvCommand.IsSensitive(name), $"{name} was redacted but carries no secret");

    /// <summary>
    /// Deliberate, reviewed exception to the "nothing should trip the filter" rule below.
    /// <c>STINGRAY_DBG_TOKEN_RANK</c> holds a numeric LLM token ID for a <c>--verbose-prompt</c>
    /// debug probe (see <c>docs/env-var-inventory.md</c>), not an authentication token — the name
    /// collides with the credential filter's "TOKEN" substring purely coincidentally. Reviewed and
    /// accepted 2026-09-01: masking its value in <c>list-env</c> is harmless (it's a debug integer,
    /// nobody needs to read it off a screenshot), so over-redaction here costs nothing, and
    /// renaming would break continuity with the historical investigation docs that reference this
    /// exact name. Do not add further names here without the same explicit review.
    /// </summary>
    private static readonly string[] KnownFalsePositives = ["STINGRAY_DBG_TOKEN_RANK"];

    /// <summary>
    /// Nothing in the engine's real known-variable surface should currently trip the credential
    /// filter, except the reviewed exceptions in <see cref="KnownFalsePositives"/>. If this fails
    /// with a name not in that list, a genuinely secret-bearing variable has been added — decide
    /// deliberately whether it belongs in the environment at all, then update this test.
    /// </summary>
    [Fact]
    public void NoKnownEngineVariableCurrentlyLooksLikeACredential()
    {
        var flagged = KnownEnvironmentVariables.All.Where(ListEnvCommand.IsSensitive)
            .Except(KnownFalsePositives).ToList();
        Assert.True(flagged.Count == 0,
            "These known variables match the credential filter and would be redacted: " +
            string.Join(", ", flagged));
    }
}
