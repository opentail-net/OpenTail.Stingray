namespace OpenTail.Stingray.Core.Net;

public readonly record struct ExternalDecision(bool Allowed, string Reason, string? HowToEnable);

/// <summary>
/// The single answer to "may this process touch the network?". Everything that leaves the machine asks here (through <see cref="ExternalHttpClient"/>),
/// so there is exactly one place where policy lives.
///
/// The default is ALLOWED. Deny wins whenever it is asked for:
///   1. Offline mode (<c>STINGRAY_OFFLINE</c> or <c>HF_HUB_OFFLINE</c> = 1/true): deny. Predates this class and is kept as is.
///   2. <c>STINGRAY_ALLOW_EXTERNAL</c> = 0/false/off/no/deny: deny.
///   3. Otherwise allow (including <c>STINGRAY_ALLOW_EXTERNAL</c> = 1/true/on/yes/allow, which only states the default out loud).
/// An unrecognised value of <c>STINGRAY_ALLOW_EXTERNAL</c> is treated as a deny rather than guessed at, so a typo cannot silently open the network.
/// </summary>
public static class ExternalAccess
{
    public const string AllowVariable = "STINGRAY_ALLOW_EXTERNAL";
    public static readonly IReadOnlyList<string> OfflineVariables = ["STINGRAY_OFFLINE", "HF_HUB_OFFLINE"];

    private static string? Read(string name) => Environment.GetEnvironmentVariable(name);

    private static bool IsTrue(string? v) => v is "1" || (v?.Equals("true", StringComparison.OrdinalIgnoreCase) ?? false);

    /// <summary>True when an offline variable is set. This is rule 1 only; <see cref="Evaluate"/> is the full answer.</summary>
    public static bool OfflineFromEnvironment(Func<string, string?>? env = null)
    {
        env ??= Read;
        return OfflineVariables.Any(n => IsTrue(env(n)));
    }

    public static ExternalDecision Evaluate(Func<string, string?>? env = null)
    {
        env ??= Read;
        foreach (string n in OfflineVariables)
            if (IsTrue(env(n)))
                return new(false, $"offline mode is on ({n}=1)", $"unset {n}");

        string? raw = env(AllowVariable)?.Trim();
        if (string.IsNullOrEmpty(raw))
            return new(true, "external access is allowed by default", null);

        switch (raw.ToLowerInvariant())
        {
            case "0" or "false" or "off" or "no" or "deny":
                return new(false, $"external access is switched off ({AllowVariable}={raw})", $"unset {AllowVariable}, or set it to 1");
            case "1" or "true" or "on" or "yes" or "allow":
                return new(true, $"external access is allowed ({AllowVariable}={raw})", null);
            default:
                return new(false, $"{AllowVariable}='{raw}' is not recognised; treated as off", $"set {AllowVariable}=1 to allow or {AllowVariable}=0 to deny");
        }
    }

    /// <summary>Only Hugging Face hosts are ever contacted: the API and web host, the hf.co short domain, and their CDN/storage subdomains.</summary>
    public static bool IsAllowedHost(string? host)
    {
        if (string.IsNullOrEmpty(host)) return false;
        host = host.TrimEnd('.').ToLowerInvariant();
        return host is "huggingface.co" or "hf.co"
            || host.EndsWith(".huggingface.co", StringComparison.Ordinal)
            || host.EndsWith(".hf.co", StringComparison.Ordinal);
    }

    /// <summary>The bearer token (<c>HF_TOKEN</c>) goes to the Hub itself and never to a CDN or storage host it redirects to.</summary>
    public static bool MayReceiveToken(string? host) =>
        host is not null && (host.Equals("huggingface.co", StringComparison.OrdinalIgnoreCase) || host.Equals("www.huggingface.co", StringComparison.OrdinalIgnoreCase));
}

/// <summary>Thrown instead of making a request the policy does not allow. The message says why and how to change it.</summary>
public sealed class ExternalAccessDeniedException(string reason, string? howToEnable)
    : IOException(howToEnable is null ? $"External access refused: {reason}." : $"External access refused: {reason}. To allow it: {howToEnable}.")
{
    public string Reason { get; } = reason;
    public string? HowToEnable { get; } = howToEnable;
}
