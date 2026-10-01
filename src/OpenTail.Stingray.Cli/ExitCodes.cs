namespace OpenTail.Stingray.Cli;

/// <summary>
/// Process exit codes. <see cref="Failure"/> stays the catch-all (existing callers and tests rely on 1);
/// <see cref="Usage"/> is for a missing or invalid command-line argument, so a script can tell "fix the
/// invocation" from "the run failed". Adopted command by command; commands not yet migrated return 1 for both.
/// </summary>
internal static class ExitCodes
{
    public const int Success = 0;
    public const int Failure = 1;
    /// <summary>The input uses something this engine cannot execute (e.g. a chat template it cannot run). Predates this class.</summary>
    public const int Unsupported = 2;
    /// <summary>Missing or invalid command-line argument (BSD sysexits EX_USAGE).</summary>
    public const int Usage = 64;
    /// <summary>128 + SIGINT, the conventional shell exit code for an interrupted process.</summary>
    public const int Interrupted = 130;
}
