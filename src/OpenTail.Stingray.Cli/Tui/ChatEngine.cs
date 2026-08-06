using OpenTail.TUI.Runtime;

namespace OpenTail.Stingray.Cli.Tui;

/// <summary>
/// Adapts one turn of inference for the chat UI: takes a user message, streams decoded text to
/// a callback, and returns the visible token count.
/// </summary>
/// <remarks>
/// Kept as a delegate-backed shim rather than a new abstraction over the engine, so the TUI
/// consumes exactly the same prefill/decode path as the non-interactive CLI.
/// </remarks>
internal sealed class ChatEngine(Func<string, Action<string>, CancellationToken, int> generate)
{
    /// <summary>Run one turn. <paramref name="onText"/> is called from a background thread.</summary>
    internal int Generate(string prompt, Action<string> onText, CancellationToken ct) =>
        generate(prompt, onText, ct);
}

/// <summary>Entry point for the interactive chat TUI.</summary>
internal static class ChatTui
{
    /// <summary>
    /// True when a full-screen TUI is usable. The renderer takes over the alternate screen
    /// buffer, so it needs a real terminal on both ends — a redirected stdin or stdout means
    /// the caller is scripting and must get the plain line-oriented loop instead.
    /// </summary>
    internal static bool IsSupported =>
        !Console.IsInputRedirected &&
        !Console.IsOutputRedirected &&
        !string.Equals(Environment.GetEnvironmentVariable("TERM"), "dumb", StringComparison.OrdinalIgnoreCase);

    /// <summary>Run the chat UI to completion. Returns the process exit code.</summary>
    internal static int Run(ChatEngine engine, string modelName)
    {
        var app = new TuiApp(new ChatModel(engine, modelName));
        app.RunAsync().GetAwaiter().GetResult();
        return 0;
    }
}
