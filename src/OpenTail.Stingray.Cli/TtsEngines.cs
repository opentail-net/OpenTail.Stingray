namespace OpenTail.Stingray.Cli;

/// <summary>
/// The TTS engines <c>stingray tts</c> can drive: one list shared by the <c>--engine</c> help text and the
/// unknown-engine error (they had drifted: the help named 5 engines while the code accepted 12), the accepted
/// aliases, and which engines have a sampling step that <c>--seed</c> controls.
/// </summary>
internal static class TtsEngines
{
    /// <summary>Canonical engine names in help-text order.</summary>
    public const string Names = "kokoro (default), piper, f5tts, chatterbox, melo, cosyvoice, parler, qwentts, fishspeech, orpheus, mms, xtts";

    private static readonly Dictionary<string, string> Aliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["kokoro"] = "kokoro",
        ["piper"] = "piper",
        ["f5"] = "f5tts", ["f5tts"] = "f5tts", ["f5-tts"] = "f5tts",
        ["chatterbox"] = "chatterbox", ["chatterbox-turbo"] = "chatterbox",
        ["melo"] = "melo", ["melotts"] = "melo",
        ["cosyvoice"] = "cosyvoice", ["cosyvoice3"] = "cosyvoice", ["cosy"] = "cosyvoice",
        ["parler"] = "parler", ["parler-tts"] = "parler", ["parlertts"] = "parler",
        ["qwen"] = "qwentts", ["qwentts"] = "qwentts", ["qwen-tts"] = "qwentts", ["qwen-talker"] = "qwentts",
        ["fish"] = "fishspeech", ["fishspeech"] = "fishspeech", ["fish-speech"] = "fishspeech", ["s2"] = "fishspeech", ["s2-pro"] = "fishspeech",
        ["orpheus"] = "orpheus", ["orpheus-tts"] = "orpheus", ["orpheustts"] = "orpheus",
        ["mms"] = "mms", ["mms-tts"] = "mms", ["mmstts"] = "mms",
        ["xtts"] = "xtts", ["xtts-v2"] = "xtts", ["xttsv2"] = "xtts",
    };

    /// <summary>The canonical engine name for an accepted spelling, or null if unknown.</summary>
    public static string? Canonical(string name) => Aliases.TryGetValue(name.Trim(), out var c) ? c : null;

    /// <summary>Whether <c>--seed</c> changes this engine's output (the engine has a sampling step wired to <c>AudioGenerationRequest.Seed</c>).</summary>
    public static bool HonorsSeed(string canonical) => canonical is "fishspeech" or "parler" or "qwentts" or "xtts" or "mms" or "cosyvoice";

    /// <summary>Every canonical name (without the "(default)" annotation), for tests and tooling.</summary>
    public static IReadOnlyList<string> CanonicalNames { get; } = Aliases.Values.Distinct().ToList();
}
