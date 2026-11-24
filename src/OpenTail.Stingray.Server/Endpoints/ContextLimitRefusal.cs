using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// How every endpoint reports a <see cref="ContextLimitExceededException"/>: a 400 that names the context length, with
/// OpenAI's <c>context_length_exceeded</c> code where the wire format has one. A prompt longer than the window is the
/// client's input, not a server fault, so it must not reach the client as a bare 500.
/// </summary>
internal static class ContextLimitRefusal
{
    public const int StatusCode = StatusCodes.Status400BadRequest;
    public const string ErrorType = "invalid_request_error";
    public const string ErrorCode = "context_length_exceeded";

    public static string Message(ContextLimitExceededException ex) =>
        $"This model's maximum context length is {ex.MaxContextTokens} tokens, and the request needs " +
        $"{ex.CurrentTokens + ex.RequestedTokens} tokens. Shorten the prompt or the conversation history.";
}
