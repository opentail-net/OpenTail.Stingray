namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// How every acquisition site reports a <see cref="NotSupportedException"/> thrown while a model loads (an architecture or checkpoint layout the
/// engine refuses, such as the Gemma 4 mixture-of-experts file): a client error that carries the refusal text, not a bare 500.
/// </summary>
internal static class ModelLoadRefusal
{
    public const int StatusCode = StatusCodes.Status400BadRequest;
    public const string ErrorType = "invalid_request_error";
}
