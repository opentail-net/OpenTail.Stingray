using System;
using System.Text;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Lightweight, transport-safe continuation token representing a versioned handle to a session's last committed checkpoint.
/// <para>
/// <b>No Silent Rewind:</b> Storing a monotonic <see cref="Generation"/> counter ensures that attempting to continue
/// an advanced/mutated session using an older token throws <see cref="StaleContinuationException"/> rather than silently rewinding state.
/// </para>
/// </summary>
public readonly record struct ResponseContinuationToken(
    SessionId SessionId,
    long TokenPosition,
    long Generation)
{
    /// <summary>Encodes the token into a compact Base64URL string for transport over REST/gRPC APIs.</summary>
    public string Encode()
    {
        string raw = $"{SessionId}:{TokenPosition}:{Generation}";
        byte[] bytes = Encoding.UTF8.GetBytes(raw);
        return Convert.ToBase64String(bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=');
    }

    /// <summary>Parses an encoded Base64URL string back into a <see cref="ResponseContinuationToken"/>.</summary>
    public static ResponseContinuationToken Parse(string value)
    {
        ArgumentException.ThrowIfNullOrEmpty(value);
        string incoming = value.Replace('-', '+').Replace('_', '/');
        switch (incoming.Length % 4)
        {
            case 2: incoming += "=="; break;
            case 3: incoming += "="; break;
        }

        byte[] bytes = Convert.FromBase64String(incoming);
        string raw = Encoding.UTF8.GetString(bytes);
        string[] parts = raw.Split(':');

        if (parts.Length != 3 ||
            !Guid.TryParse(parts[0], out Guid parsedGuid) ||
            !long.TryParse(parts[1], out long pos) ||
            !long.TryParse(parts[2], out long gen))
        {
            throw new FormatException($"Invalid continuation token format: '{value}'.");
        }

        return new ResponseContinuationToken(new SessionId(parsedGuid), pos, gen);
    }
}
