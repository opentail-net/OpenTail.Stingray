
namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Lightweight application-level metadata bag associated with an <see cref="IInferenceSession"/>.
/// <para>
/// <b>Separation of Concerns:</b> Metadata stores host application context (e.g. user identity, workflow phase,
/// permission scope) and does <i>not</i> participate in KV cache keys, prefix-cache indexing, sampling, or inference determinism.
/// </para>
/// </summary>
public interface ISessionMetadata
{
    /// <summary>Retrieves a metadata value by key, or null if key does not exist.</summary>
    object? Get(string key);

    /// <summary>Retrieves a typed metadata value by key, or default(T) if key does not exist or type does not match.</summary>
    T? Get<T>(string key);

    /// <summary>Sets or updates a metadata key-value entry.</summary>
    void Set(string key, object? value);

    /// <summary>Removes a metadata key entry. Returns true if key existed and was removed.</summary>
    bool Remove(string key);

    /// <summary>Attempts to retrieve a typed metadata value by key. Returns true if key exists and type matches.</summary>
    bool TryGet<T>(string key, out T? value);

    /// <summary>Gets or sets a metadata entry via indexer syntax.</summary>
    object? this[string key] { get; set; }

    /// <summary>Creates an independent metadata container populated with key-value references from this instance.</summary>
    ISessionMetadata Clone();

    /// <summary>Retrieves a read-only snapshot dictionary of all current metadata key-value entries.</summary>
    System.Collections.Generic.IReadOnlyDictionary<string, object?> GetEntries();
}
