
namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Concrete thread-safe implementation of <see cref="ISessionMetadata"/> backed by a concurrent dictionary.
/// </summary>
public sealed class SessionMetadata : ISessionMetadata
{
    private readonly ConcurrentDictionary<string, object?> _entries;

    public SessionMetadata()
    {
        _entries = new ConcurrentDictionary<string, object?>(StringComparer.Ordinal);
    }

    private SessionMetadata(IEnumerable<KeyValuePair<string, object?>> initialEntries)
    {
        _entries = new ConcurrentDictionary<string, object?>(initialEntries, StringComparer.Ordinal);
    }

    public object? Get(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        _entries.TryGetValue(key, out var val);
        return val;
    }

    public T? Get<T>(string key)
    {
        if (TryGet<T>(key, out var val))
        {
            return val;
        }
        return default;
    }

    public void Set(string key, object? value)
    {
        ArgumentNullException.ThrowIfNull(key);
        _entries[key] = value;
    }

    public bool Remove(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return _entries.TryRemove(key, out _);
    }

    public bool TryGet<T>(string key, out T? value)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (_entries.TryGetValue(key, out var rawValue) && rawValue is T typedValue)
        {
            value = typedValue;
            return true;
        }

        value = default;
        return false;
    }

    public object? this[string key]
    {
        get => Get(key);
        set => Set(key, value);
    }

    public ISessionMetadata Clone()
    {
        return new SessionMetadata(_entries.ToArray());
    }

    public IReadOnlyDictionary<string, object?> GetEntries()
    {
        return new Dictionary<string, object?>(_entries, StringComparer.Ordinal);
    }
}
