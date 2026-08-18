using System.Collections.Concurrent;

namespace OpenTail.Stingray.Core.Lora;

/// <summary>
/// Thread-safe registry and cache for fine-tuned LoRA adapters.
/// </summary>
public sealed class LoraRegistry : IDisposable
{
    private readonly ConcurrentDictionary<string, LoraAdapter> _adapters = new(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    public static LoraRegistry Shared { get; } = new();

    public IReadOnlyCollection<LoraAdapter> RegisteredAdapters => _adapters.Values.ToList();

    public void Register(LoraAdapter adapter)
    {
        _adapters[adapter.Id] = adapter;
        if (!string.IsNullOrEmpty(adapter.Path))
        {
            _adapters[adapter.Path] = adapter;
        }
    }

    public bool TryGet(string idOrPath, out LoraAdapter? adapter)
    {
        return _adapters.TryGetValue(idOrPath, out adapter);
    }

    public LoraAdapter GetOrLoad(string path, string? id = null)
    {
        string lookup = id ?? path;
        if (_adapters.TryGetValue(lookup, out var existing))
            return existing;

        var adapter = LoraAdapter.Load(path, id);
        Register(adapter);
        return adapter;
    }

    public void Remove(string idOrPath)
    {
        if (_adapters.TryRemove(idOrPath, out var adapter))
        {
            adapter.Dispose();
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            foreach (var kv in _adapters.Values)
                kv.Dispose();
            _adapters.Clear();
        }
    }
}
