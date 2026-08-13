using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Sessions;

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(InferenceSessionSnapshot))]
internal partial class InferenceSessionJsonContext : JsonSerializerContext
{
}

/// <summary>
/// File-system backed persistent session store writing <see cref="InferenceSessionSnapshot"/> files to disk.
/// </summary>
public sealed class FileSessionStore : ISessionStore
{
    private readonly string _storageDirectory;

    public FileSessionStore(string? storageDirectory = null)
    {
        _storageDirectory = storageDirectory ?? Path.Combine(Directory.GetCurrentDirectory(), ".stingray", "sessions");
        Directory.CreateDirectory(_storageDirectory);
    }

    public string StorageDirectory => _storageDirectory;

    public async ValueTask SaveAsync(InferenceSessionSnapshot snapshot, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        string finalPath = GetFilePath(snapshot.Id);
        string tempPath = finalPath + ".tmp";
        try
        {
            string json = JsonSerializer.Serialize(snapshot, InferenceSessionJsonContext.Default.InferenceSessionSnapshot);
            await File.WriteAllTextAsync(tempPath, json, cancellationToken).ConfigureAwait(false);
            // Atomic replace: if this process is killed between Write and Move,
            // the .tmp file is left orphaned but the original checkpoint is intact.
            File.Move(tempPath, finalPath, overwrite: true);
        }
        catch
        {
            try { File.Delete(tempPath); } catch { /* best-effort cleanup of the temp file */ }
            throw;
        }
    }

    public async ValueTask SaveDeltaAsync(SessionDelta delta, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(delta);
        var snapshot = await LoadAsync(delta.SessionId, cancellationToken).ConfigureAwait(false);

        if (snapshot is null)
        {
            if (delta.BaseToken.TokenPosition != 0)
            {
                throw new InvalidOperationException($"Cannot apply delta with base position {delta.BaseToken.TokenPosition} to non-existent session snapshot '{delta.SessionId}'.");
            }

            snapshot = new InferenceSessionSnapshot
            {
                Id = delta.SessionId,
                Tokens = delta.AppendedTokens ?? Array.Empty<int>(),
                Position = delta.AppendedTokens?.Count ?? 0,
                Generation = delta.ResultToken.Generation,
                SavedAt = DateTimeOffset.UtcNow
            };
        }
        else
        {
            if (snapshot.Position != delta.BaseToken.TokenPosition || snapshot.Generation != delta.BaseToken.Generation)
            {
                throw new StaleContinuationException(
                    delta.SessionId,
                    delta.BaseToken.TokenPosition,
                    delta.BaseToken.Generation,
                    snapshot.Position,
                    snapshot.Generation);
            }

            var mergedTokens = new System.Collections.Generic.List<int>(snapshot.Tokens);
            if (delta.AppendedTokens is { Count: > 0 } newTokens)
            {
                mergedTokens.AddRange(newTokens);
            }

            snapshot = snapshot with
            {
                Tokens = mergedTokens,
                Position = mergedTokens.Count,
                Generation = delta.ResultToken.Generation,
                SavedAt = DateTimeOffset.UtcNow
            };
        }

        await SaveAsync(snapshot, cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask<InferenceSessionSnapshot?> LoadAsync(SessionId id, CancellationToken cancellationToken = default)
    {
        string filePath = GetFilePath(id);
        if (!File.Exists(filePath))
        {
            return null;
        }

        string json = await File.ReadAllTextAsync(filePath, cancellationToken).ConfigureAwait(false);
        return JsonSerializer.Deserialize(json, InferenceSessionJsonContext.Default.InferenceSessionSnapshot);
    }

    public ValueTask<bool> DeleteAsync(SessionId id, CancellationToken cancellationToken = default)
    {
        string filePath = GetFilePath(id);
        if (File.Exists(filePath))
        {
            File.Delete(filePath);
            return ValueTask.FromResult(true);
        }
        return ValueTask.FromResult(false);
    }

    private string GetFilePath(SessionId id) => Path.Combine(_storageDirectory, $"{id.Value:N}.session.json");
}
