using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio;

/// <summary>
/// Helper for decoding audio.cpp packed GGUF embedded files metadata:
/// <c>audiocpp.embedded_files.names</c>, <c>audiocpp.embedded_files.offsets</c>, and
/// <c>audiocpp.embedded_files.data</c>.
/// </summary>
public static class AudioCppEmbeddedFiles
{
    private const string NamesKey = "audiocpp.embedded_files.names";
    private const string OffsetsKey = "audiocpp.embedded_files.offsets";
    private const string DataKey = "audiocpp.embedded_files.data";

    private static readonly ConditionalWeakTable<GgufModel, Dictionary<string, byte[]>> Cache = new();

    /// <summary>
    /// Reads and decodes all embedded files in the GGUF model metadata into a name-to-bytes dictionary.
    /// Performs boxed data conversion once and validates offset bounds, monotonicity, and array lengths.
    /// Throws <see cref="InvalidDataException"/> if the embedded files metadata is corrupt or malformed.
    /// The result is cached per <see cref="GgufModel"/> instance and its arrays are shared by every
    /// caller (here and via <see cref="TryGet"/>/<see cref="Get"/>): treat them as read-only.
    /// </summary>
    public static IReadOnlyDictionary<string, byte[]> ReadAll(GgufModel model)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        return Cache.GetValue(model, ParseEmbeddedFiles);
    }

    /// <summary>
    /// Returns the list of embedded file names stored in the GGUF metadata, or an empty list if none exist.
    /// </summary>
    public static IReadOnlyList<string> Names(GgufModel model)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        return ReadAll(model).Keys.ToArray();
    }

    /// <summary>
    /// Tries to extract an embedded file by name from the GGUF model metadata.
    /// Returns false if the file is not found. Throws <see cref="InvalidDataException"/> if metadata is malformed.
    /// Matching is case-insensitive.
    /// </summary>
    public static bool TryGet(GgufModel model, string name, [NotNullWhen(true)] out byte[]? fileBytes)
    {
        fileBytes = null;
        if (model == null || string.IsNullOrEmpty(name)) return false;

        return ReadAll(model).TryGetValue(name, out fileBytes);
    }

    /// <summary>
    /// Extracts an embedded file by name, throwing <see cref="FileNotFoundException"/> if not found.
    /// </summary>
    public static byte[] Get(GgufModel model, string name)
    {
        if (TryGet(model, name, out var fileBytes))
            return fileBytes;
        throw new FileNotFoundException($"audiocpp embedded file '{name}' not found in GGUF metadata.");
    }

    private static Dictionary<string, byte[]> ParseEmbeddedFiles(GgufModel model)
    {
        bool hasNames = model.Metadata.TryGetValue(NamesKey, out var namesObj);
        bool hasOffsets = model.Metadata.TryGetValue(OffsetsKey, out var offsetsObj);
        bool hasData = model.Metadata.TryGetValue(DataKey, out var dataObj);

        // If the model contains none of the embedded files keys, treat as empty (not corrupt).
        if (!hasNames && !hasOffsets && !hasData)
        {
            return new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);
        }

        if (!hasNames || namesObj is not object[] names)
            throw new InvalidDataException("GGUF missing 'audiocpp.embedded_files.names' metadata array.");

        if (!hasOffsets || offsetsObj is not object[] offsets)
            throw new InvalidDataException("GGUF missing 'audiocpp.embedded_files.offsets' metadata array.");

        if (offsets.Length != names.Length && offsets.Length != names.Length + 1)
            throw new InvalidDataException($"audiocpp.embedded_files.offsets length ({offsets.Length}) must be equal to names length ({names.Length}) or names length + 1.");

        if (!hasData || dataObj == null)
            throw new InvalidDataException("GGUF missing 'audiocpp.embedded_files.data' metadata.");

        byte[] dataBytes = dataObj switch
        {
            byte[] bArr => bArr,
            object[] oArr => oArr.Select(o => (byte)Convert.ToInt64(o)).ToArray(),
            _ => throw new InvalidDataException("GGUF 'audiocpp.embedded_files.data' format unrecognized.")
        };

        // Validate offsets monotonicity and bounds
        long prevOffset = 0;
        for (int i = 0; i < offsets.Length; i++)
        {
            long off = Convert.ToInt64(offsets[i]);
            if (off < 0 || off > dataBytes.Length)
                throw new InvalidDataException($"audiocpp.embedded_files offset at index {i} ({off}) is out of bounds [0, {dataBytes.Length}].");
            if (off < prevOffset)
                throw new InvalidDataException($"audiocpp.embedded_files offsets are not monotonically non-decreasing: index {i} has offset {off} < {prevOffset}.");
            prevOffset = off;
        }

        var dict = new Dictionary<string, byte[]>(names.Length, StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < names.Length; i++)
        {
            string entryName = names[i]?.ToString() ?? throw new InvalidDataException($"audiocpp.embedded_files name at index {i} is null.");
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : dataBytes.Length;
            if (start > end || end > dataBytes.Length)
                throw new InvalidDataException($"audiocpp.embedded_files range for '{entryName}' [{start}..{end}] is invalid for buffer of size {dataBytes.Length}.");
            dict[entryName] = dataBytes[(int)start..(int)end];
        }

        return dict;
    }
}
