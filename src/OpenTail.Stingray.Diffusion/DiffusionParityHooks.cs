namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Environment-driven parity hooks for diffusion pipelines, used to diff against a reference implementation
/// (e.g. examples/stable-diffusion.cpp's SD_DUMP_NOISE_PATH / SD_DUMP_LATENT_PATH, audio.cpp's SA3_DUMP_DIR): load the
/// reference's initial noise instead of our own RNG, and dump the final latent. Raw little-endian float32, no header.
/// No-ops unless the named variable is set.
/// </summary>
internal static class DiffusionParityHooks
{
    /// <summary>Overwrites <paramref name="latent"/> with the float32 file named by <paramref name="envVar"/>, if set.
    /// <paramref name="fromFileLayout"/> (optional) converts from the file's layout into ours.</summary>
    public static bool TryLoadNoise(string envVar, float[] latent, Func<float[], float[]>? fromFileLayout = null)
    {
        if (Environment.GetEnvironmentVariable(envVar) is not { Length: > 0 } path) return false;
        var raw = File.ReadAllBytes(path);
        if (raw.Length != latent.Length * 4)
            throw new InvalidOperationException($"{envVar} has {raw.Length / 4} floats, latent needs {latent.Length}.");
        var data = new float[latent.Length];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        (fromFileLayout?.Invoke(data) ?? data).AsSpan().CopyTo(latent);
        return true;
    }

    /// <summary>Writes <paramref name="data"/> to the file named by <paramref name="envVar"/>, if set.</summary>
    public static void DumpToFile(string envVar, float[] data)
    {
        if (Environment.GetEnvironmentVariable(envVar) is not { Length: > 0 } path) return;
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }
}
