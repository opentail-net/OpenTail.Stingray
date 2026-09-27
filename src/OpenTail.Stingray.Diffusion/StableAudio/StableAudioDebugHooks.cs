namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Parity-debugging hooks shared by the Small and Medium pipelines (used 2026-09-27 against audio.cpp's
/// <c>SA3_DUMP_DIR</c> dumps). <c>STINGRAY_SA3_NOISE</c>: raw f32 initial latent to start from (token-major
/// [seqLen, 256]; the reference writes channel-major, so transpose it first). <c>STINGRAY_SA3_DUMP</c>: directory
/// for raw f32 dumps. Both are no-ops when unset.
/// </summary>
internal static class StableAudioDebugHooks
{
    public static void MaybeLoadNoise(float[] latent)
    {
        if (Environment.GetEnvironmentVariable("STINGRAY_SA3_NOISE") is not { Length: > 0 } path) return;
        var raw = File.ReadAllBytes(path);
        if (raw.Length != latent.Length * 4)
            throw new InvalidOperationException($"STINGRAY_SA3_NOISE has {raw.Length / 4} floats, latent needs {latent.Length}.");
        Buffer.BlockCopy(raw, 0, latent, 0, raw.Length);
    }

    public static void Dump(string name, float[] data)
    {
        if (Environment.GetEnvironmentVariable("STINGRAY_SA3_DUMP") is not { Length: > 0 } dir) return;
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(Path.Combine(dir, name + ".f32"), bytes);
    }
}
