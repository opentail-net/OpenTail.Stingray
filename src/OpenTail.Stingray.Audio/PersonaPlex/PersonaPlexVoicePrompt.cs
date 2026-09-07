namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>
/// Real per-voice-id bootstrap state, ported from `request.cpp`'s `load_personaplex_voice_prompt`/
/// `PersonaPlexVoicePromptState` (not guessed). Loaded from one of the 18 real
/// `voices_safetensors/{VOICE_ID}.safetensors` assets embedded directly in this checkpoint's
/// packed GGUF (confirmed via a real embedded-files dump -- NOT external files, contrary to an
/// earlier this-session assumption; see docs/audio-review-progress.md's correction entry).
/// Real, confirmed tensor layout (parsed directly from one real extracted asset's safetensors
/// header, not guessed): `cache` (`I64`, shape `[1, NumStreams=17, DelayCacheSteps=4]`, 68
/// elements) and `embeddings` (`BF16`, shape `[frames, 1, 1, hiddenSize=4096]`).
/// </summary>
public sealed class PersonaPlexVoicePrompt
{
    public required int Frames { get; init; }
    public required int HiddenDim { get; init; }
    public required float[] Embeddings { get; init; } // [frames][hiddenDim] row-major
    public required long[] Cache { get; init; } // [NumStreams * DelayCacheSteps], real int64 delay-ring-buffer snapshot

    public static PersonaPlexVoicePrompt Load(string safetensorsPath, int hiddenDim)
    {
        using var loader = SafetensorsLoader.Open(safetensorsPath);
        var embeddingsShape = loader.GetShape("embeddings");
        if (embeddingsShape.Length != 4 || embeddingsShape[1] != 1 || embeddingsShape[2] != 1 || embeddingsShape[3] != hiddenDim)
            throw new InvalidDataException($"PersonaPlex voice prompt 'embeddings' shape mismatch: [{string.Join(",", embeddingsShape)}].");
        int frames = embeddingsShape[0];
        var embeddings = loader.ReadF32("embeddings");

        var cacheShape = loader.GetShape("cache");
        int expectedCache = PersonaPlexDelayState.NumStreams * PersonaPlexDelayState.DelayCacheSteps;
        if (cacheShape.Length != 3 || cacheShape[0] != 1 || cacheShape[1] * cacheShape[2] != expectedCache)
            throw new InvalidDataException($"PersonaPlex voice prompt 'cache' shape mismatch: [{string.Join(",", cacheShape)}].");
        var cacheRaw = loader.ReadRaw("cache", out string dtype);
        if (dtype != "I64")
            throw new InvalidDataException($"PersonaPlex voice prompt 'cache' expected I64, got {dtype}.");
        var cache = new long[expectedCache];
        Buffer.BlockCopy(cacheRaw, 0, cache, 0, cacheRaw.Length);

        return new PersonaPlexVoicePrompt { Frames = frames, HiddenDim = hiddenDim, Embeddings = embeddings, Cache = cache };
    }
}
