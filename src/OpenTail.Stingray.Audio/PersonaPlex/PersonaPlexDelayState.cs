namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>
/// Native C# port of PersonaPlex's real multi-stream DELAY state machine, from `session.cpp`'s
/// `PersonaPlexDelayState` (not guessed). Real, confirmed 17-stream layout: stream 0 = text,
/// streams 1-8 = "moshi" (the model's OWN generated audio -- the real decode-worthy stream),
/// streams 9-16 = "user" (duplex conditioning input; also self-predicted by the Depformer when no
/// live user audio is provided, per `finish_with_sampling`'s real `kPersonaPlexDepformerAudioStreams
/// =16`-wide write). Each stream has its own real per-stream DELAY offset (`kDelays`) into a
/// small ring buffer (`kDelayCacheSteps=4`) -- text and stream 9 (the FIRST user stream) have
/// delay 0, every other stream has delay 1, so what the model consumes as "the current frame" is
/// genuinely NOT all-16-codebooks-of-the-same-instant; earlier streams see one step further
/// ahead than delayed ones. `Prepare`/`FinishWithSampling` reproduce the real ring-buffer
/// bootstrap/advance/read-back logic exactly (`flat_index`, `write_stream`, `initial_token`,
/// the real `offset_==0` double-bootstrap special case, and the real `max_delay=1` output-readback
/// window in `finish_with_sampling`).
/// </summary>
public sealed class PersonaPlexDelayState
{
    public const int MimiFrameCodebooks = 8;
    public const int DelayCacheSteps = 4;
    public const int TextInitialToken = 32000;
    public const int AudioInitialToken = 2048;
    public const int ZeroTextToken = 3;
    public const int UngeneratedToken = -2;
    public const int NumStreams = 17;
    public const int DepformerAudioStreams = 16;

    public static readonly int[] Delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1];

    private readonly int[] _cache = InitCache();
    private readonly byte[] _provided = new byte[NumStreams * DelayCacheSteps];
    private long _offset;

    public long Offset => _offset;

    private static int[] InitCache()
    {
        var cache = new int[NumStreams * DelayCacheSteps];
        Array.Fill(cache, UngeneratedToken);
        return cache;
    }

    private static int FlatIndex(int stream, long step)
    {
        long m = step % DelayCacheSteps;
        if (m < 0) m += DelayCacheSteps;
        return (int)(stream * (long)DelayCacheSteps + m);
    }

    private static int InitialToken(int stream) => stream == 0 ? TextInitialToken : AudioInitialToken;

    private void WriteStream(int stream, long logicalStep, int token)
    {
        int idx = FlatIndex(stream, logicalStep);
        _cache[idx] = token;
        _provided[idx] = 1;
    }

    public readonly struct DelayedStep
    {
        public required int ModelInputPosition { get; init; }
        public required int TargetPosition { get; init; }
        public required int[] Tokens { get; init; } // [NumStreams] -- real model input frame
        public required int[] Target { get; init; } // [NumStreams]
        public required byte[] Provided { get; init; } // [NumStreams]
    }

    /// <summary>Real `PersonaPlexDelayState::prepare`. `userTokens`/`moshiTokens` are each real
    /// `[MimiFrameCodebooks]` (8) externally-provided codes, or `null` for "not provided this
    /// step" (the model self-predicts). Returns `null` exactly once, on the real `offset_==0`
    /// bootstrap step (no model step should run that round).</summary>
    public DelayedStep? Prepare(int[]? userTokens, int[]? moshiTokens, int? textToken)
    {
        if (userTokens != null)
            for (int q = 0; q < MimiFrameCodebooks; q++)
                WriteStream(1 + MimiFrameCodebooks + q, _offset + Delays[1 + MimiFrameCodebooks + q], userTokens[q]);
        if (moshiTokens != null)
            for (int q = 0; q < MimiFrameCodebooks; q++)
                WriteStream(1 + q, _offset + Delays[1 + q], moshiTokens[q]);
        if (textToken.HasValue)
            WriteStream(0, _offset + Delays[0], textToken.Value);

        for (int stream = 0; stream < NumStreams; stream++)
            if (_offset <= Delays[stream])
                WriteStream(stream, _offset, InitialToken(stream));

        if (_offset == 0)
        {
            for (int stream = 0; stream < NumStreams; stream++)
                WriteStream(stream, 0, InitialToken(stream));
            _offset++;
            return null;
        }

        int modelInputPosition = (int)((_offset - 1) % DelayCacheSteps);
        int targetPosition = (int)(_offset % DelayCacheSteps);
        var tokens = new int[NumStreams];
        var target = new int[NumStreams];
        var provided = new byte[NumStreams];
        for (int stream = 0; stream < NumStreams; stream++)
        {
            tokens[stream] = _cache[FlatIndex(stream, modelInputPosition)];
            target[stream] = _cache[FlatIndex(stream, targetPosition)];
            provided[stream] = _provided[FlatIndex(stream, targetPosition)];
        }
        return new DelayedStep
        {
            ModelInputPosition = modelInputPosition,
            TargetPosition = targetPosition,
            Tokens = tokens,
            Target = target,
            Provided = provided,
        };
    }

    /// <summary>Real `PersonaPlexDelayState::finish_with_sampling`. `sampledAudioTokens` is the
    /// real Depformer's full 16-wide output (moshi 1-8 AND self-predicted user 9-16, real
    /// `kPersonaPlexDepformerAudioStreams=16`). Returns the real decode-ready 8 "moshi" codes
    /// once `offset_ > max_delay(=1)`, else `null` (still warming up).</summary>
    public int[]? FinishWithSampling(int sampledTextToken, int[] sampledAudioTokens)
    {
        if (sampledAudioTokens.Length != DepformerAudioStreams)
            throw new ArgumentException($"Expected {DepformerAudioStreams} audio tokens.", nameof(sampledAudioTokens));

        long targetPosition = _offset % DelayCacheSteps;
        if (_provided[FlatIndex(0, targetPosition)] == 0)
            _cache[FlatIndex(0, targetPosition)] = sampledTextToken;
        for (int q = 0; q < DepformerAudioStreams; q++)
        {
            int stream = 1 + q;
            if (_provided[FlatIndex(stream, targetPosition)] == 0)
                _cache[FlatIndex(stream, targetPosition)] = sampledAudioTokens[q];
        }

        long modelInputPosition = _offset - 1;
        for (int stream = 0; stream < NumStreams; stream++)
            _provided[FlatIndex(stream, modelInputPosition)] = 0;

        int[]? output = null;
        const int maxDelay = 1;
        if (_offset > maxDelay)
        {
            output = new int[MimiFrameCodebooks];
            for (int q = 0; q < MimiFrameCodebooks; q++)
            {
                int stream = 1 + q;
                long sourcePosition = _offset - maxDelay + Delays[stream];
                output[q] = _cache[FlatIndex(stream, sourcePosition)];
            }
        }
        _offset++;
        return output;
    }
}
