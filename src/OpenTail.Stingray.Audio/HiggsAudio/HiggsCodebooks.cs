namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Real MusicGen-style delay-pattern helpers for Higgs Audio TTS's multi-codebook audio stream,
/// ported from `codebooks.cpp`'s `higgs_delayed_frame_count`/`apply_higgs_delay_pattern`/
/// `reverse_higgs_delay_pattern` (not guessed): codebook `k`'s real code for raw frame `f` is
/// placed at delayed frame `k+f` (staggered so all codebooks needed to decode delayed frame `t`
/// were emitted no later than step `t`), with `kHiggsBocId`/`kHiggsEocId` (`1024`/`1025`) filling
/// each codebook's own leading BOC-padding frames before its real codes start.
/// </summary>
public static class HiggsCodebooks
{
    public const int BocId = 1024;
    public const int EocId = 1025;
    public const int StopCode = -1;

    public static int DelayedFrameCount(int rawFrames, int codebooks)
    {
        if (rawFrames <= 0 || codebooks <= 0)
            throw new ArgumentException("Higgs TTS delayed frame count requires positive dimensions.");
        return rawFrames + codebooks - 1;
    }

    /// <summary>`rawCodes` is `[frame][codebook]` row-major (`frame*codebooks+codebook`), `frames`
    /// major. Returns the delayed `[delayedFrame][codebook]` matrix, same flat layout.</summary>
    public static int[] ApplyDelayPattern(int[] rawCodes, int rawFrames, int codebooks)
    {
        RequireCodebookMatrix(rawCodes, rawFrames, codebooks);
        int delayedFrames = DelayedFrameCount(rawFrames, codebooks);
        var delayed = new int[delayedFrames * codebooks];
        Array.Fill(delayed, EocId);

        for (int codebook = 0; codebook < codebooks; codebook++)
        {
            for (int frame = 0; frame < codebook; frame++)
                delayed[FlatIndex(frame, codebook, codebooks)] = BocId;
            for (int frame = 0; frame < rawFrames; frame++)
                delayed[FlatIndex(codebook + frame, codebook, codebooks)] = rawCodes[FlatIndex(frame, codebook, codebooks)];
        }
        return delayed;
    }

    public static int[] ReverseDelayPattern(int[] delayedCodes, int delayedFrames, int codebooks)
    {
        RequireCodebookMatrix(delayedCodes, delayedFrames, codebooks);
        int rawFrames = delayedFrames - (codebooks - 1);
        if (rawFrames <= 0)
            throw new ArgumentException("Higgs TTS delayed codes must include at least one recoverable raw frame.");

        var raw = new int[rawFrames * codebooks];
        for (int codebook = 0; codebook < codebooks; codebook++)
            for (int frame = 0; frame < rawFrames; frame++)
                raw[FlatIndex(frame, codebook, codebooks)] = delayedCodes[FlatIndex(codebook + frame, codebook, codebooks)];
        return raw;
    }

    private static void RequireCodebookMatrix(int[] codes, int frames, int codebooks)
    {
        if (frames <= 0 || codebooks <= 0)
            throw new ArgumentException("Higgs TTS requires positive frames and codebooks.");
        if (codes.Length != frames * codebooks)
            throw new ArgumentException("Higgs TTS code matrix shape mismatch.");
    }

    private static int FlatIndex(int frame, int codebook, int codebooks) => frame * codebooks + codebook;
}
