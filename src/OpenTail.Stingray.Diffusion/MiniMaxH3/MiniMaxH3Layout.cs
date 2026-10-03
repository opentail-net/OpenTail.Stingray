namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// Sequence layout and token indexing for MiniMax-H3.
/// Manages the packed multimodal sequence:
///   [text tokens | visual conditioning tokens | video target tokens | audio target tokens]
/// and provides channel-major, patch-minor packing/unpacking for 3D video latents.
/// </summary>
public sealed class MiniMaxH3Layout
{
    public int NumTextTokens { get; }
    public int NumVisualCondTokens { get; }
    public int NumVideoTokens { get; }
    public int NumAudioTokens { get; }

    public int TextStart => 0;
    public int VisualCondStart => NumTextTokens;
    public int VideoTargetStart => NumTextTokens + NumVisualCondTokens;
    public int AudioTargetStart => NumTextTokens + NumVisualCondTokens + NumVideoTokens;
    public int TotalTokens => NumTextTokens + NumVisualCondTokens + NumVideoTokens + NumAudioTokens;

    public int VideoFrames { get; }
    public int VideoHeight { get; }
    public int VideoWidth { get; }
    public int VideoPatchH => VideoHeight / MiniMaxH3Config.PatchHeight;
    public int VideoPatchW => VideoWidth / MiniMaxH3Config.PatchWidth;

    public int AudioFrames { get; }

    public MiniMaxH3Layout(
        int numTextTokens,
        int numVisualCondTokens,
        int videoFrames,
        int videoHeight,
        int videoWidth,
        int audioFrames)
    {
        if (videoHeight % MiniMaxH3Config.PatchHeight != 0)
            throw new ArgumentException($"Video height {videoHeight} must be divisible by {MiniMaxH3Config.PatchHeight}.");
        if (videoWidth % MiniMaxH3Config.PatchWidth != 0)
            throw new ArgumentException($"Video width {videoWidth} must be divisible by {MiniMaxH3Config.PatchWidth}.");

        NumTextTokens = numTextTokens;
        NumVisualCondTokens = numVisualCondTokens;
        VideoFrames = videoFrames;
        VideoHeight = videoHeight;
        VideoWidth = videoWidth;
        NumVideoTokens = videoFrames * VideoPatchH * VideoPatchW;

        AudioFrames = audioFrames;
        NumAudioTokens = audioFrames;
    }

    /// <summary>
    /// Identify the modality of a token at global index i in [0, TotalTokens).
    /// </summary>
    public MiniMaxH3Modality GetModality(int tokenIndex)
    {
        if (tokenIndex < VideoTargetStart) return MiniMaxH3Modality.Context;
        if (tokenIndex < AudioTargetStart) return MiniMaxH3Modality.Video;
        if (tokenIndex < TotalTokens) return MiniMaxH3Modality.Audio;
        throw new ArgumentOutOfRangeException(nameof(tokenIndex), $"Token index {tokenIndex} exceeds TotalTokens {TotalTokens}.");
    }

    /// <summary>
    /// Pack video latent [T, C=24, H, W] into video patch tokens [NumVideoTokens, 96].
    /// Channel-major, patch-minor ordering:
    /// For patch at (t, hp, wp), within the 96-dim vector:
    ///   idx = c * 4 + ph * 2 + pw.
    /// </summary>
    public void PackVideoLatent(ReadOnlySpan<float> videoLatentTCHW, Span<float> patchTokens)
    {
        int cDim = MiniMaxH3Config.VideoLatentChannels; // 24
        int patchDim = MiniMaxH3Config.VideoPatchDim; // 96
        int expectedLatentLen = VideoFrames * cDim * VideoHeight * VideoWidth;
        if (videoLatentTCHW.Length < expectedLatentLen)
            throw new ArgumentException($"Video latent length {videoLatentTCHW.Length} < expected {expectedLatentLen}.");
        if (patchTokens.Length < NumVideoTokens * patchDim)
            throw new ArgumentException($"Patch tokens buffer length {patchTokens.Length} < expected {NumVideoTokens * patchDim}.");

        int patchH = VideoPatchH;
        int patchW = VideoPatchW;
        int spatialLatentSize = VideoHeight * VideoWidth;
        int frameLatentSize = cDim * spatialLatentSize;

        for (int t = 0; t < VideoFrames; t++)
        {
            int frameOffset = t * frameLatentSize;
            for (int hp = 0; hp < patchH; hp++)
            {
                for (int wp = 0; wp < patchW; wp++)
                {
                    int tokenIndex = (t * patchH + hp) * patchW + wp;
                    int tokenOffset = tokenIndex * patchDim;

                    int dst = 0;
                    for (int c = 0; c < cDim; c++)
                    {
                        int cOffset = frameOffset + c * spatialLatentSize;
                        for (int ph = 0; ph < MiniMaxH3Config.PatchHeight; ph++)
                        {
                            int h = hp * MiniMaxH3Config.PatchHeight + ph;
                            int rowOffset = cOffset + h * VideoWidth;
                            for (int pw = 0; pw < MiniMaxH3Config.PatchWidth; pw++)
                            {
                                int w = wp * MiniMaxH3Config.PatchWidth + pw;
                                patchTokens[tokenOffset + dst++] = videoLatentTCHW[rowOffset + w];
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Unpack video patch tokens [NumVideoTokens, 96] into video latent [T, C=24, H, W].
    /// Inverts the channel-major, patch-minor layout.
    /// </summary>
    public void UnpackVideoLatent(ReadOnlySpan<float> patchTokens, Span<float> videoLatentTCHW)
    {
        int cDim = MiniMaxH3Config.VideoLatentChannels; // 24
        int patchDim = MiniMaxH3Config.VideoPatchDim; // 96
        int patchH = VideoPatchH;
        int patchW = VideoPatchW;
        int spatialLatentSize = VideoHeight * VideoWidth;
        int frameLatentSize = cDim * spatialLatentSize;

        for (int t = 0; t < VideoFrames; t++)
        {
            int frameOffset = t * frameLatentSize;
            for (int hp = 0; hp < patchH; hp++)
            {
                for (int wp = 0; wp < patchW; wp++)
                {
                    int tokenIndex = (t * patchH + hp) * patchW + wp;
                    int tokenOffset = tokenIndex * patchDim;

                    int src = 0;
                    for (int c = 0; c < cDim; c++)
                    {
                        int cOffset = frameOffset + c * spatialLatentSize;
                        for (int ph = 0; ph < MiniMaxH3Config.PatchHeight; ph++)
                        {
                            int h = hp * MiniMaxH3Config.PatchHeight + ph;
                            int rowOffset = cOffset + h * VideoWidth;
                            for (int pw = 0; pw < MiniMaxH3Config.PatchWidth; pw++)
                            {
                                int w = wp * MiniMaxH3Config.PatchWidth + pw;
                                videoLatentTCHW[rowOffset + w] = patchTokens[tokenOffset + src++];
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Pack audio latent [T_a, 32] directly into audio tokens [NumAudioTokens, 32].
    /// </summary>
    public void PackAudioLatent(ReadOnlySpan<float> audioLatentTC, Span<float> audioTokens)
    {
        int audioDim = MiniMaxH3Config.AudioPatchDim;
        int count = NumAudioTokens * audioDim;
        if (audioLatentTC.Length < count || audioTokens.Length < count)
            throw new ArgumentException("Audio latent buffer sizes mismatch.");
        audioLatentTC.Slice(0, count).CopyTo(audioTokens.Slice(0, count));
    }

    /// <summary>
    /// Unpack audio tokens [NumAudioTokens, 32] directly into audio latent [T_a, 32].
    /// </summary>
    public void UnpackAudioLatent(ReadOnlySpan<float> audioTokens, Span<float> audioLatentTC)
    {
        int audioDim = MiniMaxH3Config.AudioPatchDim;
        int count = NumAudioTokens * audioDim;
        if (audioTokens.Length < count || audioLatentTC.Length < count)
            throw new ArgumentException("Audio latent buffer sizes mismatch.");
        audioTokens.Slice(0, count).CopyTo(audioLatentTC.Slice(0, count));
    }

    /// <summary>
    /// Get spatial-temporal 3D coordinate (t, hp, wp) for a video token index.
    /// </summary>
    public (int T, int Hp, int Wp) GetVideoCoordinates(int videoTokenIndex)
    {
        int patchW = VideoPatchW;
        int patchH = VideoPatchH;
        int spatialTokens = patchH * patchW;

        int t = videoTokenIndex / spatialTokens;
        int rem = videoTokenIndex % spatialTokens;
        int hp = rem / patchW;
        int wp = rem % patchW;
        return (t, hp, wp);
    }
}
