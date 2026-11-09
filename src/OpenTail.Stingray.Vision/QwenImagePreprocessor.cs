namespace OpenTail.Stingray.Vision;

/// <summary>
/// Qwen-VL image preprocessing for the qwen4exp projector: a size aligned to 32 pixels (patch 16 times spatial merge 2) inside a
/// pixel budget, then planar CHW in [-1, 1] (mean 0.5, std 0.5). The budget is a server policy, not a checkpoint value.
/// </summary>
public static class QwenImagePreprocessor
{
    /// <summary>Pixels per soft token: the patch size times the spatial merge.</summary>
    public const int Align = 32;

    public const int MinTokens = 4;

    /// <summary>Upper bound on soft tokens per image (256 tokens is a 512 x 512 image).</summary>
    public const int MaxTokens = 256;

    public static PreprocessedImage Preprocess(byte[] rgbHwc, int srcWidth, int srcHeight)
    {
        long perToken = (long)Align * Align;
        var (w, h) = ImagePreprocessor.CalcSizePreservedRatio(srcWidth, srcHeight, Align, MinTokens * perToken, MaxTokens * perToken);
        var u8 = ImagePreprocessor.ResizeToU8Chw(rgbHwc, srcWidth, srcHeight, w, h);
        var chw = new float[u8.Length];
        for (int i = 0; i < chw.Length; i++) chw[i] = 2f * u8[i] / 255f - 1f;
        return new PreprocessedImage(chw, w, h);
    }
}
