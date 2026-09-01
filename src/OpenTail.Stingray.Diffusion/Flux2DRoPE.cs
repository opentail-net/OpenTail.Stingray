
namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// 2D Rotary Position Embeddings for FLUX image patches.
///
/// Real FLUX RoPE (confirmed against black-forest-labs/flux's actual `flux/math.py`'s
/// `rope`/`apply_rope` and `flux/modules/layers.py`'s `EmbedND`, not guessed) is genuinely
/// different from this project's LLM RoPE in two ways that matter:
///
/// <para><b>1. Three position axes, not two, and NOT an even split.</b> `EmbedND` is built with
/// `axes_dim=[16, 56, 56]` (summing to `head_dim=128`): a leading "axis 0" (always position 0 for
/// plain text-to-image — no video/time dimension), then row, then col. Each axis's frequencies are
/// computed independently (its own `theta**(-i/axis_dim)` scale using ITS OWN axis_dim, 56, not
/// `head_dim`) and concatenated along the head-dim axis in that fixed order: elements
/// [0,16) ← axis 0 (always angle 0 → identity, never actually rotates anything, so this
/// implementation skips modeling it explicitly and leaves those elements untouched), [16,72) ←
/// row, [72,128) ← col. A previous version of this file split head_dim exactly in half between
/// row/col (64/64) with no identity portion at all -- wrong on both the proportions and the
/// existence of the identity range.</para>
///
/// <para><b>2. Interleaved-pair rotation, not "rotate-half."</b> `apply_rope` reshapes each head's
/// vector into ADJACENT pairs `(x[2i], x[2i+1])`, each independently rotated by its own 2x2
/// `[[cos,-sin],[sin,cos]]` matrix -- the GPT-NeoX/interleaved convention. A previous version of
/// this file implemented "rotate-half" instead (splitting the FULL head_dim into two halves and
/// rotating `x[i]` against `x[i+head_dim/2]`), which is simply a different, incompatible
/// convention. Compounding that mismatch, that version's rotation loop only ever read frequency
/// slots `[0, head_dim/2)` regardless of axis -- meaning the col-axis frequencies (stored at
/// `[head_dim/2, head_dim)` under its own now-removed row/col split) were computed but NEVER READ,
/// so every patch's column position was silently ignored entirely. That combination (wrong pairing
/// convention AND column position never influencing the rotation at all) produces exactly the
/// periodic small-tile-repeated-across-the-whole-frame artifact this bug was found chasing
/// (2026-09-01): with no way to distinguish horizontal position, the model has no choice but to
/// repeat whatever local pattern it can produce from row-position alone.</para>
/// </summary>
internal static class Flux2DRoPE
{
    private const int TimeAxisDim = 16; // always position 0 for image generation -> pure identity
    private const int SpatialAxisDim = 56; // row axis, and separately col axis
    private const int SpatialAxisPairs = SpatialAxisDim / 2; // 28 frequency pairs per spatial axis

    /// <summary>
    /// Build per-patch (cos, sin) tables, one value per adjacent-pair index (headDim/2 total),
    /// matching real FLUX's axis layout: pairs [0, 8) are the identity time-axis (cos=1, sin=0),
    /// pairs [8, 36) are the row axis (its own theta scale over SpatialAxisDim=56), pairs
    /// [36, 64) are the col axis. <paramref name="headDim"/> must be 128 (FLUX's real head_dim);
    /// asserted, not silently handled for other sizes, since the 16/56/56 axis split is a real
    /// FLUX architecture constant, not derived from head_dim.
    /// </summary>
    public static (float[] cos, float[] sin) BuildFreqs(
        ReadOnlySpan<int> positions, int nPatches, int headDim, float theta = 10000f)
    {
        if (headDim != TimeAxisDim + SpatialAxisDim * 2)
            throw new ArgumentException(
                $"Flux2DRoPE assumes real FLUX's fixed 16/56/56 axis split (head_dim=128); got head_dim={headDim}.");

        int nPairs = headDim / 2; // 64
        int timeAxisPairs = TimeAxisDim / 2; // 8

        var cos = new float[nPatches * nPairs];
        var sin = new float[nPatches * nPairs];

        // Real FLUX: omega[i] = theta ** (-(2i)/axisDim), i.e. scale = arange(0, axisDim, 2) / axisDim.
        var omega = new float[SpatialAxisPairs];
        for (int i = 0; i < SpatialAxisPairs; i++)
            omega[i] = 1f / MathF.Pow(theta, 2f * i / SpatialAxisDim);

        for (int p = 0; p < nPatches; p++)
        {
            int row = positions[p * 2];
            int col = positions[p * 2 + 1];
            int outBase = p * nPairs;

            // Time axis: always position 0 -> angle 0 -> cos=1, sin=0 (identity). Left as the
            // array's zero-initialized default for sin; cos must be explicitly set to 1.
            for (int i = 0; i < timeAxisPairs; i++)
                cos[outBase + i] = 1f;

            int rowBase = outBase + timeAxisPairs;
            for (int i = 0; i < SpatialAxisPairs; i++)
            {
                float angle = row * omega[i];
                cos[rowBase + i] = MathF.Cos(angle);
                sin[rowBase + i] = MathF.Sin(angle);
            }

            int colBase = rowBase + SpatialAxisPairs;
            for (int i = 0; i < SpatialAxisPairs; i++)
            {
                float angle = col * omega[i];
                cos[colBase + i] = MathF.Cos(angle);
                sin[colBase + i] = MathF.Sin(angle);
            }
        }

        return (cos, sin);
    }

    /// <summary>
    /// Build patch position ids for an image of (heightPatches × widthPatches).
    /// Returns flat array of row,col pairs: [nPatches × 2].
    /// </summary>
    public static int[] ImagePatchIds(int heightPatches, int widthPatches)
    {
        int n = heightPatches * widthPatches;
        var ids = new int[n * 2];
        for (int r = 0; r < heightPatches; r++)
            for (int c = 0; c < widthPatches; c++)
            {
                int p = r * widthPatches + c;
                ids[p * 2]     = r;
                ids[p * 2 + 1] = c;
            }
        return ids;
    }

    /// <summary>
    /// Apply RoPE to Q or K tensor in-place.
    /// x layout: [nSeq, nHeads, headDim], cos/sin: [nSeq, headDim/2] (one value per adjacent pair).
    /// Only applied to the first <paramref name="nImgPatches"/> positions (image patches); text
    /// positions are left unchanged, which is exactly equivalent to real FLUX's own behavior of
    /// applying an always-identity (position-0) rotation to text tokens.
    /// </summary>
    public static void ApplyInPlace(float[] x, float[] cos, float[] sin,
                                    int nSeq, int nHeads, int headDim, int nImgPatches)
    {
        int nPairs = headDim / 2;
        for (int s = 0; s < nImgPatches && s < nSeq; s++)
        {
            int freqOff = s * nPairs;
            for (int h = 0; h < nHeads; h++)
            {
                int xOff = (s * nHeads + h) * headDim;
                RotateInterleavedPairsInPlace(x, xOff, cos, sin, freqOff, nPairs);
            }
        }
    }

    /// <summary>
    /// Real FLUX interleaved-pair rotation (GPT-NeoX convention): adjacent elements
    /// (x[2i], x[2i+1]) are rotated together by a 2x2 [[cos,-sin],[sin,cos]] matrix, one pair per
    /// frequency -- NOT the "rotate-half" convention (x[i] paired with x[i+headDim/2]) this file
    /// used before being fixed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RotateInterleavedPairsInPlace(float[] x, int xOff,
                                                        float[] cos, float[] sin, int freqOff,
                                                        int nPairs)
    {
        for (int i = 0; i < nPairs; i++)
        {
            int j = xOff + i * 2;
            float x0 = x[j];
            float x1 = x[j + 1];
            float c = cos[freqOff + i];
            float s = sin[freqOff + i];
            x[j]     = x0 * c - x1 * s;
            x[j + 1] = x0 * s + x1 * c;
        }
    }
}
