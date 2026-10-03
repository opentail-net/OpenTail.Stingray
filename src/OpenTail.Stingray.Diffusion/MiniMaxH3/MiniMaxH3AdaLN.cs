namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// AdaLN conditioning for MiniMax-H3.
/// H3 uses a learned timestep curve table `[8, 1025]` interpolated at continuous timestep t in [0, 1000],
/// followed by a per-block linear projection from 8 -> 96,768.
/// The 96,768 output is partitioned across 3 modalities (Video, Audio, Context),
/// each receiving 6 modulation vectors of dimension HiddenDim (5376):
/// (gamma1, beta1, alpha1, gamma2, beta2, alpha2).
/// </summary>
public static class MiniMaxH3AdaLN
{
    /// <summary>
    /// Linearly interpolate the 8-channel curve vector at timestep t in [0, 1000] from the table [8, 1025].
    /// </summary>
    public static void InterpolateTimestepCurve(
        ReadOnlySpan<float> table8x1025,
        float timestep0To1000,
        Span<float> curveVec8)
    {
        if (table8x1025.Length < MiniMaxH3Config.AdaLnTableRows * MiniMaxH3Config.AdaLnTableCols)
            throw new ArgumentException("AdaLN table must have at least 8 * 1025 elements.", nameof(table8x1025));
        if (curveVec8.Length < MiniMaxH3Config.AdaLnTableRows)
            throw new ArgumentException("curveVec8 must have length >= 8.", nameof(curveVec8));

        float clampedT = Math.Clamp(timestep0To1000, 0.0f, 1000.0f);
        float pos = clampedT * (1024.0f / 1000.0f);
        int idx = (int)pos;
        if (idx >= 1024) idx = 1023;
        float frac = pos - idx;

        for (int c = 0; c < MiniMaxH3Config.AdaLnTableRows; c++)
        {
            int rowOffset = c * MiniMaxH3Config.AdaLnTableCols;
            float v0 = table8x1025[rowOffset + idx];
            float v1 = table8x1025[rowOffset + idx + 1];
            curveVec8[c] = v0 + frac * (v1 - v0);
        }
    }

    /// <summary>
    /// Project the 8-dim curve vector to modulation floats for a block (96,768 in full model: 3 * 6 * 5376).
    /// weight is [projDim, 8] row-major.
    /// </summary>
    public static void ProjectModulation(
        ReadOnlySpan<float> curveVec8,
        ReadOnlySpan<float> weightRowsx8,
        ReadOnlySpan<float> biasRows,
        Span<float> modulationOutput)
    {
        if (curveVec8.Length < MiniMaxH3Config.AdaLnTableRows)
            throw new ArgumentException("curveVec8 must have at least 8 elements.");
        int projDim = modulationOutput.Length;
        if (weightRowsx8.Length < projDim * MiniMaxH3Config.AdaLnTableRows)
            throw new ArgumentException($"weight must have {projDim} * 8 elements.");

        bool hasBias = !biasRows.IsEmpty;
        for (int i = 0; i < projDim; i++)
        {
            int rowStart = i * MiniMaxH3Config.AdaLnTableRows;
            float sum = hasBias ? biasRows[i] : 0.0f;
            for (int k = 0; k < MiniMaxH3Config.AdaLnTableRows; k++)
            {
                sum += weightRowsx8[rowStart + k] * curveVec8[k];
            }
            modulationOutput[i] = sum;
        }
    }

    /// <summary>
    /// Slices the modulation vector into the 6 AdaLN-Zero vectors for a given modality:
    /// (gamma1, beta1, alpha1, gamma2, beta2, alpha2), each of dimension hiddenDim (5376 in full model).
    /// </summary>
    public static ModalityModulation GetModalityModulation(
        ReadOnlySpan<float> modulation,
        MiniMaxH3Modality modality,
        int hiddenDim = MiniMaxH3Config.HiddenDim)
    {
        int modDimPerModality = 6 * hiddenDim;
        int modalityOffset = (int)modality * modDimPerModality;
        var slice = modulation.Slice(modalityOffset, modDimPerModality);

        return new ModalityModulation(
            Gamma1: slice.Slice(0 * hiddenDim, hiddenDim),
            Beta1: slice.Slice(1 * hiddenDim, hiddenDim),
            Alpha1: slice.Slice(2 * hiddenDim, hiddenDim),
            Gamma2: slice.Slice(3 * hiddenDim, hiddenDim),
            Beta2: slice.Slice(4 * hiddenDim, hiddenDim),
            Alpha2: slice.Slice(5 * hiddenDim, hiddenDim)
        );
    }

    /// <summary>
    /// Apply AdaLN modulation to normalized hidden state: (1 + gamma) * x + beta.
    /// </summary>
    public static void ModulateNorm(Span<float> x, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta)
    {
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (1.0f + gamma[i]) * x[i] + beta[i];
        }
    }

    /// <summary>
    /// Apply AdaLN gate scaling: x * alpha.
    /// </summary>
    public static void ApplyGate(Span<float> x, ReadOnlySpan<float> alpha)
    {
        for (int i = 0; i < x.Length; i++)
        {
            x[i] *= alpha[i];
        }
    }
}

/// <summary>
/// The 6 AdaLN-Zero modulation vectors for a specific modality in a DiT block.
/// </summary>
public readonly ref struct ModalityModulation
{
    public readonly ReadOnlySpan<float> Gamma1;
    public readonly ReadOnlySpan<float> Beta1;
    public readonly ReadOnlySpan<float> Alpha1;
    public readonly ReadOnlySpan<float> Gamma2;
    public readonly ReadOnlySpan<float> Beta2;
    public readonly ReadOnlySpan<float> Alpha2;

    public ModalityModulation(
        ReadOnlySpan<float> Gamma1,
        ReadOnlySpan<float> Beta1,
        ReadOnlySpan<float> Alpha1,
        ReadOnlySpan<float> Gamma2,
        ReadOnlySpan<float> Beta2,
        ReadOnlySpan<float> Alpha2)
    {
        this.Gamma1 = Gamma1;
        this.Beta1 = Beta1;
        this.Alpha1 = Alpha1;
        this.Gamma2 = Gamma2;
        this.Beta2 = Beta2;
        this.Alpha2 = Alpha2;
    }
}
