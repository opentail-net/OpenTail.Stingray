using System.Numerics.Tensors;

namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Stable Audio 3 per-layer local conditioning (<c>to_local_embed</c>) for plain text-to-audio generation, shared by
/// the Small and Medium DiTs. With no inpainting or init audio the local input is all zeros, so
/// <c>to_local_embed(0) = W2 * silu(b1) + b2</c> is a constant added to every LATENT token of every layer, after
/// cross-attention (memory tokens get zeros) -- audio.cpp <c>rf_dit.cpp build_local_embedding</c>.
/// </summary>
/// <remarks>
/// Omitted from both ports until 2026-09-27. Found by diffing against the vendored reference with identical noise:
/// step-0 velocity cosine 0.9996 but 4% too small in norm, final latent cosine 0.986 after 20 steps; with this
/// constant, step 0 matches to 0.999998 and the final latent to 0.9993. This was the "darker than the reference"
/// symptom (docs/102 #11).
/// </remarks>
internal static class StableAudioLocalConditioning
{
    /// <summary><c>W2 * silu(b1) + b2</c> for one layer, or null when the checkpoint has no local-conditioning
    /// weights. <paramref name="read"/> returns a tensor as float32 (Linear weights are [out, in] row-major).</summary>
    public static float[]? ZeroInputConstant(Func<string, bool> contains, Func<string, float[]> read, int layer, int dim)
    {
        string p = $"model.model.transformer.layers.{layer}.to_local_embed";
        if (!contains($"{p}.0.bias") || !contains($"{p}.2.weight")) return null;
        var h = read($"{p}.0.bias").ToArray();
        for (int i = 0; i < h.Length; i++) h[i] = h[i] / (1f + MathF.Exp(-h[i]));
        var w2 = read($"{p}.2.weight");
        var outp = read($"{p}.2.bias").ToArray();
        for (int o = 0; o < dim; o++)
            outp[o] += TensorPrimitives.Dot(new ReadOnlySpan<float>(w2, o * h.Length, h.Length), h);
        return outp;
    }

    /// <summary>Host buffer for a GPU add: <paramref name="totalRows"/> x dim, zeros for the memory rows and the
    /// constant for every latent row.</summary>
    public static float[] BuildRowAdd(int totalRows, int memoryTokens, float[] rowConstant)
    {
        int dim = rowConstant.Length;
        var host = new float[totalRows * dim];
        for (int r = memoryTokens; r < totalRows; r++) Array.Copy(rowConstant, 0, host, r * dim, dim);
        return host;
    }
}
