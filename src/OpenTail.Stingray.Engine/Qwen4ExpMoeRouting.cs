namespace OpenTail.Stingray.Engine;

/// <summary>
/// qwen4exp routed-expert selection, mirroring llama.cpp <c>build_moe_ffn</c> as called from
/// <c>qwen4exp.cpp</c>: softmax over all router logits, top-k by probability, weights renormalised
/// to sum 1 (norm_w = true), then multiplied by <c>expert_weights_scale</c> when it is non-zero.
/// </summary>
public static class Qwen4ExpMoeRouting
{
    public static void Route(ReadOnlySpan<float> logits, int topK, float weightsScale, Span<int> indices, Span<float> weights)
    {
        int n = logits.Length;
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) if (logits[i] > max) max = logits[i];

        Span<float> probs = n <= 512 ? stackalloc float[n] : new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            probs[i] = MathF.Exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < n; i++) probs[i] /= sum;

        // top-k by descending probability; ties resolve to the lower expert index (stable)
        Span<bool> taken = n <= 512 ? stackalloc bool[n] : new bool[n];
        taken.Clear();
        float selected = 0f;
        for (int k = 0; k < topK; k++)
        {
            int best = -1;
            for (int i = 0; i < n; i++)
            {
                if (!taken[i] && (best < 0 || probs[i] > probs[best])) best = i;
            }
            taken[best] = true;
            indices[k] = best;
            weights[k] = probs[best];
            selected += probs[best];
        }

        float norm = selected > 0f ? 1f / selected : 1f;
        float scale = weightsScale != 0f ? weightsScale : 1f;
        for (int k = 0; k < topK; k++) weights[k] *= norm * scale;
    }
}
