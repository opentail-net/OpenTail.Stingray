namespace OpenTail.Stingray.Cpu;

/// <summary>
/// The RWKV "WKV" recurrences, one token at a time, ported from ggml-cpu ops.cpp's scalar routes.
/// State is heads × headSize × headSize per layer.
/// </summary>
public static unsafe class WkvKernels
{
    /// <summary>
    /// RWKV-7 (GGML_OP_RWKV_WKV7, <c>ggml_compute_forward_rwkv_wkv7_f32</c>). Per head, with state
    /// <c>S</c> indexed [value i][key j]:
    /// <code>
    ///   sa_i    = Σ_j a_j · S[i][j]
    ///   S[i][j] = S[i][j] · w_j + v_i · k_j + sa_i · b_j
    ///   y_i     = Σ_j S[i][j] · r_j
    /// </code>
    /// <c>sa_i</c> reads row i before any write to it, and row i only reads row i, so the in-place
    /// update is exact.
    /// </summary>
    public static void Wkv7Step(float* state, float* r, float* w, float* k, float* v, float* a, float* b,
        float* y, int heads, int headSize)
    {
        for (int h = 0; h < heads; h++)
        {
            int off = h * headSize;
            float* s = state + (long)h * headSize * headSize;
            float* rh = r + off, wh = w + off, kh = k + off, ah = a + off, bh = b + off;
            for (int i = 0; i < headSize; i++)
            {
                float* row = s + i * headSize;
                float vi = v[off + i];
                float sa = 0f;
                for (int j = 0; j < headSize; j++) sa += ah[j] * row[j];
                float result = 0f;
                for (int j = 0; j < headSize; j++)
                {
                    float nv = row[j] * wh[j] + vi * kh[j] + sa * bh[j];
                    row[j] = nv;
                    result += nv * rh[j];
                }
                y[off + i] = result;
            }
        }
    }

    /// <summary>
    /// RWKV-6 (GGML_OP_RWKV_WKV6, <c>ggml_compute_forward_rwkv_wkv6_f32</c>). Per head, with state
    /// <c>S</c> indexed [key i][value j] and the per-channel bonus <c>u</c> (time_first):
    /// <code>
    ///   y_j     = Σ_i r_i · (u_i · k_i · v_j + S[i][j])
    ///   S[i][j] = S[i][j] · w_i + k_i · v_j
    /// </code>
    /// </summary>
    public static void Wkv6Step(float* state, float* r, float* w, float* k, float* v, float* u,
        float* y, int heads, int headSize)
    {
        for (int h = 0; h < heads; h++)
        {
            int off = h * headSize;
            float* s = state + (long)h * headSize * headSize;
            float* yh = y + off, vh = v + off;
            new Span<float>(yh, headSize).Clear();
            for (int i = 0; i < headSize; i++)
            {
                float* row = s + i * headSize;
                float ki = k[off + i], ri = r[off + i], ui = u[off + i], wi = w[off + i];
                for (int j = 0; j < headSize; j++)
                {
                    float kv = vh[j] * ki;
                    float prev = row[j];
                    yh[j] += (kv * ui + prev) * ri;
                    row[j] = prev * wi + kv;
                }
            }
        }
    }
}
