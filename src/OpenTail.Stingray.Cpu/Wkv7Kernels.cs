namespace OpenTail.Stingray.Cpu;

/// <summary>
/// RWKV-7 "WKV7" recurrence (GGML_OP_RWKV_WKV7), one token at a time. Port of ggml-cpu ops.cpp
/// <c>ggml_compute_forward_rwkv_wkv7_f32</c>'s scalar route. Per head, with state <c>S</c> of
/// shape [headSize (value index i)][headSize (key index j)]:
/// <code>
///   sa_i    = Σ_j a_j · S[i][j]
///   S[i][j] = S[i][j] · w_j + v_i · k_j + sa_i · b_j
///   y_i     = Σ_j S[i][j] · r_j
/// </code>
/// The update reads the PREVIOUS state for <c>sa_i</c> (computed before any write to row i), and
/// row i only ever reads row i, so the in-place update is exact.
/// </summary>
public static unsafe class Wkv7Kernels
{
    /// <param name="state">heads × headSize × headSize, updated in place.</param>
    /// <param name="y">heads × headSize output.</param>
    public static void Step(float* state, float* r, float* w, float* k, float* v, float* a, float* b,
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
}
