using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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
    /// One head of <see cref="Wkv7Step"/>, vectorized over the key index j (AVX2 when available,
    /// else the scalar loop). The two reductions (<c>sa_i</c> and <c>y_i</c>) are summed in 8-wide
    /// lanes, so results differ from the scalar step in the last bits; the update itself is
    /// element-wise and unchanged. Every pointer is the head's own slice (state: hs × hs).
    /// </summary>
    public static void Wkv7StepHead(float* s, float* r, float* w, float* k, float* v, float* a, float* b,
        float* y, int hs)
    {
        if (!Avx2.IsSupported || hs % 8 != 0)
        {
            Wkv7Step(s, r, w, k, v, a, b, y, 1, hs);
            return;
        }
        for (int i = 0; i < hs; i++)
        {
            float* row = s + i * hs;
            var saV = Vector256<float>.Zero;
            for (int j = 0; j < hs; j += 8)
                saV += Vector256.Load(a + j) * Vector256.Load(row + j);
            var sa = Vector256.Create(Vector256.Sum(saV));
            var vi = Vector256.Create(v[i]);
            var acc = Vector256<float>.Zero;
            for (int j = 0; j < hs; j += 8)
            {
                var nv = Vector256.Load(row + j) * Vector256.Load(w + j)
                         + vi * Vector256.Load(k + j)
                         + sa * Vector256.Load(b + j);
                nv.Store(row + j);
                acc += nv * Vector256.Load(r + j);
            }
            y[i] = Vector256.Sum(acc);
        }
    }

    /// <summary>
    /// One head of <see cref="Wkv6Step"/>, vectorized over the value index j. Same arithmetic per
    /// element as the scalar step (the sum over i is accumulated in the same order), so results are
    /// identical. Every pointer is the head's own slice.
    /// </summary>
    public static void Wkv6StepHead(float* s, float* r, float* w, float* k, float* v, float* u, float* y, int hs)
    {
        if (!Avx2.IsSupported || hs % 8 != 0)
        {
            Wkv6Step(s, r, w, k, v, u, y, 1, hs);
            return;
        }
        for (int j = 0; j < hs; j += 8) Vector256<float>.Zero.Store(y + j);
        for (int i = 0; i < hs; i++)
        {
            float* row = s + i * hs;
            var ki = Vector256.Create(k[i]); var ri = Vector256.Create(r[i]);
            var ui = Vector256.Create(u[i]); var wi = Vector256.Create(w[i]);
            for (int j = 0; j < hs; j += 8)
            {
                var kv = Vector256.Load(v + j) * ki;
                var prev = Vector256.Load(row + j);
                (Vector256.Load(y + j) + (kv * ui + prev) * ri).Store(y + j);
                (prev * wi + kv).Store(row + j);
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
