#include <immintrin.h>
#include <stdint.h>
#include <string.h>

// Helper to horizontally add one __m256 into a single float
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 sums = _mm_add_ps(sum128, shuf);
    __m128 shuf2 = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
}

// Real hardware F16C-based dot product: converts 8 packed fp16 weight values per iteration
// to float32 via VCVTPH2PS (a single instruction, ~3-4 cycle latency) directly into a register,
// FMA'd against the already-F32 activation vector. Weights never touch a scratch F32 buffer.
__declspec(dllexport)
float f16c_dot(const float* input, const uint16_t* weightF16Bits, int k) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= k - 8; i += 8) {
        __m128i wBits = _mm_loadu_si128((const __m128i*)(weightF16Bits + i));
        __m256 wVec = _mm256_cvtph_ps(wBits);
        __m256 xVec = _mm256_loadu_ps(input + i);
        acc = _mm256_fmadd_ps(wVec, xVec, acc);
    }

    float sum = hsum256_ps(acc);

    for (; i < k; i++) {
        __m128i wb = _mm_cvtsi32_si128((int)weightF16Bits[i]);
        __m128 wf = _mm_cvtph_ps(wb);
        sum += _mm_cvtss_f32(wf) * input[i];
    }

    return sum;
}

__declspec(dllexport)
void f16c_matvec_row(const float* input, const uint16_t* weightF16Bits, int k, float* output) {
    *output = f16c_dot(input, weightF16Bits, k);
}

// Computes outRows dot products for a single input vector of length inDim against a contiguous weight matrix.
// 4-way register unrolling across output rows: loads input once into YMM registers and uses across 4 weight rows.
__declspec(dllexport)
void f16c_matvec_rows(const float* input, const uint16_t* weightsF16, int inDim, int outRows, float* output) {
    int o = 0;
    for (; o <= outRows - 4; o += 4) {
        const uint16_t* w0 = weightsF16 + (size_t)(o + 0) * inDim;
        const uint16_t* w1 = weightsF16 + (size_t)(o + 1) * inDim;
        const uint16_t* w2 = weightsF16 + (size_t)(o + 2) * inDim;
        const uint16_t* w3 = weightsF16 + (size_t)(o + 3) * inDim;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int i = 0;
        for (; i <= inDim - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(input + i);
            __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0 + i)));
            acc0 = _mm256_fmadd_ps(v0, x, acc0);
            __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1 + i)));
            acc1 = _mm256_fmadd_ps(v1, x, acc1);
            __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2 + i)));
            acc2 = _mm256_fmadd_ps(v2, x, acc2);
            __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3 + i)));
            acc3 = _mm256_fmadd_ps(v3, x, acc3);
        }

        float sum0 = hsum256_ps(acc0);
        float sum1 = hsum256_ps(acc1);
        float sum2 = hsum256_ps(acc2);
        float sum3 = hsum256_ps(acc3);

        for (; i < inDim; i++) {
            float xi = input[i];
            __m128 wf0 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)w0[i]));
            sum0 += _mm_cvtss_f32(wf0) * xi;
            __m128 wf1 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)w1[i]));
            sum1 += _mm_cvtss_f32(wf1) * xi;
            __m128 wf2 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)w2[i]));
            sum2 += _mm_cvtss_f32(wf2) * xi;
            __m128 wf3 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)w3[i]));
            sum3 += _mm_cvtss_f32(wf3) * xi;
        }

        output[o + 0] = sum0;
        output[o + 1] = sum1;
        output[o + 2] = sum2;
        output[o + 3] = sum3;
    }

    for (; o < outRows; o++) {
        output[o] = f16c_dot(input, weightsF16 + (size_t)o * inDim, inDim);
    }
}

// Batched 3 tokens x 4 output channels tiled GEMM block in AVX2 F16C.
// Uses 12 YMM accumulators + 3 input registers + 1 weight register = 16 YMM registers (100% register utilization).
// Weights are loaded and converted once per 3 tokens, cutting memory traffic by 3x.
__declspec(dllexport)
void f16c_gemm_block(
    const float* input,
    const uint16_t* weightsF16,
    float* output,
    int seqRows,
    int inDim,
    int outRows,
    int outStride)
{
    int t = 0;
    for (; t <= seqRows - 3; t += 3) {
        const float* x0 = input + (size_t)(t + 0) * inDim;
        const float* x1 = input + (size_t)(t + 1) * inDim;
        const float* x2 = input + (size_t)(t + 2) * inDim;
        float* out0 = output + (size_t)(t + 0) * outStride;
        float* out1 = output + (size_t)(t + 1) * outStride;
        float* out2 = output + (size_t)(t + 2) * outStride;

        int o = 0;
        for (; o <= outRows - 4; o += 4) {
            const uint16_t* w0 = weightsF16 + (size_t)(o + 0) * inDim;
            const uint16_t* w1 = weightsF16 + (size_t)(o + 1) * inDim;
            const uint16_t* w2 = weightsF16 + (size_t)(o + 2) * inDim;
            const uint16_t* w3 = weightsF16 + (size_t)(o + 3) * inDim;

            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps(), a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps(), a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
            __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps(), a22 = _mm256_setzero_ps(), a23 = _mm256_setzero_ps();

            int i = 0;
            for (; i <= inDim - 8; i += 8) {
                __m256 in0 = _mm256_loadu_ps(x0 + i);
                __m256 in1 = _mm256_loadu_ps(x1 + i);
                __m256 in2 = _mm256_loadu_ps(x2 + i);

                __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0 + i)));
                a00 = _mm256_fmadd_ps(v0, in0, a00);
                a10 = _mm256_fmadd_ps(v0, in1, a10);
                a20 = _mm256_fmadd_ps(v0, in2, a20);

                __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1 + i)));
                a01 = _mm256_fmadd_ps(v1, in0, a01);
                a11 = _mm256_fmadd_ps(v1, in1, a11);
                a21 = _mm256_fmadd_ps(v1, in2, a21);

                __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2 + i)));
                a02 = _mm256_fmadd_ps(v2, in0, a02);
                a12 = _mm256_fmadd_ps(v2, in1, a12);
                a22 = _mm256_fmadd_ps(v2, in2, a22);

                __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3 + i)));
                a03 = _mm256_fmadd_ps(v3, in0, a03);
                a13 = _mm256_fmadd_ps(v3, in1, a13);
                a23 = _mm256_fmadd_ps(v3, in2, a23);
            }

            float s00 = hsum256_ps(a00), s01 = hsum256_ps(a01), s02 = hsum256_ps(a02), s03 = hsum256_ps(a03);
            float s10 = hsum256_ps(a10), s11 = hsum256_ps(a11), s12 = hsum256_ps(a12), s13 = hsum256_ps(a13);
            float s20 = hsum256_ps(a20), s21 = hsum256_ps(a21), s22 = hsum256_ps(a22), s23 = hsum256_ps(a23);

            for (; i < inDim; i++) {
                float xi0 = x0[i];
                float xi1 = x1[i];
                float xi2 = x2[i];

                __m128 wf0 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)w0[i]));
                float f0 = _mm_cvtss_f32(wf0);
                s00 += f0 * xi0; s10 += f0 * xi1; s20 += f0 * xi2;

                __m128 wf1 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)w1[i]));
                float f1 = _mm_cvtss_f32(wf1);
                s01 += f1 * xi0; s11 += f1 * xi1; s21 += f1 * xi2;

                __m128 wf2 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)w2[i]));
                float f2 = _mm_cvtss_f32(wf2);
                s02 += f2 * xi0; s12 += f2 * xi1; s22 += f2 * xi2;

                __m128 wf3 = _mm_cvtph_ps(_mm_cvtsi32_si128((int)w3[i]));
                float f3 = _mm_cvtss_f32(wf3);
                s03 += f3 * xi0; s13 += f3 * xi1; s23 += f3 * xi2;
            }

            out0[o + 0] = s00; out0[o + 1] = s01; out0[o + 2] = s02; out0[o + 3] = s03;
            out1[o + 0] = s10; out1[o + 1] = s11; out1[o + 2] = s12; out1[o + 3] = s13;
            out2[o + 0] = s20; out2[o + 1] = s21; out2[o + 2] = s22; out2[o + 3] = s23;
        }

        for (; o < outRows; o++) {
            const uint16_t* wRow = weightsF16 + (size_t)o * inDim;
            out0[o] = f16c_dot(x0, wRow, inDim);
            out1[o] = f16c_dot(x1, wRow, inDim);
            out2[o] = f16c_dot(x2, wRow, inDim);
        }
    }

    // Process leftover 1 or 2 tokens
    for (; t < seqRows; t++) {
        const float* x = input + (size_t)t * inDim;
        float* out = output + (size_t)t * outStride;
        f16c_matvec_rows(x, weightsF16, inDim, outRows, out);
    }
}
