#include <immintrin.h>
#include <stdint.h>

// Real hardware F16C-based dot product: converts 8 packed fp16 weight values per iteration
// to float32 via VCVTPH2PS (a single instruction, ~3-4 cycle latency) directly into a register,
// FMA'd against the already-F32 activation vector. Weights never touch a scratch F32 buffer.
// This is the exact mechanism ggml's ggml_vec_dot_f16 uses (see docs/audio-review-progress.md's
// ggml investigation entry) that .NET's managed SIMD API cannot express (Half is not a legal
// Vector128/256<T> element type in .NET 10, and both a hand-rolled software bit-trick AND relying
// on .NET's scalar (float)Half cast measured 9-15x SLOWER than plain F32 -- see the same doc entry).
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

    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 sums = _mm_add_ps(sum128, shuf);
    __m128 shuf2 = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(0, 1, 2, 3));
    float sum = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

    for (; i < k; i++) {
        // scalar leftover tail -- also hardware F16C (single-value convert), still cheap
        __m128i wb = _mm_cvtsi32_si128((int)weightF16Bits[i]);
        __m128 wf = _mm_cvtph_ps(wb);
        sum += _mm_cvtss_f32(wf) * input[i];
    }

    return sum;
}

// Batched: computes n dot products (one per weight row) against the same input vector.
// Parallelization is left to the C# caller (Parallel.For over rows), this just does one row.
__declspec(dllexport)
void f16c_matvec_row(const float* input, const uint16_t* weightF16Bits, int k, float* output) {
    *output = f16c_dot(input, weightF16Bits, k);
}
