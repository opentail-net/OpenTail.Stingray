// Q6_K x Q8_K dot-product harness: llama.cpp reference numbers for the C# comparison.
//
// Purpose: answer ONE question — does RyuJIT emit competitive code for this kernel, or not?
// That decides whether OpenTail.Stingray's CPU prefill gap is codegen (a ceiling to design around)
// or structural (threading/layout/dispatch — fixable).
//
// This is an ISOLATED microbenchmark, which OpenTail's perf log has repeatedly shown does NOT
// predict end-to-end behaviour. That is fine here: we are not asking "will this make the app
// faster", we are comparing two implementations of the SAME algorithm on the SAME data. Whatever
// distorts one side distorts the other. The RATIO transfers; the absolute magnitude does not.
//
// Build/run: see README.md in this directory.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "quants.h"
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// Deterministic, reproducible input. Not random: the C# side must feed byte-identical data, and
// "same seed, different RNG" is exactly how two harnesses end up measuring different work.
static float synth(int i) {
    return std::sin(static_cast<double>(i) * 0.017) * 2.0 + std::cos(static_cast<double>(i) * 0.0031);
}

struct Result { double best_ms; double mean_ms; double sd_ms; float value; };

template <typename Fn>
static Result bench(Fn fn, int reps, int warmup) {
    float out = 0.0f;
    for (int i = 0; i < warmup; ++i) out = fn();

    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        double t0 = now_ms();
        out = fn();
        samples.push_back(now_ms() - t0);
    }
    double sum = 0.0;
    for (double s : samples) sum += s;
    double mean = sum / samples.size();
    double var = 0.0;
    for (double s : samples) var += (s - mean) * (s - mean);
    return Result{
        *std::min_element(samples.begin(), samples.end()),
        mean,
        std::sqrt(var / samples.size()),
        out
    };
}

int main(int argc, char** argv) {
    // Shape matches OpenTail's reference model FFN width so the comparison is against a real
    // tensor shape, not a synthetic one that happens to sit in L1.
    const int64_t k    = (argc > 1) ? std::atoll(argv[1]) : 8192;   // elements per dot
    const int     rows = (argc > 2) ? std::atoi(argv[2])  : 512;    // dots per timed iteration
    const int     reps = (argc > 3) ? std::atoi(argv[3])  : 8;

    if (k % QK_K != 0) { std::fprintf(stderr, "k must be a multiple of %d\n", QK_K); return 1; }
    const int64_t nblk = k / QK_K;

    // Direct CPU quant/dot entry points use the CPU backend's FP16 lookup table. ggml_init alone
    // creates a tensor context but does not initialize that backend table; skipping ggml_cpu_init
    // produces zero-valued dots with plausible timings.
    ggml_cpu_init();
    ggml_init_params ip{};
    ip.mem_size = 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);

    std::vector<float> src((size_t)k);
    for (int64_t i = 0; i < k; ++i) src[i] = synth((int)i);

    // Activation side: one Q8_K row.
    std::vector<uint8_t> vy((size_t)nblk * sizeof(block_q8_K));
    quantize_row_q8_K(src.data(), vy.data(), k);

    // Weight side: `rows` independent Q6_K rows, so the working set resembles a real matvec rather
    // than one row replayed out of L1.
    const size_t q6k_row_bytes = (size_t)nblk * sizeof(block_q6_K);
    std::vector<uint8_t> vx((size_t)rows * q6k_row_bytes);
    {
        std::vector<float> rowsrc((size_t)k);
        std::vector<int64_t> hist(1 << 4, 0);
        for (int r = 0; r < rows; ++r) {
            for (int64_t i = 0; i < k; ++i) rowsrc[i] = synth((int)(i + r * 7919));
            quantize_row_q6_K_ref(rowsrc.data(), (block_q6_K*)(vx.data() + (size_t)r * q6k_row_bytes), k);
        }
        (void)hist;
    }

    // A non-zero scale proves the direct quantizer calls produced real blocks before the
    // vectorised and generic dots are timed. Keep this diagnostic in the harness: a plausible
    // timing from zero-filled blocks is worse than a hard failure.
    const auto * q8 = reinterpret_cast<const block_q8_K *>(vy.data());
    const auto * q6 = reinterpret_cast<const block_q6_K *>(vx.data());
    std::printf("input check q8.d=%.8f q6.d=%g q6.scale0=%d\n",
                q8[0].d, ggml_fp16_to_fp32(q6[0].d), (int)q6[0].scales[0]);

    auto run_arch = [&]() {
        float acc = 0.0f;
        for (int r = 0; r < rows; ++r) {
            float s = 0.0f;
            ggml_vec_dot_q6_K_q8_K((int)k, &s, 0, vx.data() + (size_t)r * q6k_row_bytes, 0, vy.data(), 0, 1);
            acc += s;
        }
        return acc;
    };
    auto run_generic = [&]() {
        float acc = 0.0f;
        for (int r = 0; r < rows; ++r) {
            float s = 0.0f;
            ggml_vec_dot_q6_K_q8_K_generic((int)k, &s, 0, vx.data() + (size_t)r * q6k_row_bytes, 0, vy.data(), 0, 1);
            acc += s;
        }
        return acc;
    };

    Result arch    = bench(run_arch,    reps, 3);
    Result generic = bench(run_generic, reps, 3);

    const double dots = (double)rows;
    std::printf("k=%lld rows=%d reps=%d  (QK_K=%d, %lld blocks/row)\n",
                (long long)k, rows, reps, QK_K, (long long)nblk);
    std::printf("checksum arch    = %.6f\n", arch.value);
    std::printf("checksum generic = %.6f\n", generic.value);
    std::printf("arch    best=%.4f ms  mean=%.4f ms  sd=%.4f  (%.1f Mdot/s)\n",
                arch.best_ms, arch.mean_ms, arch.sd_ms, dots / arch.best_ms / 1000.0);
    std::printf("generic best=%.4f ms  mean=%.4f ms  sd=%.4f  (%.1f Mdot/s)\n",
                generic.best_ms, generic.mean_ms, generic.sd_ms, dots / generic.best_ms / 1000.0);

    // Self-checks. Both matter more than the timings: a plausible number computed from the wrong
    // work is the failure mode this whole comparison exists to avoid.
    const float diff = std::fabs(arch.value - generic.value);
    const float tol  = std::fabs(generic.value) * 1e-4f + 1e-3f;
    if (diff > tol) {
        std::printf("\nFAIL: arch and generic disagree (|d|=%.6f > tol=%.6f) — they are not "
                    "computing the same thing; do not use these numbers.\n", diff, tol);
        ggml_free(ctx);
        return 2;
    }
    const double speedup = generic.best_ms / arch.best_ms;
    std::printf("\narch is %.2fx the generic scalar reference.\n", speedup);
    if (speedup < 1.15) {
        std::printf("WARNING: barely faster than the scalar reference — you have very likely linked\n"
                    "the GENERIC ggml-cpu variant. Benchmarking that and comparing it to C# would\n"
                    "invert the conclusion. Force a single-arch build (see README) before trusting this.\n");
        ggml_free(ctx);
        return 3;
    }

    std::printf("\nFeed these to the C# side: same k, same rows, same synth() sequence.\n");
    std::printf("Compare CHECKSUMS FIRST. Only compare timings once they agree.\n");
    ggml_free(ctx);
    return 0;
}
