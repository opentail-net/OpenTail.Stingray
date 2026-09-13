using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class FluxGemmMicrobenchmarkTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _output = output;

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private struct MultiHeadAttentionTiledParams
    {
        public uint qSeq;
        public uint kvSeq;
        public uint numHeads;
        public float scale;
    }

    // Baseline MultiHeadAttentionTiled128 (from Shaders.cs)
    private const string MultiHeadAttentionTiled128_Baseline = """
        #version 450
        #extension GL_KHR_shader_subgroup_basic  : require
        #extension GL_KHR_shader_subgroup_shuffle : require

        #define HEAD_DIM   128
        #define BR         8
        #define BC         32
        #define WG_SIZE    256
        #define D_SPLIT    32
        #define D_PER_LANE 4

        const float NEG_INF = -3.402823466e+38;

        layout(std430, binding = 0) readonly  buffer QBuffer { float q_data[]; };
        layout(std430, binding = 1) readonly  buffer KBuffer { float k_data[]; };
        layout(std430, binding = 2) readonly  buffer VBuffer { float v_data[]; };
        layout(std430, binding = 3) writeonly buffer OBuffer { float o_data[]; };

        layout(push_constant) uniform Params {
            uint qSeq;
            uint kvSeq;
            uint numHeads;
            float scale;
        } p;

        shared float q_tile[BR * HEAD_DIM];
        shared float k_tile[BC * HEAD_DIM];
        shared float v_tile[BC * HEAD_DIM];
        shared float score_tile[BR * BC];
        shared float row_m[BR];
        shared float row_l[BR];
        shared float row_alpha[BR];

        layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

        void main() {
            const uint tid = gl_LocalInvocationIndex;
            const uint row_in_tile = tid / D_SPLIT;
            const uint lane_in_row = tid % D_SPLIT;
            const uint q_row = gl_WorkGroupID.x * BR + row_in_tile;

            const uint h = gl_WorkGroupID.z;
            const uint dim = p.numHeads * HEAD_DIM;
            const uint headOff = h * HEAD_DIM;

            for (uint i = tid; i < BR * HEAD_DIM; i += WG_SIZE) {
                uint r = i / HEAD_DIM;
                uint d = i % HEAD_DIM;
                uint globalQRow = gl_WorkGroupID.x * BR + r;
                q_tile[i] = (globalQRow < p.qSeq) ? q_data[globalQRow * dim + headOff + d] : 0.0;
            }

            if (lane_in_row == 0u) {
                row_m[row_in_tile] = NEG_INF;
                row_l[row_in_tile] = 0.0;
                row_alpha[row_in_tile] = 0.0;
            }
            barrier();

            float accum0 = 0.0, accum1 = 0.0, accum2 = 0.0, accum3 = 0.0;
            bool q_valid = (q_row < p.qSeq);

            for (uint tile_base = 0u; tile_base < p.kvSeq; tile_base += BC) {
                for (uint i = tid; i < BC * HEAD_DIM; i += WG_SIZE) {
                    uint krow = i / HEAD_DIM;
                    uint d = i % HEAD_DIM;
                    uint globalK = tile_base + krow;
                    if (globalK < p.kvSeq) {
                        k_tile[i] = k_data[globalK * dim + headOff + d];
                        v_tile[i] = v_data[globalK * dim + headOff + d];
                    } else {
                        k_tile[i] = 0.0;
                        v_tile[i] = 0.0;
                    }
                }
                barrier();

                if (q_valid) {
                    uint qBaseLocal = row_in_tile * HEAD_DIM;
                    uint d0 = lane_in_row * D_PER_LANE;
                    uint d1 = d0 + 1u, d2 = d0 + 2u, d3 = d0 + 3u;

                    for (uint k = 0u; k < BC; ++k) {
                        uint globalK = tile_base + k;
                        float partial = q_tile[qBaseLocal + d0] * k_tile[k * HEAD_DIM + d0]
                                       + q_tile[qBaseLocal + d1] * k_tile[k * HEAD_DIM + d1]
                                       + q_tile[qBaseLocal + d2] * k_tile[k * HEAD_DIM + d2]
                                       + q_tile[qBaseLocal + d3] * k_tile[k * HEAD_DIM + d3];

                        partial += subgroupShuffleXor(partial, 16u);
                        partial += subgroupShuffleXor(partial, 8u);
                        partial += subgroupShuffleXor(partial, 4u);
                        partial += subgroupShuffleXor(partial, 2u);
                        partial += subgroupShuffleXor(partial, 1u);

                        float score = partial * p.scale;
                        if (globalK >= p.kvSeq) score = NEG_INF;
                        if (lane_in_row == 0u) score_tile[row_in_tile * BC + k] = score;
                    }
                }
                barrier();

                if (lane_in_row == 0u && q_valid) {
                    float m_old = row_m[row_in_tile];
                    float l_old = row_l[row_in_tile];

                    float tile_max = NEG_INF;
                    for (uint k = 0u; k < BC; ++k) {
                        uint globalK = tile_base + k;
                        if (globalK < p.kvSeq)
                            tile_max = max(tile_max, score_tile[row_in_tile * BC + k]);
                    }

                    float m_new = max(m_old, tile_max);
                    float alpha = exp(m_old - m_new);
                    row_alpha[row_in_tile] = alpha;

                    float tile_sum = 0.0;
                    for (uint k = 0u; k < BC; ++k) {
                        uint globalK = tile_base + k;
                        float prob = 0.0;
                        if (globalK < p.kvSeq) {
                            float score = score_tile[row_in_tile * BC + k];
                            prob = exp(score - m_new);
                            tile_sum += prob;
                        }
                        score_tile[row_in_tile * BC + k] = prob;
                    }

                    row_l[row_in_tile] = l_old * alpha + tile_sum;
                    row_m[row_in_tile] = m_new;
                }
                barrier();

                if (q_valid) {
                    uint d0 = lane_in_row * D_PER_LANE;
                    uint d1 = d0 + 1u, d2 = d0 + 2u, d3 = d0 + 3u;
                    float alpha = row_alpha[row_in_tile];

                    accum0 *= alpha; accum1 *= alpha; accum2 *= alpha; accum3 *= alpha;

                    for (uint k = 0u; k < BC; ++k) {
                        uint globalK = tile_base + k;
                        if (globalK < p.kvSeq) {
                            float prob = score_tile[row_in_tile * BC + k];
                            uint vbase = k * HEAD_DIM;
                            accum0 += prob * v_tile[vbase + d0];
                            accum1 += prob * v_tile[vbase + d1];
                            accum2 += prob * v_tile[vbase + d2];
                            accum3 += prob * v_tile[vbase + d3];
                        }
                    }
                }
                barrier();
            }

            if (q_valid) {
                float l = row_l[row_in_tile];
                uint outBase = q_row * dim + headOff;
                uint d0 = lane_in_row * D_PER_LANE;
                uint d1 = d0 + 1u, d2 = d0 + 2u, d3 = d0 + 3u;

                if (p.kvSeq > 0u && l > 0.0) {
                    float invL = 1.0 / l;
                    o_data[outBase + d0] = accum0 * invL;
                    o_data[outBase + d1] = accum1 * invL;
                    o_data[outBase + d2] = accum2 * invL;
                    o_data[outBase + d3] = accum3 * invL;
                } else {
                    o_data[outBase + d0] = 0.0;
                    o_data[outBase + d1] = 0.0;
                    o_data[outBase + d2] = 0.0;
                    o_data[outBase + d3] = 0.0;
                }
            }
        }
        """;

    // Candidate 1: FlashAttention-style 16x16 tiled with vec4 vectorized loads and direct register dot products (no shuffle overhead)
    private const string MultiHeadAttentionTiled128_Fast16x16 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define HEAD_DIM   128
        #define BR         16
        #define BC         16
        #define WG_SIZE    64

        const float NEG_INF = -3.402823466e+38;

        layout(std430, binding = 0) readonly  buffer QVec { vec4 q_vec[]; };
        layout(std430, binding = 1) readonly  buffer KVec { vec4 k_vec[]; };
        layout(std430, binding = 2) readonly  buffer VVec { vec4 v_vec[]; };
        layout(std430, binding = 3) writeonly buffer OVec { vec4 o_vec[]; };

        layout(push_constant) uniform Params {
            uint qSeq;
            uint kvSeq;
            uint numHeads;
            float scale;
        } p;

        shared vec4 q_tile[BR * 32];
        shared vec4 k_tile[BC * 32];
        shared vec4 v_tile[BC * 32];
        shared float score_tile[BR * BC];
        shared float row_m[BR];
        shared float row_l[BR];
        shared float row_alpha[BR];

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        void main() {
            const uint tid = gl_LocalInvocationIndex; // 0..63
            const uint qr = tid >> 2;                 // 0..15 (16 query rows, 4 threads per row)
            const uint sub = tid & 3u;                // 0..3
            const uint kc = sub * 4u;                 // 0, 4, 8, 12 (4 keys per thread)
            const uint q_row = gl_WorkGroupID.x * BR + qr;

            const uint h = gl_WorkGroupID.z;
            const uint headVecOff = (h * HEAD_DIM) >> 2;
            const uint dimVec = (p.numHeads * HEAD_DIM) >> 2;

            // Load Q tile [16, 32 vec4s] = 512 vec4s across 64 threads (8 vec4s / thread)
            [[unroll]] for (uint p_load = 0u; p_load < 8u; ++p_load) {
                uint idx = tid + p_load * 64u;
                uint r = idx >> 5;
                uint d = idx & 31u;
                uint globalQ = gl_WorkGroupID.x * BR + r;
                if (globalQ < p.qSeq) {
                    q_tile[idx] = q_vec[globalQ * dimVec + headVecOff + d];
                } else {
                    q_tile[idx] = vec4(0.0);
                }
            }

            if (tid < BR) {
                row_m[tid] = NEG_INF;
                row_l[tid] = 0.0;
                row_alpha[tid] = 0.0;
            }
            barrier();

            vec4 acc0 = vec4(0.0), acc1 = vec4(0.0), acc2 = vec4(0.0), acc3 = vec4(0.0);
            vec4 acc4 = vec4(0.0), acc5 = vec4(0.0), acc6 = vec4(0.0), acc7 = vec4(0.0);
            bool q_valid = (q_row < p.qSeq);

            for (uint tile_base = 0u; tile_base < p.kvSeq; tile_base += BC) {
                // Load K and V tiles [16, 32 vec4s] = 512 vec4s across 64 threads (8 vec4s / thread)
                [[unroll]] for (uint p_load = 0u; p_load < 8u; ++p_load) {
                    uint idx = tid + p_load * 64u;
                    uint r = idx >> 5;
                    uint d = idx & 31u;
                    uint globalK = tile_base + r;
                    if (globalK < p.kvSeq) {
                        k_tile[idx] = k_vec[globalK * dimVec + headVecOff + d];
                        v_tile[idx] = v_vec[globalK * dimVec + headVecOff + d];
                    } else {
                        k_tile[idx] = vec4(0.0);
                        v_tile[idx] = vec4(0.0);
                    }
                }
                barrier();

                // 1. Compute Q * K dot products: thread computes 4 keys
                if (q_valid) {
                    uint q_base = qr * 32u;
                    float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    [[unroll]] for (uint d = 0u; d < 32u; ++d) {
                        vec4 q_val = q_tile[q_base + d];
                        s0 += dot(q_val, k_tile[(kc + 0u) * 32u + d]);
                        s1 += dot(q_val, k_tile[(kc + 1u) * 32u + d]);
                        s2 += dot(q_val, k_tile[(kc + 2u) * 32u + d]);
                        s3 += dot(q_val, k_tile[(kc + 3u) * 32u + d]);
                    }

                    s0 = (tile_base + kc + 0u < p.kvSeq) ? s0 * p.scale : NEG_INF;
                    s1 = (tile_base + kc + 1u < p.kvSeq) ? s1 * p.scale : NEG_INF;
                    s2 = (tile_base + kc + 2u < p.kvSeq) ? s2 * p.scale : NEG_INF;
                    s3 = (tile_base + kc + 3u < p.kvSeq) ? s3 * p.scale : NEG_INF;

                    score_tile[qr * BC + kc + 0u] = s0;
                    score_tile[qr * BC + kc + 1u] = s1;
                    score_tile[qr * BC + kc + 2u] = s2;
                    score_tile[qr * BC + kc + 3u] = s3;
                }
                barrier();

                // 2. Softmax update: 16 threads (tid 0..15) each handle 1 query row
                if (tid < BR && (gl_WorkGroupID.x * BR + tid < p.qSeq)) {
                    float m_old = row_m[tid];
                    float l_old = row_l[tid];

                    float tile_max = NEG_INF;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        tile_max = max(tile_max, score_tile[tid * BC + k]);
                    }

                    float m_new = max(m_old, tile_max);
                    float alpha = exp(m_old - m_new);
                    row_alpha[tid] = alpha;

                    float tile_sum = 0.0;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = (score_tile[tid * BC + k] > NEG_INF * 0.5) ? exp(score_tile[tid * BC + k] - m_new) : 0.0;
                        score_tile[tid * BC + k] = prob;
                        tile_sum += prob;
                    }

                    row_l[tid] = l_old * alpha + tile_sum;
                    row_m[tid] = m_new;
                }
                barrier();

                // 3. Accumulate P * V: each thread (qr, sub) accumulates 8 vec4s
                if (q_valid) {
                    float alpha = row_alpha[qr];
                    acc0 *= alpha; acc1 *= alpha; acc2 *= alpha; acc3 *= alpha;
                    acc4 *= alpha; acc5 *= alpha; acc6 *= alpha; acc7 *= alpha;

                    uint d_chunk = sub * 8u; // 0, 8, 16, 24
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = score_tile[qr * BC + k];
                        uint v_base = k * 32u + d_chunk;
                        acc0 += prob * v_tile[v_base + 0u];
                        acc1 += prob * v_tile[v_base + 1u];
                        acc2 += prob * v_tile[v_base + 2u];
                        acc3 += prob * v_tile[v_base + 3u];
                        acc4 += prob * v_tile[v_base + 4u];
                        acc5 += prob * v_tile[v_base + 5u];
                        acc6 += prob * v_tile[v_base + 6u];
                        acc7 += prob * v_tile[v_base + 7u];
                    }
                }
                barrier();
            }

            // Write output
            if (q_valid) {
                float l = row_l[qr];
                float invL = (l > 0.0) ? (1.0 / l) : 0.0;
                uint outBase = q_row * dimVec + headVecOff + sub * 8u;
                o_vec[outBase + 0u] = acc0 * invL;
                o_vec[outBase + 1u] = acc1 * invL;
                o_vec[outBase + 2u] = acc2 * invL;
                o_vec[outBase + 3u] = acc3 * invL;
                o_vec[outBase + 4u] = acc4 * invL;
                o_vec[outBase + 5u] = acc5 * invL;
                o_vec[outBase + 6u] = acc6 * invL;
                o_vec[outBase + 7u] = acc7 * invL;
            }
        }
        """;

    // Candidate 2: FlashAttention-style 16x32 tiled with vec4 vectorized loads and 8-key register tiles
    private const string MultiHeadAttentionTiled128_Fast16x32 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define HEAD_DIM   128
        #define BR         16
        #define BC         32
        #define WG_SIZE    64

        const float NEG_INF = -3.402823466e+38;

        layout(std430, binding = 0) readonly  buffer QVec { vec4 q_vec[]; };
        layout(std430, binding = 1) readonly  buffer KVec { vec4 k_vec[]; };
        layout(std430, binding = 2) readonly  buffer VVec { vec4 v_vec[]; };
        layout(std430, binding = 3) writeonly buffer OVec { vec4 o_vec[]; };

        layout(push_constant) uniform Params {
            uint qSeq;
            uint kvSeq;
            uint numHeads;
            float scale;
        } p;

        shared vec4 q_tile[BR * 32];
        shared vec4 k_tile[BC * 32];
        shared vec4 v_tile[BC * 32];
        shared float score_tile[BR * BC];
        shared float row_m[BR];
        shared float row_l[BR];
        shared float row_alpha[BR];

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        void main() {
            const uint tid = gl_LocalInvocationIndex; // 0..63
            const uint qr = tid >> 2;                 // 0..15 (16 query rows, 4 threads per row)
            const uint sub = tid & 3u;                // 0..3
            const uint kc = sub * 8u;                 // 0, 8, 16, 24 (8 keys per thread)
            const uint q_row = gl_WorkGroupID.x * BR + qr;

            const uint h = gl_WorkGroupID.z;
            const uint headVecOff = (h * HEAD_DIM) >> 2;
            const uint dimVec = (p.numHeads * HEAD_DIM) >> 2;

            // Load Q tile [16, 32 vec4s] = 512 vec4s across 64 threads (8 vec4s / thread)
            [[unroll]] for (uint p_load = 0u; p_load < 8u; ++p_load) {
                uint idx = tid + p_load * 64u;
                uint r = idx >> 5;
                uint d = idx & 31u;
                uint globalQ = gl_WorkGroupID.x * BR + r;
                if (globalQ < p.qSeq) {
                    q_tile[idx] = q_vec[globalQ * dimVec + headVecOff + d];
                } else {
                    q_tile[idx] = vec4(0.0);
                }
            }

            if (tid < BR) {
                row_m[tid] = NEG_INF;
                row_l[tid] = 0.0;
                row_alpha[tid] = 0.0;
            }
            barrier();

            vec4 acc0 = vec4(0.0), acc1 = vec4(0.0), acc2 = vec4(0.0), acc3 = vec4(0.0);
            vec4 acc4 = vec4(0.0), acc5 = vec4(0.0), acc6 = vec4(0.0), acc7 = vec4(0.0);
            bool q_valid = (q_row < p.qSeq);

            for (uint tile_base = 0u; tile_base < p.kvSeq; tile_base += BC) {
                // Load K and V tiles [32, 32 vec4s] = 1024 vec4s across 64 threads (16 vec4s / thread)
                [[unroll]] for (uint p_load = 0u; p_load < 16u; ++p_load) {
                    uint idx = tid + p_load * 64u;
                    uint r = idx >> 5;
                    uint d = idx & 31u;
                    uint globalK = tile_base + r;
                    if (globalK < p.kvSeq) {
                        k_tile[idx] = k_vec[globalK * dimVec + headVecOff + d];
                        v_tile[idx] = v_vec[globalK * dimVec + headVecOff + d];
                    } else {
                        k_tile[idx] = vec4(0.0);
                        v_tile[idx] = vec4(0.0);
                    }
                }
                barrier();

                // 1. Compute Q * K dot products: thread computes 8 keys in parallel
                if (q_valid) {
                    uint q_base = qr * 32u;
                    float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    float s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
                    [[unroll]] for (uint d = 0u; d < 32u; ++d) {
                        vec4 q_val = q_tile[q_base + d];
                        s0 += dot(q_val, k_tile[(kc + 0u) * 32u + d]);
                        s1 += dot(q_val, k_tile[(kc + 1u) * 32u + d]);
                        s2 += dot(q_val, k_tile[(kc + 2u) * 32u + d]);
                        s3 += dot(q_val, k_tile[(kc + 3u) * 32u + d]);
                        s4 += dot(q_val, k_tile[(kc + 4u) * 32u + d]);
                        s5 += dot(q_val, k_tile[(kc + 5u) * 32u + d]);
                        s6 += dot(q_val, k_tile[(kc + 6u) * 32u + d]);
                        s7 += dot(q_val, k_tile[(kc + 7u) * 32u + d]);
                    }

                    s0 = (tile_base + kc + 0u < p.kvSeq) ? s0 * p.scale : NEG_INF;
                    s1 = (tile_base + kc + 1u < p.kvSeq) ? s1 * p.scale : NEG_INF;
                    s2 = (tile_base + kc + 2u < p.kvSeq) ? s2 * p.scale : NEG_INF;
                    s3 = (tile_base + kc + 3u < p.kvSeq) ? s3 * p.scale : NEG_INF;
                    s4 = (tile_base + kc + 4u < p.kvSeq) ? s4 * p.scale : NEG_INF;
                    s5 = (tile_base + kc + 5u < p.kvSeq) ? s5 * p.scale : NEG_INF;
                    s6 = (tile_base + kc + 6u < p.kvSeq) ? s6 * p.scale : NEG_INF;
                    s7 = (tile_base + kc + 7u < p.kvSeq) ? s7 * p.scale : NEG_INF;

                    score_tile[qr * BC + kc + 0u] = s0;
                    score_tile[qr * BC + kc + 1u] = s1;
                    score_tile[qr * BC + kc + 2u] = s2;
                    score_tile[qr * BC + kc + 3u] = s3;
                    score_tile[qr * BC + kc + 4u] = s4;
                    score_tile[qr * BC + kc + 5u] = s5;
                    score_tile[qr * BC + kc + 6u] = s6;
                    score_tile[qr * BC + kc + 7u] = s7;
                }
                barrier();

                // 2. Softmax update: 16 threads (tid 0..15) each handle 1 query row
                if (tid < BR && (gl_WorkGroupID.x * BR + tid < p.qSeq)) {
                    float m_old = row_m[tid];
                    float l_old = row_l[tid];

                    float tile_max = NEG_INF;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        tile_max = max(tile_max, score_tile[tid * BC + k]);
                    }

                    float m_new = max(m_old, tile_max);
                    float alpha = exp(m_old - m_new);
                    row_alpha[tid] = alpha;

                    float tile_sum = 0.0;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = (score_tile[tid * BC + k] > NEG_INF * 0.5) ? exp(score_tile[tid * BC + k] - m_new) : 0.0;
                        score_tile[tid * BC + k] = prob;
                        tile_sum += prob;
                    }

                    row_l[tid] = l_old * alpha + tile_sum;
                    row_m[tid] = m_new;
                }
                barrier();

                // 3. Accumulate P * V: each thread (qr, sub) accumulates 8 vec4s
                if (q_valid) {
                    float alpha = row_alpha[qr];
                    acc0 *= alpha; acc1 *= alpha; acc2 *= alpha; acc3 *= alpha;
                    acc4 *= alpha; acc5 *= alpha; acc6 *= alpha; acc7 *= alpha;

                    uint d_chunk = sub * 8u; // 0, 8, 16, 24
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = score_tile[qr * BC + k];
                        uint v_base = k * 32u + d_chunk;
                        acc0 += prob * v_tile[v_base + 0u];
                        acc1 += prob * v_tile[v_base + 1u];
                        acc2 += prob * v_tile[v_base + 2u];
                        acc3 += prob * v_tile[v_base + 3u];
                        acc4 += prob * v_tile[v_base + 4u];
                        acc5 += prob * v_tile[v_base + 5u];
                        acc6 += prob * v_tile[v_base + 6u];
                        acc7 += prob * v_tile[v_base + 7u];
                    }
                }
                barrier();
            }

            // Write output
            if (q_valid) {
                float l = row_l[qr];
                float invL = (l > 0.0) ? (1.0 / l) : 0.0;
                uint outBase = q_row * dimVec + headVecOff + sub * 8u;
                o_vec[outBase + 0u] = acc0 * invL;
                o_vec[outBase + 1u] = acc1 * invL;
                o_vec[outBase + 2u] = acc2 * invL;
                o_vec[outBase + 3u] = acc3 * invL;
                o_vec[outBase + 4u] = acc4 * invL;
                o_vec[outBase + 5u] = acc5 * invL;
                o_vec[outBase + 6u] = acc6 * invL;
                o_vec[outBase + 7u] = acc7 * invL;
            }
        }
        """;

    // Candidate 3: FlashAttention-style 32x16 tiled with 128 threads (2 wavefronts) - halves total K/V global memory loads!
    private const string MultiHeadAttentionTiled128_Fast32x16 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define HEAD_DIM   128
        #define BR         32
        #define BC         16
        #define WG_SIZE    128

        const float NEG_INF = -3.402823466e+38;

        layout(std430, binding = 0) readonly  buffer QVec { vec4 q_vec[]; };
        layout(std430, binding = 1) readonly  buffer KVec { vec4 k_vec[]; };
        layout(std430, binding = 2) readonly  buffer VVec { vec4 v_vec[]; };
        layout(std430, binding = 3) writeonly buffer OVec { vec4 o_vec[]; };

        layout(push_constant) uniform Params {
            uint qSeq;
            uint kvSeq;
            uint numHeads;
            float scale;
        } p;

        shared vec4 q_tile[BR * 32];
        shared vec4 k_tile[BC * 32];
        shared vec4 v_tile[BC * 32];
        shared float score_tile[BR * BC];
        shared float row_m[BR];
        shared float row_l[BR];
        shared float row_alpha[BR];

        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

        void main() {
            const uint tid = gl_LocalInvocationIndex; // 0..127
            const uint qr = tid >> 2;                 // 0..31 (32 query rows, 4 threads per row)
            const uint sub = tid & 3u;                // 0..3
            const uint kc = sub * 4u;                 // 0, 4, 8, 12 (4 keys per thread)
            const uint q_row = gl_WorkGroupID.x * BR + qr;

            const uint h = gl_WorkGroupID.z;
            const uint headVecOff = (h * HEAD_DIM) >> 2;
            const uint dimVec = (p.numHeads * HEAD_DIM) >> 2;

            // Load Q tile [32, 32 vec4s] = 1024 vec4s across 128 threads (8 vec4s / thread)
            [[unroll]] for (uint p_load = 0u; p_load < 8u; ++p_load) {
                uint idx = tid + p_load * 128u;
                uint r = idx >> 5;
                uint d = idx & 31u;
                uint globalQ = gl_WorkGroupID.x * BR + r;
                if (globalQ < p.qSeq) {
                    q_tile[idx] = q_vec[globalQ * dimVec + headVecOff + d];
                } else {
                    q_tile[idx] = vec4(0.0);
                }
            }

            if (tid < BR) {
                row_m[tid] = NEG_INF;
                row_l[tid] = 0.0;
                row_alpha[tid] = 0.0;
            }
            barrier();

            vec4 acc0 = vec4(0.0), acc1 = vec4(0.0), acc2 = vec4(0.0), acc3 = vec4(0.0);
            vec4 acc4 = vec4(0.0), acc5 = vec4(0.0), acc6 = vec4(0.0), acc7 = vec4(0.0);
            bool q_valid = (q_row < p.qSeq);

            for (uint tile_base = 0u; tile_base < p.kvSeq; tile_base += BC) {
                // Load K and V tiles [16, 32 vec4s] = 512 vec4s across 128 threads (4 vec4s / thread)
                [[unroll]] for (uint p_load = 0u; p_load < 4u; ++p_load) {
                    uint idx = tid + p_load * 128u;
                    uint r = idx >> 5;
                    uint d = idx & 31u;
                    uint globalK = tile_base + r;
                    if (globalK < p.kvSeq) {
                        k_tile[idx] = k_vec[globalK * dimVec + headVecOff + d];
                        v_tile[idx] = v_vec[globalK * dimVec + headVecOff + d];
                    } else {
                        k_tile[idx] = vec4(0.0);
                        v_tile[idx] = vec4(0.0);
                    }
                }
                barrier();

                // 1. Compute Q * K dot products: thread computes 4 keys
                if (q_valid) {
                    uint q_base = qr * 32u;
                    float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    [[unroll]] for (uint d = 0u; d < 32u; ++d) {
                        vec4 q_val = q_tile[q_base + d];
                        s0 += dot(q_val, k_tile[(kc + 0u) * 32u + d]);
                        s1 += dot(q_val, k_tile[(kc + 1u) * 32u + d]);
                        s2 += dot(q_val, k_tile[(kc + 2u) * 32u + d]);
                        s3 += dot(q_val, k_tile[(kc + 3u) * 32u + d]);
                    }

                    s0 = (tile_base + kc + 0u < p.kvSeq) ? s0 * p.scale : NEG_INF;
                    s1 = (tile_base + kc + 1u < p.kvSeq) ? s1 * p.scale : NEG_INF;
                    s2 = (tile_base + kc + 2u < p.kvSeq) ? s2 * p.scale : NEG_INF;
                    s3 = (tile_base + kc + 3u < p.kvSeq) ? s3 * p.scale : NEG_INF;

                    score_tile[qr * BC + kc + 0u] = s0;
                    score_tile[qr * BC + kc + 1u] = s1;
                    score_tile[qr * BC + kc + 2u] = s2;
                    score_tile[qr * BC + kc + 3u] = s3;
                }
                barrier();

                // 2. Softmax update: 32 threads (tid 0..31) each handle 1 query row
                if (tid < BR && (gl_WorkGroupID.x * BR + tid < p.qSeq)) {
                    float m_old = row_m[tid];
                    float l_old = row_l[tid];

                    float tile_max = NEG_INF;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        tile_max = max(tile_max, score_tile[tid * BC + k]);
                    }

                    float m_new = max(m_old, tile_max);
                    float alpha = exp(m_old - m_new);
                    row_alpha[tid] = alpha;

                    float tile_sum = 0.0;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = (score_tile[tid * BC + k] > NEG_INF * 0.5) ? exp(score_tile[tid * BC + k] - m_new) : 0.0;
                        score_tile[tid * BC + k] = prob;
                        tile_sum += prob;
                    }

                    row_l[tid] = l_old * alpha + tile_sum;
                    row_m[tid] = m_new;
                }
                barrier();

                // 3. Accumulate P * V: each thread (qr, sub) accumulates 8 vec4s
                if (q_valid) {
                    float alpha = row_alpha[qr];
                    acc0 *= alpha; acc1 *= alpha; acc2 *= alpha; acc3 *= alpha;
                    acc4 *= alpha; acc5 *= alpha; acc6 *= alpha; acc7 *= alpha;

                    uint d_chunk = sub * 8u; // 0, 8, 16, 24
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = score_tile[qr * BC + k];
                        uint v_base = k * 32u + d_chunk;
                        acc0 += prob * v_tile[v_base + 0u];
                        acc1 += prob * v_tile[v_base + 1u];
                        acc2 += prob * v_tile[v_base + 2u];
                        acc3 += prob * v_tile[v_base + 3u];
                        acc4 += prob * v_tile[v_base + 4u];
                        acc5 += prob * v_tile[v_base + 5u];
                        acc6 += prob * v_tile[v_base + 6u];
                        acc7 += prob * v_tile[v_base + 7u];
                    }
                }
                barrier();
            }

            // Write output
            if (q_valid) {
                float l = row_l[qr];
                float invL = (l > 0.0) ? (1.0 / l) : 0.0;
                uint outBase = q_row * dimVec + headVecOff + sub * 8u;
                o_vec[outBase + 0u] = acc0 * invL;
                o_vec[outBase + 1u] = acc1 * invL;
                o_vec[outBase + 2u] = acc2 * invL;
                o_vec[outBase + 3u] = acc3 * invL;
                o_vec[outBase + 4u] = acc4 * invL;
                o_vec[outBase + 5u] = acc5 * invL;
                o_vec[outBase + 6u] = acc6 * invL;
                o_vec[outBase + 7u] = acc7 * invL;
            }
        }
        """;

    [Fact]
    public unsafe void CompareAttentionKernels()
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null) return;

        const int numHeads = 24;
        const int headDim = 128;
        const int qSeq = 1280;
        const int kvSeq = 1280;

        int totalFloats = qSeq * numHeads * headDim;
        var rng = new Random(42);
        var qData = new float[totalFloats];
        for (int i = 0; i < qData.Length; i++) qData[i] = (float)(rng.NextDouble() * 0.1);
        var kData = new float[totalFloats];
        for (int i = 0; i < kData.Length; i++) kData[i] = (float)(rng.NextDouble() * 0.1);
        var vData = new float[totalFloats];
        for (int i = 0; i < vData.Length; i++) vData[i] = (float)(rng.NextDouble() * 0.1);

        var q = vulkan.Upload(qData, TensorShape.D1(totalFloats), exact: true);
        var k = vulkan.Upload(kData, TensorShape.D1(totalFloats), exact: true);
        var v = vulkan.Upload(vData, TensorShape.D1(totalFloats), exact: true);
        var oRef = vulkan.Allocate(TensorShape.D1(totalFloats));
        var oTest1 = vulkan.Allocate(TensorShape.D1(totalFloats));
        var oTest2 = vulkan.Allocate(TensorShape.D1(totalFloats));
        var oTest3 = vulkan.Allocate(TensorShape.D1(totalFloats));

        using var pipeBase = new ComputePipeline(vulkan, MultiHeadAttentionTiled128_Baseline, 4, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));
        using var pipeFast1 = new ComputePipeline(vulkan, MultiHeadAttentionTiled128_Fast16x16, 4, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));
        using var pipeFast2 = new ComputePipeline(vulkan, MultiHeadAttentionTiled128_Fast16x32, 4, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));
        using var pipeFast3 = new ComputePipeline(vulkan, MultiHeadAttentionTiled128_Fast32x16, 4, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));

        try
        {
            var p = new MultiHeadAttentionTiledParams
            {
                qSeq = (uint)qSeq,
                kvSeq = (uint)kvSeq,
                numHeads = (uint)numHeads,
                scale = 1f / MathF.Sqrt(headDim)
            };

            const int iterations = 5;

            // 1. Baseline (BR=8, WG=256)
            uint groupsX_Base = (uint)((qSeq + 7) / 8);
            pipeBase.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(q), vulkan.GetBuffer(k), vulkan.GetBuffer(v), vulkan.GetBuffer(oRef)], groupsX_Base, 1u, (uint)numHeads, &p);
            vulkan.Synchronize();

            var swBase = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                pipeBase.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(q), vulkan.GetBuffer(k), vulkan.GetBuffer(v), vulkan.GetBuffer(oRef)], groupsX_Base, 1u, (uint)numHeads, &p);
            vulkan.Synchronize();
            swBase.Stop();
            double msBase = swBase.Elapsed.TotalMilliseconds / iterations;

            // 2. Fast 16x16 (BR=16, WG=64)
            uint groupsX_Fast16 = (uint)((qSeq + 15) / 16);
            pipeFast1.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(q), vulkan.GetBuffer(k), vulkan.GetBuffer(v), vulkan.GetBuffer(oTest1)], groupsX_Fast16, 1u, (uint)numHeads, &p);
            vulkan.Synchronize();

            var swFast1 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                pipeFast1.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(q), vulkan.GetBuffer(k), vulkan.GetBuffer(v), vulkan.GetBuffer(oTest1)], groupsX_Fast16, 1u, (uint)numHeads, &p);
            vulkan.Synchronize();
            swFast1.Stop();
            double msFast1 = swFast1.Elapsed.TotalMilliseconds / iterations;

            // 3. Fast 16x32 (BR=16, WG=64)
            pipeFast2.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(q), vulkan.GetBuffer(k), vulkan.GetBuffer(v), vulkan.GetBuffer(oTest2)], groupsX_Fast16, 1u, (uint)numHeads, &p);
            vulkan.Synchronize();

            var swFast2 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                pipeFast2.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(q), vulkan.GetBuffer(k), vulkan.GetBuffer(v), vulkan.GetBuffer(oTest2)], groupsX_Fast16, 1u, (uint)numHeads, &p);
            vulkan.Synchronize();
            swFast2.Stop();
            double msFast2 = swFast2.Elapsed.TotalMilliseconds / iterations;

            // 4. Fast 32x16 (BR=32, WG=128)
            uint groupsX_Fast32 = (uint)((qSeq + 31) / 32);
            pipeFast3.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(q), vulkan.GetBuffer(k), vulkan.GetBuffer(v), vulkan.GetBuffer(oTest3)], groupsX_Fast32, 1u, (uint)numHeads, &p);
            vulkan.Synchronize();

            var swFast3 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                pipeFast3.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(q), vulkan.GetBuffer(k), vulkan.GetBuffer(v), vulkan.GetBuffer(oTest3)], groupsX_Fast32, 1u, (uint)numHeads, &p);
            vulkan.Synchronize();
            swFast3.Stop();
            double msFast3 = swFast3.Elapsed.TotalMilliseconds / iterations;

            // Check parity
            var refHost = new float[totalFloats];
            var testHost1 = new float[totalFloats];
            var testHost2 = new float[totalFloats];
            var testHost3 = new float[totalFloats];
            vulkan.Download(oRef, refHost);
            vulkan.Download(oTest1, testHost1);
            vulkan.Download(oTest2, testHost2);
            vulkan.Download(oTest3, testHost3);

            float maxDiff1 = 0f, maxDiff2 = 0f, maxDiff3 = 0f;
            for (int i = 0; i < totalFloats; i++)
            {
                float diff1 = MathF.Abs(refHost[i] - testHost1[i]);
                if (diff1 > maxDiff1) maxDiff1 = diff1;
                float diff2 = MathF.Abs(refHost[i] - testHost2[i]);
                if (diff2 > maxDiff2) maxDiff2 = diff2;
                float diff3 = MathF.Abs(refHost[i] - testHost3[i]);
                if (diff3 > maxDiff3) maxDiff3 = diff3;
            }

            string results = $"[ATTENTION BASELINE]  -> {msBase:F2} ms ({(4.0 * numHeads * qSeq * kvSeq * headDim) / (msBase * 1e6):F1} GFLOP/s)\n" +
                             $"[ATTENTION FAST16x16] -> {msFast1:F2} ms ({(4.0 * numHeads * qSeq * kvSeq * headDim) / (msFast1 * 1e6):F1} GFLOP/s) -- speedup: {msBase / msFast1:F2}x, maxDiff: {maxDiff1:E3}\n" +
                             $"[ATTENTION FAST16x32] -> {msFast2:F2} ms ({(4.0 * numHeads * qSeq * kvSeq * headDim) / (msFast2 * 1e6):F1} GFLOP/s) -- speedup: {msBase / msFast2:F2}x, maxDiff: {maxDiff2:E3}\n" +
                             $"[ATTENTION FAST32x16] -> {msFast3:F2} ms ({(4.0 * numHeads * qSeq * kvSeq * headDim) / (msFast3 * 1e6):F1} GFLOP/s) -- speedup: {msBase / msFast3:F2}x, maxDiff: {maxDiff3:E3}\n";
            File.WriteAllText("attention_bench.txt", results);

            Assert.True(maxDiff1 < 1e-4f, $"Parity mismatch 16x16: maxDiff={maxDiff1}");
            Assert.True(maxDiff2 < 1e-4f, $"Parity mismatch 16x32: maxDiff={maxDiff2}");
            Assert.True(maxDiff3 < 1e-4f, $"Parity mismatch 32x16: maxDiff={maxDiff3}");
        }
        finally
        {
            vulkan.Free(q);
            vulkan.Free(k);
            vulkan.Free(v);
            vulkan.Free(oRef);
            vulkan.Free(oTest1);
            vulkan.Free(oTest2);
            vulkan.Free(oTest3);
        }
    }
}
