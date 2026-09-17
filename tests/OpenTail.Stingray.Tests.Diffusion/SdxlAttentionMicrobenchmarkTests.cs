using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class SdxlAttentionMicrobenchmarkTests
{
    private readonly ITestOutputHelper _output;

    public SdxlAttentionMicrobenchmarkTests(ITestOutputHelper output)
    {
        _output = output;
    }

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

    // ─────────────────────────────────────────────────────────────────────────────
    // Model 3: Unfused Discrete GEMM Pipeline (QK GEMM -> Softmax -> PV GEMM)
    // ─────────────────────────────────────────────────────────────────────────────
    private const string ScoreGemm_Unfused = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define HEAD_DIM   64
        #define TILE_Q     16
        #define TILE_K     16

        layout(std430, binding = 0) readonly  buffer QVec { vec4 q_vec[]; };
        layout(std430, binding = 1) readonly  buffer KVec { vec4 k_vec[]; };
        layout(std430, binding = 2) writeonly buffer SBuf { float scores[]; };

        layout(push_constant) uniform Params {
            uint qSeq;
            uint kvSeq;
            uint numHeads;
            float scale;
        } p;

        shared vec4 q_tile[TILE_Q * 16];
        shared vec4 k_tile[TILE_K * 16];

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        void main() {
            uint lx = gl_LocalInvocationID.x;
            uint ly = gl_LocalInvocationID.y;
            uint tid = gl_LocalInvocationIndex;

            uint q_row = gl_WorkGroupID.y * TILE_Q + ly;
            uint k_row = gl_WorkGroupID.x * TILE_K + lx;
            uint h = gl_WorkGroupID.z;

            uint headVecOff = (h * HEAD_DIM) >> 2;
            uint dimVec = (p.numHeads * HEAD_DIM) >> 2;

            uint qr = tid >> 4;
            uint d = tid & 15u;
            uint globalQ = gl_WorkGroupID.y * TILE_Q + qr;
            if (globalQ < p.qSeq) {
                q_tile[tid] = q_vec[globalQ * dimVec + headVecOff + d];
            } else {
                q_tile[tid] = vec4(0.0);
            }

            uint globalK = gl_WorkGroupID.x * TILE_K + qr;
            if (globalK < p.kvSeq) {
                k_tile[tid] = k_vec[globalK * dimVec + headVecOff + d];
            } else {
                k_tile[tid] = vec4(0.0);
            }
            barrier();

            if (q_row < p.qSeq && k_row < p.kvSeq) {
                float sum = 0.0;
                uint q_base = ly * 16u;
                uint k_base = lx * 16u;
                [[unroll]] for (uint i = 0u; i < 16u; ++i) {
                    sum += dot(q_tile[q_base + i], k_tile[k_base + i]);
                }
                scores[h * p.qSeq * p.kvSeq + q_row * p.kvSeq + k_row] = sum * p.scale;
            }
        }
        """;

    private const string RowSoftmax_Unfused = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(std430, binding = 0) buffer SBuf { float scores[]; };

        layout(push_constant) uniform Params {
            uint qSeq;
            uint kvSeq;
            uint numHeads;
            float scale;
        } p;

        shared float sdata[128];

        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

        void main() {
            uint tid = gl_LocalInvocationIndex;
            uint q = gl_WorkGroupID.x;
            uint h = gl_WorkGroupID.y;

            if (q >= p.qSeq || h >= p.numHeads) return;

            uint rowBase = (h * p.qSeq + q) * p.kvSeq;

            float localMax = -3.402823466e+38;
            for (uint k = tid; k < p.kvSeq; k += 128u) {
                localMax = max(localMax, scores[rowBase + k]);
            }
            sdata[tid] = localMax;
            barrier();

            for (uint s = 64u; s > 0u; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float rowMax = sdata[0];

            float localSum = 0.0;
            for (uint k = tid; k < p.kvSeq; k += 128u) {
                float val = exp(scores[rowBase + k] - rowMax);
                scores[rowBase + k] = val;
                localSum += val;
            }
            sdata[tid] = localSum;
            barrier();

            for (uint s = 64u; s > 0u; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float invSum = (sdata[0] > 0.0) ? (1.0 / sdata[0]) : 0.0;

            for (uint k = tid; k < p.kvSeq; k += 128u) {
                scores[rowBase + k] *= invSum;
            }
        }
        """;

    private const string ValueGemm_Unfused = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define HEAD_DIM 64
        #define BR 16
        #define BC 16

        layout(std430, binding = 0) readonly  buffer SBuf { float scores[]; };
        layout(std430, binding = 1) readonly  buffer VVec { vec4 v_vec[]; };
        layout(std430, binding = 2) writeonly buffer OVec { vec4 o_vec[]; };

        layout(push_constant) uniform Params {
            uint qSeq;
            uint kvSeq;
            uint numHeads;
            float scale;
        } p;

        shared vec4  v_tile[BC * 16];
        shared float s_tile[BR * BC];

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        void main() {
            uint lx = gl_LocalInvocationID.x;
            uint ly = gl_LocalInvocationID.y;
            uint tid = gl_LocalInvocationIndex;

            uint q_row = gl_WorkGroupID.x * BR + ly;
            uint h = gl_WorkGroupID.z;

            uint headVecOff = (h * HEAD_DIM) >> 2;
            uint dimVec = (p.numHeads * HEAD_DIM) >> 2;

            vec4 acc = vec4(0.0);

            for (uint tile_base = 0u; tile_base < p.kvSeq; tile_base += BC) {
                uint vr = tid >> 4;
                uint vd = tid & 15u;
                uint globalK = tile_base + vr;
                if (globalK < p.kvSeq) {
                    v_tile[tid] = v_vec[globalK * dimVec + headVecOff + vd];
                } else {
                    v_tile[tid] = vec4(0.0);
                }

                uint sq = tid >> 4;
                uint sk = tid & 15u;
                uint globalQ = gl_WorkGroupID.x * BR + sq;
                uint globalS_K = tile_base + sk;
                if (globalQ < p.qSeq && globalS_K < p.kvSeq) {
                    s_tile[tid] = scores[h * p.qSeq * p.kvSeq + globalQ * p.kvSeq + globalS_K];
                } else {
                    s_tile[tid] = 0.0;
                }
                barrier();

                if (q_row < p.qSeq) {
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = s_tile[ly * BC + k];
                        acc += prob * v_tile[k * 16u + lx];
                    }
                }
                barrier();
            }

            if (q_row < p.qSeq) {
                o_vec[q_row * dimVec + headVecOff + lx] = acc;
            }
        }
        """;

    // ─────────────────────────────────────────────────────────────────────────────
    // Model 4: Fused Online Softmax Variants (FP32 and FP16)
    // ─────────────────────────────────────────────────────────────────────────────
    private const string MultiHeadAttentionTiled64_Vec4 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define HEAD_DIM   64
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

        shared vec4 q_tile[BR * 16];
        shared vec4 k_tile[BC * 16];
        shared vec4 v_tile[BC * 16];
        shared float score_tile[BR * BC];
        shared float row_m[BR];
        shared float row_l[BR];
        shared float row_alpha[BR];

        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

        void main() {
            const uint tid = gl_LocalInvocationIndex;
            const uint qr = tid >> 2;
            const uint sub = tid & 3u;
            const uint kc = sub * 4u;
            const uint q_row = gl_WorkGroupID.x * BR + qr;

            const uint h = gl_WorkGroupID.z;
            const uint headVecOff = (h * HEAD_DIM) >> 2;
            const uint dimVec = (p.numHeads * HEAD_DIM) >> 2;

            [[unroll]] for (uint p_load = 0u; p_load < 4u; ++p_load) {
                uint idx = tid + p_load * 128u;
                uint r = idx >> 4;
                uint d = idx & 15u;
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
            bool q_valid = (q_row < p.qSeq);

            for (uint tile_base = 0u; tile_base < p.kvSeq; tile_base += BC) {
                [[unroll]] for (uint p_load = 0u; p_load < 2u; ++p_load) {
                    uint idx = tid + p_load * 128u;
                    uint r = idx >> 4;
                    uint d = idx & 15u;
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

                if (q_valid) {
                    uint q_base = qr * 16u;
                    float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    [[unroll]] for (uint d = 0u; d < 16u; ++d) {
                        vec4 q_val = q_tile[q_base + d];
                        s0 += dot(q_val, k_tile[(kc + 0u) * 16u + d]);
                        s1 += dot(q_val, k_tile[(kc + 1u) * 16u + d]);
                        s2 += dot(q_val, k_tile[(kc + 2u) * 16u + d]);
                        s3 += dot(q_val, k_tile[(kc + 3u) * 16u + d]);
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

                if (q_valid) {
                    float alpha = row_alpha[qr];
                    acc0 *= alpha; acc1 *= alpha; acc2 *= alpha; acc3 *= alpha;

                    uint d_chunk = sub * 4u;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = score_tile[qr * BC + k];
                        uint v_base = k * 16u + d_chunk;
                        acc0 += prob * v_tile[v_base + 0u];
                        acc1 += prob * v_tile[v_base + 1u];
                        acc2 += prob * v_tile[v_base + 2u];
                        acc3 += prob * v_tile[v_base + 3u];
                    }
                }
                barrier();
            }

            if (q_valid) {
                float l = row_l[qr];
                float invL = (l > 0.0) ? (1.0 / l) : 0.0;
                uint outBase = q_row * dimVec + headVecOff + sub * 4u;
                o_vec[outBase + 0u] = acc0 * invL;
                o_vec[outBase + 1u] = acc1 * invL;
                o_vec[outBase + 2u] = acc2 * invL;
                o_vec[outBase + 3u] = acc3 * invL;
            }
        }
        """;

    private const string MultiHeadAttentionTiled64_BR16_WG64 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define HEAD_DIM   64
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

        shared vec4 q_tile[BR * 16];
        shared vec4 k_tile[BC * 16];
        shared vec4 v_tile[BC * 16];
        shared float score_tile[BR * BC];
        shared float row_m[BR];
        shared float row_l[BR];
        shared float row_alpha[BR];

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        void main() {
            const uint tid = gl_LocalInvocationIndex;
            const uint qr = tid >> 2;
            const uint sub = tid & 3u;
            const uint kc = sub * 4u;
            const uint q_row = gl_WorkGroupID.x * BR + qr;

            const uint h = gl_WorkGroupID.z;
            const uint headVecOff = (h * HEAD_DIM) >> 2;
            const uint dimVec = (p.numHeads * HEAD_DIM) >> 2;

            [[unroll]] for (uint p_load = 0u; p_load < 4u; ++p_load) {
                uint idx = tid + p_load * 64u;
                uint r = idx >> 4;
                uint d = idx & 15u;
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
            bool q_valid = (q_row < p.qSeq);

            for (uint tile_base = 0u; tile_base < p.kvSeq; tile_base += BC) {
                [[unroll]] for (uint p_load = 0u; p_load < 4u; ++p_load) {
                    uint idx = tid + p_load * 64u;
                    uint r = idx >> 4;
                    uint d = idx & 15u;
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

                if (q_valid) {
                    uint q_base = qr * 16u;
                    float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    [[unroll]] for (uint d = 0u; d < 16u; ++d) {
                        vec4 q_val = q_tile[q_base + d];
                        s0 += dot(q_val, k_tile[(kc + 0u) * 16u + d]);
                        s1 += dot(q_val, k_tile[(kc + 1u) * 16u + d]);
                        s2 += dot(q_val, k_tile[(kc + 2u) * 16u + d]);
                        s3 += dot(q_val, k_tile[(kc + 3u) * 16u + d]);
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

                if (q_valid) {
                    float alpha = row_alpha[qr];
                    acc0 *= alpha; acc1 *= alpha; acc2 *= alpha; acc3 *= alpha;

                    uint d_chunk = sub * 4u;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = score_tile[qr * BC + k];
                        uint v_base = k * 16u + d_chunk;
                        acc0 += prob * v_tile[v_base + 0u];
                        acc1 += prob * v_tile[v_base + 1u];
                        acc2 += prob * v_tile[v_base + 2u];
                        acc3 += prob * v_tile[v_base + 3u];
                    }
                }
                barrier();
            }

            if (q_valid) {
                float l = row_l[qr];
                float invL = (l > 0.0) ? (1.0 / l) : 0.0;
                uint outBase = q_row * dimVec + headVecOff + sub * 4u;
                o_vec[outBase + 0u] = acc0 * invL;
                o_vec[outBase + 1u] = acc1 * invL;
                o_vec[outBase + 2u] = acc2 * invL;
                o_vec[outBase + 3u] = acc3 * invL;
            }
        }
        """;

    private const string MultiHeadAttentionTiled64_FP16 = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require
        #extension GL_EXT_control_flow_attributes : enable

        #define HEAD_DIM   64
        #define BR         32
        #define BC         16
        #define WG_SIZE    128

        const float NEG_INF = -3.402823466e+38;

        layout(std430, binding = 0) readonly  buffer QVec { f16vec4 q_vec[]; };
        layout(std430, binding = 1) readonly  buffer KVec { f16vec4 k_vec[]; };
        layout(std430, binding = 2) readonly  buffer VVec { f16vec4 v_vec[]; };
        layout(std430, binding = 3) writeonly buffer OVec { vec4    o_vec[]; };

        layout(push_constant) uniform Params {
            uint qSeq;
            uint kvSeq;
            uint numHeads;
            float scale;
        } p;

        shared f16vec4 q_tile[BR * 16];
        shared f16vec4 k_tile[BC * 16];
        shared f16vec4 v_tile[BC * 16];
        shared float score_tile[BR * BC];
        shared float row_m[BR];
        shared float row_l[BR];
        shared float row_alpha[BR];

        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

        void main() {
            const uint tid = gl_LocalInvocationIndex;
            const uint qr = tid >> 2;
            const uint sub = tid & 3u;
            const uint kc = sub * 4u;
            const uint q_row = gl_WorkGroupID.x * BR + qr;

            const uint h = gl_WorkGroupID.z;
            const uint headVecOff = (h * HEAD_DIM) >> 2;
            const uint dimVec = (p.numHeads * HEAD_DIM) >> 2;

            [[unroll]] for (uint p_load = 0u; p_load < 4u; ++p_load) {
                uint idx = tid + p_load * 128u;
                uint r = idx >> 4;
                uint d = idx & 15u;
                uint globalQ = gl_WorkGroupID.x * BR + r;
                if (globalQ < p.qSeq) {
                    q_tile[idx] = q_vec[globalQ * dimVec + headVecOff + d];
                } else {
                    q_tile[idx] = f16vec4(0.0);
                }
            }

            if (tid < BR) {
                row_m[tid] = NEG_INF;
                row_l[tid] = 0.0;
                row_alpha[tid] = 0.0;
            }
            barrier();

            vec4 acc0 = vec4(0.0), acc1 = vec4(0.0), acc2 = vec4(0.0), acc3 = vec4(0.0);
            bool q_valid = (q_row < p.qSeq);

            for (uint tile_base = 0u; tile_base < p.kvSeq; tile_base += BC) {
                [[unroll]] for (uint p_load = 0u; p_load < 2u; ++p_load) {
                    uint idx = tid + p_load * 128u;
                    uint r = idx >> 4;
                    uint d = idx & 15u;
                    uint globalK = tile_base + r;
                    if (globalK < p.kvSeq) {
                        k_tile[idx] = k_vec[globalK * dimVec + headVecOff + d];
                        v_tile[idx] = v_vec[globalK * dimVec + headVecOff + d];
                    } else {
                        k_tile[idx] = f16vec4(0.0);
                        v_tile[idx] = f16vec4(0.0);
                    }
                }
                barrier();

                if (q_valid) {
                    uint q_base = qr * 16u;
                    float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    [[unroll]] for (uint d = 0u; d < 16u; ++d) {
                        vec4 q_val = vec4(q_tile[q_base + d]);
                        s0 += dot(q_val, vec4(k_tile[(kc + 0u) * 16u + d]));
                        s1 += dot(q_val, vec4(k_tile[(kc + 1u) * 16u + d]));
                        s2 += dot(q_val, vec4(k_tile[(kc + 2u) * 16u + d]));
                        s3 += dot(q_val, vec4(k_tile[(kc + 3u) * 16u + d]));
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

                if (q_valid) {
                    float alpha = row_alpha[qr];
                    acc0 *= alpha; acc1 *= alpha; acc2 *= alpha; acc3 *= alpha;

                    uint d_chunk = sub * 4u;
                    [[unroll]] for (uint k = 0u; k < BC; ++k) {
                        float prob = score_tile[qr * BC + k];
                        uint v_base = k * 16u + d_chunk;
                        acc0 += prob * vec4(v_tile[v_base + 0u]);
                        acc1 += prob * vec4(v_tile[v_base + 1u]);
                        acc2 += prob * vec4(v_tile[v_base + 2u]);
                        acc3 += prob * vec4(v_tile[v_base + 3u]);
                    }
                }
                barrier();
            }

            if (q_valid) {
                float l = row_l[qr];
                float invL = (l > 0.0) ? (1.0 / l) : 0.0;
                uint outBase = q_row * dimVec + headVecOff + sub * 4u;
                o_vec[outBase + 0u] = acc0 * invL;
                o_vec[outBase + 1u] = acc1 * invL;
                o_vec[outBase + 2u] = acc2 * invL;
                o_vec[outBase + 3u] = acc3 * invL;
            }
        }
        """;

    private static float[] CreateRandomData(int size, int seed)
    {
        var rng = new Random(seed);
        var data = new float[size];
        for (int i = 0; i < size; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return data;
    }

    private static Half[] ToHalf(float[] data)
    {
        var halfs = new Half[data.Length];
        for (int i = 0; i < data.Length; i++)
            halfs[i] = (Half)data[i];
        return halfs;
    }

    [Theory]
    [InlineData(4096, 4096, 10, 64, 3, "SDXL L1 Self-Attn (1024x1024)")]
    [InlineData(4096, 77,   10, 64, 5, "SDXL L1 Cross-Attn (1024x1024)")]
    [InlineData(1024, 1024, 20, 64, 5, "SDXL L2 Self-Attn (1024x1024)")]
    [InlineData(1024, 77,   20, 64, 5, "SDXL L2 Cross-Attn (1024x1024)")]
    [InlineData(256,  256,  20, 64, 5, "SDXL L2 Self-Attn (512x512)")]
    [InlineData(256,  77,   20, 64, 5, "SDXL L2 Cross-Attn (512x512)")]
    public unsafe void Benchmark_Sdxl_Attention_Shapes(
        int qSeq, int kvSeq, int numHeads, int headDim, int iterations, string shapeLabel)
    {
        using var vulkan = TryCreateVulkan();
        if (vulkan is null)
        {
            _output.WriteLine($"[SKIPPED] Vulkan device not available for {shapeLabel}.");
            return;
        }

        int dim = numHeads * headDim;
        int totalQ = qSeq * dim;
        int totalKV = kvSeq * dim;
        double gflops = (4.0 * numHeads * qSeq * kvSeq * headDim) / 1e9;

        // Effective algorithmic IO footprint: Q + K + V + Output
        double ioBytesFp32 = ((long)totalQ + 2L * totalKV + totalQ) * sizeof(float);
        double ioBytesFp16 = ((long)totalQ + 2L * totalKV) * sizeof(Half) + (long)totalQ * sizeof(float);

        var qData = CreateRandomData(totalQ, 42);
        var kData = CreateRandomData(totalKV, 43);
        var vData = CreateRandomData(totalKV, 44);

        // 1. CPU SIMD
        var outCpu = new float[totalQ];
        DiffusionOps.MultiHeadAttention(qData, kData, vData, outCpu.AsSpan(), qSeq, kvSeq, numHeads, headDim);
        var swCpu = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            DiffusionOps.MultiHeadAttention(qData, kData, vData, outCpu.AsSpan(), qSeq, kvSeq, numHeads, headDim);
        }
        swCpu.Stop();
        double msCpu = swCpu.Elapsed.TotalMilliseconds / iterations;
        double gflopsCpu = gflops / (msCpu / 1000.0);
        double bwCpu = (ioBytesFp32 / 1e9) / (msCpu / 1000.0);

        // Upload FP32 tensors to GPU
        var qGpu = vulkan.Upload(qData, TensorShape.D1(totalQ));
        var kGpu = vulkan.Upload(kData, TensorShape.D1(totalKV));
        var vGpu = vulkan.Upload(vData, TensorShape.D1(totalKV));

        // Upload FP16 tensors to GPU
        var qHalfGpu = vulkan.UploadHalf(ToHalf(qData), TensorShape.D1(totalQ));
        var kHalfGpu = vulkan.UploadHalf(ToHalf(kData), TensorShape.D1(totalKV));
        var vHalfGpu = vulkan.UploadHalf(ToHalf(vData), TensorShape.D1(totalKV));

        var outGpuNaive = vulkan.Allocate(TensorShape.D1(totalQ));
        var outGpuUnfused = vulkan.Allocate(TensorShape.D1(totalQ));
        var outGpuTiledBase = vulkan.Allocate(TensorShape.D1(totalQ));
        var outGpuTiledVec4 = vulkan.Allocate(TensorShape.D1(totalQ));
        var outGpuTiledBR16 = vulkan.Allocate(TensorShape.D1(totalQ));
        var outGpuTiledFP16 = vulkan.Allocate(TensorShape.D1(totalQ));

        Tensor? intermediateScores = null;

        try
        {
            // 2. GPU Naive
            double msNaive = 0, gflopsNaive = 0, bwNaive = 0;
            float maxDiffNaive = 0f;
            if ((long)qSeq * kvSeq <= 4096L * 1024L)
            {
                vulkan.MultiHeadAttention(outGpuNaive, qGpu, kGpu, vGpu, qSeq, kvSeq, numHeads, headDim);
                vulkan.Synchronize();

                var swNaive = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                {
                    vulkan.MultiHeadAttention(outGpuNaive, qGpu, kGpu, vGpu, qSeq, kvSeq, numHeads, headDim);
                }
                vulkan.Synchronize();
                swNaive.Stop();
                msNaive = swNaive.Elapsed.TotalMilliseconds / iterations;
                gflopsNaive = gflops / (msNaive / 1000.0);
                bwNaive = (ioBytesFp32 / 1e9) / (msNaive / 1000.0);

                var outHostNaive = new float[totalQ];
                vulkan.Download(outGpuNaive, outHostNaive);
                for (int i = 0; i < totalQ; i++)
                {
                    float diff = MathF.Abs(outCpu[i] - outHostNaive[i]);
                    if (diff > maxDiffNaive) maxDiffNaive = diff;
                }
            }

            // 3. GPU Unfused Discrete GEMM (Score GEMM -> In-Place Softmax -> Value GEMM)
            double msUnfused = 0, gflopsUnfused = 0, bwUnfused = 0;
            float maxDiffUnfused = 0f;
            long scoreElements = (long)numHeads * qSeq * kvSeq;
            if (scoreElements <= 180_000_000L) // <= ~720MB VRAM
            {
                intermediateScores = vulkan.Allocate(TensorShape.D1((int)scoreElements));
                var pParams = new MultiHeadAttentionTiledParams { qSeq = (uint)qSeq, kvSeq = (uint)kvSeq, numHeads = (uint)numHeads, scale = 1f / MathF.Sqrt(headDim) };

                using var pipeScore = new ComputePipeline(vulkan, ScoreGemm_Unfused, 3, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));
                using var pipeSoftmax = new ComputePipeline(vulkan, RowSoftmax_Unfused, 1, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));
                using var pipeValue = new ComputePipeline(vulkan, ValueGemm_Unfused, 3, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));

                uint gxScore = (uint)((kvSeq + 15) / 16);
                uint gyScore = (uint)((qSeq + 15) / 16);
                uint gzScore = (uint)numHeads;

                uint gxSoftmax = (uint)qSeq;
                uint gySoftmax = (uint)numHeads;

                uint gxValue = (uint)((qSeq + 15) / 16);
                uint gzValue = (uint)numHeads;

                pipeScore.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(qGpu), vulkan.GetBuffer(kGpu), vulkan.GetBuffer(intermediateScores!)], gxScore, gyScore, gzScore, &pParams);
                pipeSoftmax.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(intermediateScores!)], gxSoftmax, gySoftmax, 1u, &pParams);
                pipeValue.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(intermediateScores!), vulkan.GetBuffer(vGpu), vulkan.GetBuffer(outGpuUnfused)], gxValue, 1u, gzValue, &pParams);
                vulkan.Synchronize();

                var swUnfused = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                {
                    pipeScore.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(qGpu), vulkan.GetBuffer(kGpu), vulkan.GetBuffer(intermediateScores!)], gxScore, gyScore, gzScore, &pParams);
                    pipeSoftmax.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(intermediateScores!)], gxSoftmax, gySoftmax, 1u, &pParams);
                    pipeValue.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(intermediateScores!), vulkan.GetBuffer(vGpu), vulkan.GetBuffer(outGpuUnfused)], gxValue, 1u, gzValue, &pParams);
                }
                vulkan.Synchronize();
                swUnfused.Stop();
                msUnfused = swUnfused.Elapsed.TotalMilliseconds / iterations;
                gflopsUnfused = gflops / (msUnfused / 1000.0);
                bwUnfused = (ioBytesFp32 / 1e9) / (msUnfused / 1000.0);

                var outHostUnfused = new float[totalQ];
                vulkan.Download(outGpuUnfused, outHostUnfused);
                for (int i = 0; i < totalQ; i++)
                {
                    float diff = MathF.Abs(outCpu[i] - outHostUnfused[i]);
                    if (diff > maxDiffUnfused) maxDiffUnfused = diff;
                }
            }

            // 4. GPU Tiled Base (Production MultiHeadAttentionTiled from VulkanBackend)
            vulkan.MultiHeadAttentionTiled(outGpuTiledBase, qGpu, kGpu, vGpu, qSeq, kvSeq, numHeads, headDim);
            vulkan.Synchronize();

            var swTiledBase = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                vulkan.MultiHeadAttentionTiled(outGpuTiledBase, qGpu, kGpu, vGpu, qSeq, kvSeq, numHeads, headDim);
            }
            vulkan.Synchronize();
            swTiledBase.Stop();
            double msTiledBase = swTiledBase.Elapsed.TotalMilliseconds / iterations;
            double gflopsTiledBase = gflops / (msTiledBase / 1000.0);
            double bwTiledBase = (ioBytesFp32 / 1e9) / (msTiledBase / 1000.0);

            var outHostTiledBase = new float[totalQ];
            vulkan.Download(outGpuTiledBase, outHostTiledBase);
            float maxDiffTiledBase = 0f;
            for (int i = 0; i < totalQ; i++)
            {
                float diff = MathF.Abs(outCpu[i] - outHostTiledBase[i]);
                if (diff > maxDiffTiledBase) maxDiffTiledBase = diff;
            }

            // 5. GPU Tiled Vec4 FP32 (BR=32, BC=16, WG=128)
            var pTiled = new MultiHeadAttentionTiledParams { qSeq = (uint)qSeq, kvSeq = (uint)kvSeq, numHeads = (uint)numHeads, scale = 1f / MathF.Sqrt(headDim) };
            using var pipeTiledVec4 = new ComputePipeline(vulkan, MultiHeadAttentionTiled64_Vec4, 4, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));
            uint groupsX_Vec4 = (uint)((qSeq + 31) / 32);

            pipeTiledVec4.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(qGpu), vulkan.GetBuffer(kGpu), vulkan.GetBuffer(vGpu), vulkan.GetBuffer(outGpuTiledVec4)], groupsX_Vec4, 1u, (uint)numHeads, &pTiled);
            vulkan.Synchronize();

            var swTiledVec4 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                pipeTiledVec4.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(qGpu), vulkan.GetBuffer(kGpu), vulkan.GetBuffer(vGpu), vulkan.GetBuffer(outGpuTiledVec4)], groupsX_Vec4, 1u, (uint)numHeads, &pTiled);
            }
            vulkan.Synchronize();
            swTiledVec4.Stop();
            double msTiledVec4 = swTiledVec4.Elapsed.TotalMilliseconds / iterations;
            double gflopsTiledVec4 = gflops / (msTiledVec4 / 1000.0);
            double bwTiledVec4 = (ioBytesFp32 / 1e9) / (msTiledVec4 / 1000.0);

            var outHostTiledVec4 = new float[totalQ];
            vulkan.Download(outGpuTiledVec4, outHostTiledVec4);
            float maxDiffTiledVec4 = 0f;
            for (int i = 0; i < totalQ; i++)
            {
                float diff = MathF.Abs(outCpu[i] - outHostTiledVec4[i]);
                if (diff > maxDiffTiledVec4) maxDiffTiledVec4 = diff;
            }

            // 6. GPU Tiled Vec4 FP32 (BR=16, BC=16, WG=64: Single-Wavefront)
            using var pipeTiledBR16 = new ComputePipeline(vulkan, MultiHeadAttentionTiled64_BR16_WG64, 4, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));
            uint groupsX_BR16 = (uint)((qSeq + 15) / 16);
            pipeTiledBR16.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(qGpu), vulkan.GetBuffer(kGpu), vulkan.GetBuffer(vGpu), vulkan.GetBuffer(outGpuTiledBR16)], groupsX_BR16, 1u, (uint)numHeads, &pTiled);
            vulkan.Synchronize();

            var swTiledBR16 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                pipeTiledBR16.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(qGpu), vulkan.GetBuffer(kGpu), vulkan.GetBuffer(vGpu), vulkan.GetBuffer(outGpuTiledBR16)], groupsX_BR16, 1u, (uint)numHeads, &pTiled);
            }
            vulkan.Synchronize();
            swTiledBR16.Stop();
            double msTiledBR16 = swTiledBR16.Elapsed.TotalMilliseconds / iterations;
            double gflopsTiledBR16 = gflops / (msTiledBR16 / 1000.0);
            double bwTiledBR16 = (ioBytesFp32 / 1e9) / (msTiledBR16 / 1000.0);

            var outHostTiledBR16 = new float[totalQ];
            vulkan.Download(outGpuTiledBR16, outHostTiledBR16);
            float maxDiffTiledBR16 = 0f;
            for (int i = 0; i < totalQ; i++)
            {
                float diff = MathF.Abs(outCpu[i] - outHostTiledBR16[i]);
                if (diff > maxDiffTiledBR16) maxDiffTiledBR16 = diff;
            }

            // 7. GPU Tiled Vec4 FP16 (BR=32, BC=16, WG=128: FP16 Q/K/V storage with FP32 math)
            using var pipeTiledFP16 = new ComputePipeline(vulkan, MultiHeadAttentionTiled64_FP16, 4, pushConstantSize: sizeof(MultiHeadAttentionTiledParams));
            pipeTiledFP16.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(qHalfGpu), vulkan.GetBuffer(kHalfGpu), vulkan.GetBuffer(vHalfGpu), vulkan.GetBuffer(outGpuTiledFP16)], groupsX_Vec4, 1u, (uint)numHeads, &pTiled);
            vulkan.Synchronize();

            var swTiledFP16 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                pipeTiledFP16.DispatchWith(vulkan.TransferCmd, [vulkan.GetBuffer(qHalfGpu), vulkan.GetBuffer(kHalfGpu), vulkan.GetBuffer(vHalfGpu), vulkan.GetBuffer(outGpuTiledFP16)], groupsX_Vec4, 1u, (uint)numHeads, &pTiled);
            }
            vulkan.Synchronize();
            swTiledFP16.Stop();
            double msTiledFP16 = swTiledFP16.Elapsed.TotalMilliseconds / iterations;
            double gflopsTiledFP16 = gflops / (msTiledFP16 / 1000.0);
            double bwTiledFP16 = (ioBytesFp16 / 1e9) / (msTiledFP16 / 1000.0);

            var outHostTiledFP16 = new float[totalQ];
            vulkan.Download(outGpuTiledFP16, outHostTiledFP16);
            float maxDiffTiledFP16 = 0f;
            for (int i = 0; i < totalQ; i++)
            {
                float diff = MathF.Abs(outCpu[i] - outHostTiledFP16[i]);
                if (diff > maxDiffTiledFP16) maxDiffTiledFP16 = diff;
            }

            // Log Results Table
            _output.WriteLine("=============================================================================================================================");
            _output.WriteLine($"[SHAPE]: {shapeLabel} | qSeq={qSeq}, kvSeq={kvSeq}, nHeads={numHeads}, headDim={headDim} | Work: {gflops:F3} GFLOP");
            _output.WriteLine("-----------------------------------------------------------------------------------------------------------------------------");
            _output.WriteLine($"  1. CPU SIMD (AVX2+FMA)         : {msCpu,8:F2} ms  |  {gflopsCpu,7:F1} GFLOP/s  | {bwCpu,6:F2} GB/s | (reference)");
            if (msNaive > 0)
                _output.WriteLine($"  2. GPU Naive (GlobalMem)       : {msNaive,8:F2} ms  |  {gflopsNaive,7:F1} GFLOP/s  | {bwNaive,6:F2} GB/s | speedup: {msCpu / msNaive,5:F2}x | maxDiff: {maxDiffNaive:E2}");
            else
                _output.WriteLine($"  2. GPU Naive (GlobalMem)       : [SKIPPED - O(N^2) memory explosion would timeout TDR]");

            if (msUnfused > 0)
                _output.WriteLine($"  3. GPU Unfused GEMM (Discrete) : {msUnfused,8:F2} ms  |  {gflopsUnfused,7:F1} GFLOP/s  | {bwUnfused,6:F2} GB/s | speedup: {msCpu / msUnfused,5:F2}x | maxDiff: {maxDiffUnfused:E2}");
            else
                _output.WriteLine($"  3. GPU Unfused GEMM (Discrete) : [SKIPPED - intermediate scores exceed VRAM limit]");

            _output.WriteLine($"  4. GPU Tiled Base (Engine Prod): {msTiledBase,8:F2} ms  |  {gflopsTiledBase,7:F1} GFLOP/s  | {bwTiledBase,6:F2} GB/s | speedup: {msCpu / msTiledBase,5:F2}x | maxDiff: {maxDiffTiledBase:E2}");
            _output.WriteLine($"  5. GPU Tiled Vec4 FP32 (BR=32) : {msTiledVec4,8:F2} ms  |  {gflopsTiledVec4,7:F1} GFLOP/s  | {bwTiledVec4,6:F2} GB/s | speedup: {msCpu / msTiledVec4,5:F2}x | speedup vs Base: {msTiledBase / msTiledVec4,5:F2}x | maxDiff: {maxDiffTiledVec4:E2}");
            _output.WriteLine($"  6. GPU Tiled Vec4 FP32 (BR=16) : {msTiledBR16,8:F2} ms  |  {gflopsTiledBR16,7:F1} GFLOP/s  | {bwTiledBR16,6:F2} GB/s | speedup: {msCpu / msTiledBR16,5:F2}x | speedup vs BR=32: {msTiledVec4 / msTiledBR16,5:F2}x | maxDiff: {maxDiffTiledBR16:E2}");
            _output.WriteLine($"  7. GPU Tiled Vec4 FP16 (BR=32) : {msTiledFP16,8:F2} ms  |  {gflopsTiledFP16,7:F1} GFLOP/s  | {bwTiledFP16,6:F2} GB/s | speedup: {msCpu / msTiledFP16,5:F2}x | speedup vs FP32: {msTiledVec4 / msTiledFP16,5:F2}x | maxDiff: {maxDiffTiledFP16:E2}");
            _output.WriteLine("=============================================================================================================================");

            if (msUnfused > 0)
                Assert.True(maxDiffUnfused < 1e-2f, $"Unfused parity error {maxDiffUnfused}");
            Assert.True(maxDiffTiledBase < 5e-3f, $"Baseline parity error {maxDiffTiledBase}");
            Assert.True(maxDiffTiledVec4 < 5e-3f, $"Vec4 parity error {maxDiffTiledVec4}");
            Assert.True(maxDiffTiledBR16 < 5e-3f, $"BR16 parity error {maxDiffTiledBR16}");
            Assert.True(maxDiffTiledFP16 < 1e-2f, $"FP16 parity error {maxDiffTiledFP16}");
        }
        finally
        {
            vulkan.Free(qGpu);
            vulkan.Free(kGpu);
            vulkan.Free(vGpu);
            vulkan.Free(qHalfGpu);
            vulkan.Free(kHalfGpu);
            vulkan.Free(vHalfGpu);
            vulkan.Free(outGpuNaive);
            vulkan.Free(outGpuUnfused);
            vulkan.Free(outGpuTiledBase);
            vulkan.Free(outGpuTiledVec4);
            vulkan.Free(outGpuTiledBR16);
            vulkan.Free(outGpuTiledFP16);
            if (intermediateScores is not null) vulkan.Free(intermediateScores);
        }
    }
}
