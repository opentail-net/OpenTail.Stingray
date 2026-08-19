namespace OpenTail.Stingray.Vulkan;

/// <summary>
/// GLSL compute shader source code for all inference operations.
/// Compiled to SPIR-V at runtime via ShaderCompiler.
/// </summary>
internal static class Shaders
{
    /// <summary>
    /// RMS Normalization: output[i] = input[i] / rms * weight[i]
    /// where rms = sqrt(mean(input^2) + eps).
    ///
    /// Uses workgroup shared memory for parallel reduction of sum-of-squares.
    /// Push constants: { uint n, float eps }.
    /// Bindings: 0=input, 1=weight, 2=output.
    /// Dispatch: 1 workgroup of 256 threads.
    /// </summary>
    internal const string RmsNorm = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Input  { float input_data[]; };
        layout(binding = 1) readonly buffer Weight { float weight_data[]; };
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float eps;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint gid = gl_GlobalInvocationID.x;

            // Phase 1: each thread accumulates sum of squares for its stride
            float sum = 0.0;
            for (uint i = tid; i < n; i += 256) {
                float v = input_data[i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            // Phase 2: parallel reduction in shared memory
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            // Phase 3: compute scale factor
            float scale = inversesqrt(sdata[0] / float(n) + eps);

            // Phase 4: apply normalization and weight
            for (uint i = tid; i < n; i += 256) {
                output_data[i] = input_data[i] * scale * weight_data[i];
            }
        }
        """;

    /// <summary>
    /// Batched RMS Normalization: normalizes each of <c>num_tokens</c> independent rows of a
    /// <c>[num_tokens][n]</c> buffer in a single dispatch. Row r (token r) is normalized EXACTLY
    /// as the single-row <see cref="RmsNorm"/> — its own sum-of-squares reduction over its n
    /// elements, then scale + the shared <c>[n]</c> weight. Bit-identical to <c>num_tokens</c>
    /// separate <see cref="RmsNorm"/> calls (the per-row math is independent; floating-point
    /// reduction order within a row matches the single-row shader's 256-stride + tree reduction).
    ///
    /// One workgroup per row: row index r = <c>gl_WorkGroupID.x</c> (dispatch num_tokens groups).
    /// Push constants: { uint n, float eps, uint num_tokens }.
    /// Bindings: 0=input ([num_tokens][n]), 1=weight ([n], shared), 2=output ([num_tokens][n]).
    /// </summary>
    internal const string RmsNormBatched = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Input  { float input_data[]; };
        layout(binding = 1) readonly buffer Weight { float weight_data[]; };
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float eps;
            uint num_tokens;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row = gl_WorkGroupID.x;
            if (row >= num_tokens) return;

            uint base_off = row * n;

            // Phase 1: each thread accumulates sum of squares for its stride within this row.
            float sum = 0.0;
            for (uint i = tid; i < n; i += 256) {
                float v = input_data[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            // Phase 2: parallel reduction in shared memory.
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            // Phase 3: compute scale factor.
            float scale = inversesqrt(sdata[0] / float(n) + eps);

            // Phase 4: apply normalization and the shared weight.
            for (uint i = tid; i < n; i += 256) {
                output_data[base_off + i] = input_data[base_off + i] * scale * weight_data[i];
            }
        }
        """;

    /// <summary>
    /// Fused SiLU(gate) * up: gate[i] = gate[i] * sigmoid(gate[i]) * up[i]
    /// Push constants: { uint n }.
    /// Bindings: 0=gate (in/out), 1=up (in).
    /// </summary>
    internal const string SiLuMul = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Gate { float gate_data[]; };
        layout(binding = 1) readonly buffer Up { float up_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            float g = gate_data[i];
            gate_data[i] = g / (1.0 + exp(-g)) * up_data[i];
        }
        """;

    /// <summary>
    /// Fused tanh-approximate GELU(gate) * up (Gemma FFN activation):
    /// gate[i] = 0.5 * g * (1 + tanh(0.7978845608028654 * (g + 0.044715 * g^3))) * up[i]
    /// where g = gate[i]. Clone of <see cref="SiLuMul"/> with SiLU swapped for GELU-tanh.
    /// Push constants: { uint n }.
    /// Bindings: 0=gate (in/out), 1=up (in).
    /// Matches the CPU reference SimdKernels.GeluTanhMul / CUDA llm_gelu_tanh_mul.
    /// </summary>
    internal const string GeluTanhMul = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Gate { float gate_data[]; };
        layout(binding = 1) readonly buffer Up { float up_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            float g = gate_data[i];
            float inner = 0.7978845608028654 * (g + 0.044715 * g * g * g);
            // GLSL spec-defines tanh(x) as (e^x - e^-x)/(e^x + e^-x), and drivers implement it
            // literally: float32 exp overflows past ~88, so |inner| > ~44 yields inf/inf = NaN.
            // A Gemma 4 E4B gate value of g=20.3 gives inner=315 and produced exactly one NaN in
            // a 10240-wide FFN, which the following ffn_down matmul then spread across all 2560
            // output rows — the whole trunk dead from one element. |inner| > 10 already saturates
            // tanh to +/-1 within float32 precision, so the clamp is invisible to the result.
            // Mirrors the identical clamp in SimdKernels.GeluTanhMul (which is why the CPU
            // backend never showed this).
            inner = clamp(inner, -10.0, 10.0);
            gate_data[i] = 0.5 * g * (1.0 + tanh(inner)) * up_data[i];
        }
        """;

    /// <summary>
    /// SiLU (Swish) activation in-place: x[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i])).
    /// Push constants: { uint n }.
    /// Bindings: 0=x (in/out).
    /// Standalone (unfused) counterpart to <see cref="SiLuMul"/>; matches the CPU
    /// GdnKernels.SiLu / CUDA SiLUInPlace formula.
    /// </summary>
    internal const string SiLU = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            float x = x_data[i];
            x_data[i] = x / (1.0 + exp(-x));
        }
        """;

    /// <summary>
    /// Vector add in-place: dst[i] += src[i]
    /// Push constants: { uint n }.
    /// Bindings: 0=dst (in/out), 1=src (in).
    /// </summary>
    internal const string AddInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Dst { float dst_data[]; };
        layout(binding = 1) readonly buffer Src { float src_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            dst_data[i] += src_data[i];
        }
        """;

    /// <summary>
    /// Vector add in-place with scalar weight: dst[i] += scale * src[i]
    /// Push constants: { uint n, float scale }.
    /// Bindings: 0=dst (in/out), 1=src (in).
    /// </summary>
    internal const string AddScaledInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Dst { float dst_data[]; };
        layout(binding = 1) readonly buffer Src { float src_data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float scale;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            dst_data[i] += scale * src_data[i];
        }
        """;

    /// <summary>
    /// In-place scalar multiply: data[i] *= scale for i in [0, n).
    /// Push constants: { uint n, float scale }.
    /// Bindings: 0=data (in/out).
    /// </summary>
    internal const string ScaleInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float scale;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            data[i] *= scale;
        }
        """;

    /// <summary>
    /// In-place final-logit softcap: x[i] = tanh(x[i] / cap) * cap for i in [0, n).
    /// Used by Gemma to clip extreme logits before sampling (cap=30).
    /// Push constants: { uint n, float cap } (reuses the ScaleParams layout, scale=cap).
    /// Bindings: 0=data (in/out).
    /// Matches the CPU reference SimdKernels.SoftcapInPlace / CUDA llm_softcap_inplace.
    /// </summary>
    internal const string Softcap = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float cap;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            // Same overflow as GeluTanhMul: a pre-softcap logit above ~44*cap makes the driver's
            // tanh evaluate inf/inf = NaN. Clamping the argument to +/-10 is invisible to the
            // cap*tanh result and matches SimdKernels.SoftcapInPlace.
            data[i] = tanh(clamp(data[i] / cap, -10.0, 10.0)) * cap;
        }
        """;

    /// <summary>
    /// Raw buffer copy: dst_data[dst_offset + i] = src_data[src_offset + i] for i in [0, count).
    /// Operates on uint32 words (4-byte aligned). All offsets are in uint32 units.
    /// Push constants: { uint count, uint src_offset, uint dst_offset }.
    /// Bindings: 0=src (readonly), 1=dst (writeonly).
    /// Dispatch: ceil(count / 256) workgroups of 256 threads.
    /// </summary>
    internal const string BufferCopy = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Src { uint src_data[]; };
        layout(binding = 1) writeonly buffer Dst { uint dst_data[]; };

        layout(push_constant) uniform Params {
            uint count;
            uint src_offset;
            uint dst_offset;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= count) return;
            dst_data[dst_offset + i] = src_data[src_offset + i];
        }
        """;

    /// <summary>
    /// Fill a buffer with zeros.
    /// Push constants: { uint n }.
    /// Bindings: 0=dst (in/out).
    /// </summary>
    internal const string Clear = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Dst { float dst_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            dst_data[i] = 0.0;
        }
        """;

    /// <summary>
    /// Gated-DeltaNet depthwise causal conv1d for a single decode token. One thread per
    /// channel. State layout <c>[(kernel-1), channels]</c> row-major, oldest first; updated
    /// in place. Weight layout <c>[kernel, channels]</c>.
    ///   output[c] = weight[K-1,c]*x[c] + Σ_{k=0..K-2} weight[k,c]*state[k,c]
    ///   shift state: state[0..K-3] = state[1..K-2]; state[K-2] = x[c]
    /// Mirrors CUDA llm_gdn_conv1d_decode / CPU GdnKernels.CausalDepthwiseConv1dDecode.
    /// Push constants: { uint channels, uint kernel_size }.
    /// Bindings: 0=x (in), 1=state (in/out), 2=weight (in), 3=output (out).
    /// Dispatch: ceil(channels / 256) workgroups of 256 threads.
    /// </summary>
    internal const string GdnConv1dDecode = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer X     { float x_data[]; };
        layout(binding = 1)           buffer State { float state_data[]; };
        layout(binding = 2) readonly  buffer W     { float w_data[]; };
        layout(binding = 3) writeonly buffer O     { float o_data[]; };

        layout(push_constant) uniform Params {
            uint channels;
            uint kernel_size;
        };

        void main() {
            uint c = gl_GlobalInvocationID.x;
            if (c >= channels) return;

            uint retained = kernel_size - 1u;

            // Read old state values into registers (kernel_size <= 4 in our models).
            float s_old[4];
            for (uint k = 0u; k < retained; k++)
                s_old[k] = state_data[k * channels + c];

            float x_c = x_data[c];
            float sum = w_data[retained * channels + c] * x_c;
            for (uint k = 0u; k < retained; k++)
                sum += w_data[k * channels + c] * s_old[k];
            o_data[c] = sum;

            // Shift state forward in time (drop oldest, append x).
            for (uint k = 0u; k + 1u < retained; k++)
                state_data[k * channels + c] = s_old[k + 1u];
            if (retained >= 1u)
                state_data[(retained - 1u) * channels + c] = x_c;
        }
        """;

    /// <summary>
    /// Gated-DeltaNet L2 normalization per head (no learned weights). One workgroup per head,
    /// 256-thread tree reduction. Matches GdnKernels.L2NormPerHead / CUDA llm_gdn_l2_norm_per_head:
    ///   scale = 1 / max(sqrt(Σ x²), eps).
    /// This differs from <see cref="HeadNormPure"/> which divides by sqrt(mean + eps). Operates on
    /// the sub-region of the bound buffer starting at <c>offset</c> float elements.
    /// Push constants: { uint head_dim, uint num_heads, float eps, uint offset }.
    /// Bindings: 0=data (in/out).
    /// Dispatch: num_heads workgroups of 256 threads.
    /// </summary>
    internal const string GdnL2NormPerHead = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data_buf[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
            uint offset;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            if (head >= num_heads) return;

            uint base_off = offset + head * head_dim;

            float sum = 0.0;
            for (uint i = tid; i < head_dim; i += 256u) {
                float v = data_buf[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            [[unroll]] for (uint s = 128u; s > 0u; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            float norm = sqrt(sdata[0]);
            float divisor = norm > eps ? norm : eps;
            float inv = 1.0 / divisor;
            for (uint i = tid; i < head_dim; i += 256u) {
                data_buf[base_off + i] = data_buf[base_off + i] * inv;
            }
        }
        """;

    /// <summary>
    /// Gated-DeltaNet tile-heads (GQA-style broadcast). One thread per dst element.
    ///   dst[h_dst, j] = src[h_dst % src_heads, j] for h_dst in [0, src_heads*repeat).
    /// Matches GdnKernels.TileHeads / CUDA llm_gdn_tile_heads (tile, NOT torch repeat_interleave).
    /// <c>src_offset</c>/<c>dst_offset</c> are float-element offsets into the bound buffers.
    /// Push constants: { uint src_heads, uint repeat, uint head_dim, uint src_offset, uint dst_offset }.
    /// Bindings: 0=src (in), 1=dst (out).
    /// Dispatch: ceil(src_heads*repeat*head_dim / 256) workgroups of 256 threads.
    /// </summary>
    internal const string GdnTileHeads = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Src { float src_data[]; };
        layout(binding = 1) writeonly buffer Dst { float dst_data[]; };

        layout(push_constant) uniform Params {
            uint src_heads;
            uint repeat;
            uint head_dim;
            uint src_offset;
            uint dst_offset;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            uint total = src_heads * repeat * head_dim;
            if (idx >= total) return;
            uint j = idx % head_dim;
            uint h_dst = idx / head_dim;
            uint h_src = h_dst % src_heads;
            dst_data[dst_offset + idx] = src_data[src_offset + h_src * head_dim + j];
        }
        """;

    /// <summary>
    /// Gated-DeltaNet recurrence delta-rule scan for a single decode token (issue #356).
    /// One workgroup per v-head; <c>local_size_x = headDim</c> (HARDCODED 128 — both target
    /// models qwen36-35b-a3b / qwen36-27b-mtp have headDim=128, so every one of the 128
    /// invocations is active and reaches every <c>barrier()</c>). Each thread owns output
    /// column <c>j</c>. State layout <c>S[h*d*d + i*d + j]</c> (i=key axis, j=value/output
    /// axis), updated in place. Per head:
    ///   decay = exp(softplus(alpha_in[h]+dt_bias[h]) · ssm_a[h]); b = sigmoid(beta[h])
    ///   pass A: S *= decay; p[j] = Σ_i k[i]·S[i,j]
    ///   d[j]   = b·(v[j] − p[j])
    ///   pass B: S[i,j] += k[i]·d[j]; o[j] = (1/√d)·Σ_i q[i]·S[i,j]
    ///   o = RMSNorm(o)·norm_weight; o *= SiLU(z)
    /// Mirrors CUDA <c>llm_gdn_recurrence_decode</c> / CPU <c>GdnKernels.GdnRecurrenceDecode</c>
    /// op-for-op (full-precision exp/log/inversesqrt to track the CPU oracle tightly).
    /// Push constants: { uint hv, uint d, float norm_eps }.
    /// Bindings: 0=state (in/out), 1=q, 2=k, 3=v, 4=alpha_in, 5=beta, 6=ssm_a, 7=dt_bias,
    ///           8=norm_weight, 9=z (all readonly), 10=output (writeonly).
    /// Dispatch: hv workgroups of 128 threads.
    /// </summary>
    internal const string GdnRecurrenceDecode = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 128) in;

        layout(binding = 0)           buffer State  { float state_data[]; };
        layout(binding = 1) readonly  buffer Q      { float q_data[]; };
        layout(binding = 2) readonly  buffer K      { float k_data[]; };
        layout(binding = 3) readonly  buffer V      { float v_data[]; };
        layout(binding = 4) readonly  buffer AlphaIn{ float alpha_data[]; };
        layout(binding = 5) readonly  buffer Beta   { float beta_data[]; };
        layout(binding = 6) readonly  buffer SsmA   { float ssma_data[]; };
        layout(binding = 7) readonly  buffer DtBias { float dtbias_data[]; };
        layout(binding = 8) readonly  buffer NormW  { float normw_data[]; };
        layout(binding = 9) readonly  buffer Z      { float z_data[]; };
        layout(binding = 10) writeonly buffer O     { float o_data[]; };

        layout(push_constant) uniform Params {
            uint hv;
            uint d;
            float norm_eps;
        };

        shared float sK[128];
        shared float sQ[128];
        shared float sV[128];
        shared float sZ[128];
        shared float sNormW[128];
        shared float sP[128];
        shared float sD[128];
        shared float sRed[128];

        void main() {
            uint h = gl_WorkGroupID.x;
            uint j = gl_LocalInvocationID.x;

            // Load per-head Q, K, V, Z and per-dim norm weight into shared memory.
            uint hd_off = h * d;
            sK[j]     = k_data[hd_off + j];
            sQ[j]     = q_data[hd_off + j];
            sV[j]     = v_data[hd_off + j];
            sZ[j]     = z_data[hd_off + j];
            sNormW[j] = normw_data[j];
            barrier();

            // Per-head scalar gates.
            float alpha_x = alpha_data[h] + dtbias_data[h];
            float dt      = alpha_x >= 20.0 ? alpha_x : log(1.0 + exp(alpha_x));   // softplus
            float decay   = exp(dt * ssma_data[h]);
            float b_sc    = 1.0 / (1.0 + exp(-beta_data[h]));

            // d is fixed at 128 (== local_size_x == headDim, enforced by the wrapper). Using the
            // literal lets the SPIR-V compiler strength-reduce i*128 → i<<7 and unroll the loops.
            uint state_base = h * 16384u;   // h * d * d

            // Pass A: decay S, then accumulate p[j] = Σ_i k[i] · S[i,j].
            float p_local = 0.0;
            for (uint i = 0u; i < 128u; i++) {
                uint off = state_base + i * 128u + j;
                float sij = state_data[off] * decay;
                state_data[off] = sij;
                p_local += sK[i] * sij;
            }
            sP[j] = p_local;
            barrier();

            // Compute d[j].
            float d_j = b_sc * (sV[j] - sP[j]);
            sD[j] = d_j;
            barrier();

            // Pass B: rank-1 update S[i,j] += k[i] · d[j], fused with readout o[j].
            float o_local = 0.0;
            for (uint i = 0u; i < 128u; i++) {
                uint off = state_base + i * 128u + j;
                float sij = state_data[off] + sK[i] * d_j;
                state_data[off] = sij;
                o_local += sQ[i] * sij;
            }

            // Scale by 1/sqrt(d), d=128.
            o_local *= inversesqrt(128.0);

            // RMSNorm: scale = rsqrt(sumSq/d + eps), then o = o * scale * normWeight.
            sRed[j] = o_local * o_local;
            barrier();
            [[unroll]] for (uint s = 64u; s > 0u; s >>= 1) {
                if (j < s) sRed[j] += sRed[j + s];
                barrier();
            }
            float scale = inversesqrt(sRed[0] / 128.0 + norm_eps);

            float o_normed = o_local * scale * sNormW[j];

            // SiLU(z) gate.
            float zv = sZ[j];
            float silu = zv / (1.0 + exp(-zv));

            o_data[hd_off + j] = o_normed * silu;
        }
        """;

    // ════════════════════════════════════════════════════════════════════════════
    //  Issue #356 PR5a: batched + fused-scan GDN shaders for the one-dispatch-per-
    //  stage batched PREFILL trunk (PR5b consumes them). Each is BYTE-IDENTICAL to
    //  N sequential single-token GDN calls (same per-row / per-position arithmetic
    //  and reduction order as the PR1/PR2 single-token shaders above); only the
    //  per-token host dispatch overhead is removed. Mirror the CUDA #114-B / #290
    //  kernels (llm_gdn_*_batched / llm_gdn_recurrence_scan) op-for-op.
    // ════════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Batched GDN depthwise causal conv1d over a chunk of <c>n_tok</c> tokens (read-only
    /// state). Bit-identical to <c>n_tok</c> sequential <see cref="GdnConv1dDecode"/> calls.
    /// Each (channel, token) invocation computes <c>output[i,c]</c> reading the chunk inputs
    /// <c>x[n_tok, channels]</c> plus the carried pre-chunk <c>state[(K-1), channels]</c>
    /// (oldest-first); state is NOT mutated here (the advance is a separate dispatch, so all
    /// concurrent token groups read one snapshot). Sum order matches the single-token shader:
    /// current tap weight[K-1] first, then taps k=0..K-2 oldest→newest. Mirrors CUDA
    /// <c>llm_gdn_conv1d_decode_batched</c>.
    /// Push constants: { uint channels, uint kernel_size, uint n_tok }.
    /// Bindings: 0=x (in), 1=state (in, read-only), 2=weight (in), 3=output (out).
    /// Dispatch: (ceil(channels/256), n_tok) workgroups of 256 threads.
    /// </summary>
    internal const string GdnConv1dDecodeBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer X     { float x_data[]; };
        layout(binding = 1) readonly  buffer State { float state_data[]; };
        layout(binding = 2) readonly  buffer W     { float w_data[]; };
        layout(binding = 3) writeonly buffer O     { float o_data[]; };

        layout(push_constant) uniform Params {
            uint channels;
            uint kernel_size;
            uint n_tok;
        };

        void main() {
            uint c = gl_GlobalInvocationID.x;
            uint i = gl_WorkGroupID.y;
            if (c >= channels || i >= n_tok) return;

            int retained = int(kernel_size) - 1;
            float x_c = x_data[i * channels + c];
            float sum = w_data[uint(retained) * channels + c] * x_c;
            for (int k = 0; k < retained; k++) {
                int p = int(i) - retained + k;   // chunk-relative position of tap k
                float val = (p >= 0)
                    ? x_data[uint(p) * channels + c]
                    : state_data[uint(p + retained) * channels + c];
                sum += w_data[uint(k) * channels + c] * val;
            }
            o_data[i * channels + c] = sum;
        }
        """;

    /// <summary>
    /// Advance the GDN conv1d state past a chunk of <c>n_tok</c> tokens (matches the sequential
    /// state evolution after <c>n_tok</c> <see cref="GdnConv1dDecode"/> calls). Reproduces the
    /// retained-window exactly: <c>new_state[r,c]</c> is the chunk input at position
    /// <c>(n_tok-(K-1)+r)</c>, or the carried old state when that index is still before the chunk.
    /// All sources are read into a <c>float tmp[4]</c> (K-1 ≤ 4) BEFORE any write to tolerate the
    /// in-place aliasing of the small-N case. Mirrors CUDA <c>llm_gdn_conv1d_state_update_batched</c>.
    /// Push constants: { uint channels, uint kernel_size, uint n_tok }.
    /// Bindings: 0=x (in), 1=state (in/out).
    /// Dispatch: ceil(channels/256) workgroups of 256 threads.
    /// </summary>
    internal const string GdnConv1dStateUpdateBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer X     { float x_data[]; };
        layout(binding = 1)          buffer State { float state_data[]; };

        layout(push_constant) uniform Params {
            uint channels;
            uint kernel_size;
            uint n_tok;
        };

        void main() {
            uint c = gl_GlobalInvocationID.x;
            if (c >= channels) return;

            int retained = int(kernel_size) - 1;
            float tmp[4];   // K-1 <= 4 for our models
            for (int r = 0; r < retained; r++) {
                int p = int(n_tok) - retained + r;
                tmp[r] = (p >= 0)
                    ? x_data[uint(p) * channels + c]
                    : state_data[uint(p + retained) * channels + c];
            }
            for (int r = 0; r < retained; r++)
                state_data[uint(r) * channels + c] = tmp[r];
        }
        """;

    /// <summary>
    /// #357 PR1: capture every batched-verify ring slot's GDN conv1d state in ONE launch. Slot
    /// <c>i</c> (i ∈ [0, <c>n_capture</c>)) receives the conv state the sequential decode loop would
    /// hold AFTER token <c>i</c> — byte-identical to <see cref="GdnConv1dStateUpdateBatched"/> with
    /// <c>n_tok = i+1</c>. Reads the PRE-update <c>state</c> (the caller runs this BEFORE advancing
    /// the live conv state). Per slot the retained window is <c>[i+1-(K-1) .. i]</c>, drawing from the
    /// carried pre-chunk <c>state</c> for the early-token (p &lt; 0) padding. <c>ring_float_offset</c>
    /// offsets to this layer's region in slot 0; <c>ring_slot_stride</c> is the float stride between
    /// consecutive slots. No barriers (pure per-(c,slot) writes). Mirrors CUDA
    /// <c>llm_gdn_conv1d_state_capture_ring</c> / <c>CudaBackend.GdnConv1dStateCaptureRing</c>.
    /// Push constants: { uint channels, uint kernel_size, uint ring_slot_stride, uint ring_float_offset, uint n_capture }.
    /// Bindings: 0=x (in), 1=state (in, read-only), 2=ring (out).
    /// Dispatch: (ceil(channels/256), n_capture) workgroups of 256 threads.
    /// </summary>
    internal const string GdnConv1dStateCaptureRing = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer X     { float x_data[]; };
        layout(binding = 1) readonly  buffer State { float state_data[]; };
        layout(binding = 2) writeonly buffer Ring  { float ring_data[]; };

        layout(push_constant) uniform Params {
            uint channels;
            uint kernel_size;
            uint ring_slot_stride;
            uint ring_float_offset;
            uint n_capture;
        };

        void main() {
            uint c = gl_GlobalInvocationID.x;
            uint slot = gl_WorkGroupID.y;
            if (c >= channels || slot >= n_capture) return;

            int retained = int(kernel_size) - 1;
            int n_eff = int(slot) + 1;   // state after processing tokens [0, n_eff)
            for (int r = 0; r < retained; r++) {
                int p = n_eff - retained + r;
                float v = (p >= 0)
                    ? x_data[uint(p) * channels + c]
                    : state_data[uint(p + retained) * channels + c];
                ring_data[ring_float_offset + slot * ring_slot_stride + uint(r) * channels + c] = v;
            }
        }
        """;

    /// <summary>
    /// Batched GDN L2-norm per head over <c>n_tok</c> rows. One workgroup per (head, token),
    /// 256-thread tree reduction. The bound buffer is the data region; <c>offset</c> is the float
    /// base (host-offset to the Q or K region), <c>row_stride</c> the per-token element stride
    /// (= conv channels). Per (head, token) ggml L2: <c>scale = 1 / max(sqrt(Σ x²), eps)</c>.
    /// Bit-identical to <c>n_tok</c> sequential <see cref="GdnL2NormPerHead"/> calls. Mirrors CUDA
    /// <c>llm_gdn_l2_norm_per_head_batched</c>.
    /// Push constants: { uint head_dim, uint num_heads, float eps, uint offset, uint row_stride, uint n_tok }.
    /// Bindings: 0=data (in/out).
    /// Dispatch: (num_heads, n_tok) workgroups of 256 threads.
    /// </summary>
    internal const string GdnL2NormPerHeadBatched = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data_buf[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
            uint offset;
            uint row_stride;
            uint n_tok;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            uint i = gl_WorkGroupID.y;
            if (head >= num_heads || i >= n_tok) return;

            uint base_off = offset + i * row_stride + head * head_dim;

            float sum = 0.0;
            for (uint e = tid; e < head_dim; e += 256u) {
                float v = data_buf[base_off + e];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            [[unroll]] for (uint s = 128u; s > 0u; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            float norm = sqrt(sdata[0]);
            float divisor = norm > eps ? norm : eps;
            float inv = 1.0 / divisor;
            for (uint e = tid; e < head_dim; e += 256u)
                data_buf[base_off + e] = data_buf[base_off + e] * inv;
        }
        """;

    /// <summary>
    /// Batched GDN tile-heads (GQA-style broadcast) over <c>n_tok</c> rows. One thread per dst
    /// element: <c>dst[i,idx] = src[i*src_stride + (idx/head_dim % src_heads)*head_dim + idx%head_dim]</c>
    /// (tile, NOT torch repeat_interleave). <c>src_offset</c>/<c>dst_offset</c> are float base
    /// offsets (host-offset to Q or K region); <c>src_stride</c>/<c>dst_stride</c> are per-token
    /// strides (= conv channels / value_dim). Bit-identical to <c>n_tok</c> sequential
    /// <see cref="GdnTileHeads"/> calls. Mirrors CUDA <c>llm_gdn_tile_heads_batched</c>.
    /// Push constants: { uint src_heads, uint repeat, uint head_dim, uint src_offset, uint dst_offset, uint src_stride, uint dst_stride, uint n_tok }.
    /// Bindings: 0=src (in), 1=dst (out).
    /// Dispatch: (ceil(src_heads*repeat*head_dim/256), n_tok) workgroups of 256 threads.
    /// </summary>
    internal const string GdnTileHeadsBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Src { float src_data[]; };
        layout(binding = 1) writeonly buffer Dst { float dst_data[]; };

        layout(push_constant) uniform Params {
            uint src_heads;
            uint repeat;
            uint head_dim;
            uint src_offset;
            uint dst_offset;
            uint src_stride;
            uint dst_stride;
            uint n_tok;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            uint i = gl_WorkGroupID.y;
            uint total = src_heads * repeat * head_dim;
            if (idx >= total || i >= n_tok) return;
            uint j = idx % head_dim;
            uint h_dst = idx / head_dim;
            uint h_src = h_dst % src_heads;
            dst_data[dst_offset + i * dst_stride + idx] =
                src_data[src_offset + i * src_stride + h_src * head_dim + j];
        }
        """;

    /// <summary>
    /// Fused sequential GDN recurrence scan over a chunk of <c>n_tok</c> tokens (issue #356 PR5a).
    /// ONE workgroup per v-head, <c>local_size_x = headDim</c> (HARDCODED 128 — same specialization
    /// as <see cref="GdnRecurrenceDecode"/>); each thread owns output column <c>j</c>. The workgroup
    /// loops the <c>n_tok</c> positions INTERNALLY, running the EXACT passes of the single-token
    /// decode at each step (decay→p[j]→d[j]→rank-1+readout→1/√d→RMSNorm→SiLU(z)) and carrying the
    /// per-head <c>D×D</c> state in the global state buffer between steps; a trailing
    /// <c>barrier()</c> makes the position boundary clean before the next step reloads shared inputs.
    /// This is the bit-identical fused form of <c>n_tok</c> sequential
    /// <see cref="GdnRecurrenceDecode"/> launches — NOT the parallel chunked-scan (which reorders
    /// the FP reductions). Mirrors CUDA <c>llm_gdn_recurrence_scan</c> op-for-op.
    ///
    /// Per-head input strides let q/k come from the tiled <c>[n_tok, value_dim]</c> buffers
    /// (q_stride/k_stride; head h at <c>i*stride + h*d</c>), v straight from the silu'd conv output
    /// <c>[n_tok, conv_ch]</c> at <c>v_head_off + h*d</c> (v_stride), z from <c>[n_tok, value_dim]</c>
    /// (z_stride), alpha/beta from <c>[n_tok, num_v_heads]</c>, output to <c>[n_tok, value_dim]</c>
    /// (o_stride).
    ///
    /// #290/#357 ring capture: when <c>n_capture &gt; 0</c> and <c>i &lt; n_capture</c>, each
    /// post-Pass-B state element is ALSO written into the ring buffer at
    /// <c>ring_scan_off + i*ring_slot_stride + state_base + ii*d + j</c> (the exact value the live
    /// state now holds → byte-identical to the device copy the per-position loop would issue). The
    /// ring binding is ALWAYS present; PR5a's prefill use and the unit test bind <c>state</c> as a
    /// placeholder and pass <c>n_capture = 0</c> so nothing is written (every ring store is guarded
    /// by <c>n_capture &gt; 0 &amp;&amp; i &lt; n_capture</c>). The scan arithmetic and the live
    /// state evolution are byte-unchanged regardless of capture.
    /// Push constants: { uint hv, uint d, float norm_eps, uint q_stride, uint k_stride, uint v_stride,
    ///   uint v_head_off, uint z_stride, uint o_stride, uint n_tok, uint ring_slot_stride,
    ///   uint n_capture, uint ring_scan_off }.
    /// Bindings: 0=state (in/out), 1=q, 2=k, 3=v, 4=alpha_in, 5=beta, 6=ssm_a, 7=dt_bias,
    ///   8=norm_weight, 9=z (all readonly), 10=output (writeonly), 11=ring (writeonly).
    /// Dispatch: hv workgroups of 128 threads (groupY=1; loops n_tok internally).
    /// </summary>
    internal const string GdnRecurrenceScan = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 128) in;

        layout(binding = 0)            buffer State  { float state_data[]; };
        layout(binding = 1) readonly   buffer Q      { float q_data[]; };
        layout(binding = 2) readonly   buffer K      { float k_data[]; };
        layout(binding = 3) readonly   buffer V      { float v_data[]; };
        layout(binding = 4) readonly   buffer AlphaIn{ float alpha_data[]; };
        layout(binding = 5) readonly   buffer Beta   { float beta_data[]; };
        layout(binding = 6) readonly   buffer SsmA   { float ssma_data[]; };
        layout(binding = 7) readonly   buffer DtBias { float dtbias_data[]; };
        layout(binding = 8) readonly   buffer NormW  { float normw_data[]; };
        layout(binding = 9) readonly   buffer Z      { float z_data[]; };
        layout(binding = 10) writeonly buffer O      { float o_data[]; };
        layout(binding = 11) writeonly buffer Ring   { float ring_data[]; };

        layout(push_constant) uniform Params {
            uint hv;
            uint d;
            float norm_eps;
            uint q_stride;
            uint k_stride;
            uint v_stride;
            uint v_head_off;
            uint z_stride;
            uint o_stride;
            uint n_tok;
            uint ring_slot_stride;
            uint n_capture;
            uint ring_scan_off;
        };

        shared float sK[128];
        shared float sQ[128];
        shared float sV[128];
        shared float sZ[128];
        shared float sNormW[128];
        shared float sP[128];
        shared float sD[128];
        shared float sRed[128];

        void main() {
            uint h = gl_WorkGroupID.x;
            uint j = gl_LocalInvocationID.x;

            sNormW[j] = normw_data[j];          // layer-constant; each thread reads own j
            // d is fixed at 128 (== local_size_x == headDim, enforced by the wrapper). Using the
            // literal lets the SPIR-V compiler strength-reduce i*128 → i<<7 and unroll the loops.
            uint state_base = h * 16384u;       // h * d * d

            for (uint i = 0u; i < n_tok; i++) {
                uint qoff = i * q_stride + h * d;
                uint koff = i * k_stride + h * d;
                uint voff = i * v_stride + v_head_off + h * d;
                uint zoff = i * z_stride + h * d;
                sK[j] = k_data[koff + j];
                sQ[j] = q_data[qoff + j];
                sV[j] = v_data[voff + j];
                sZ[j] = z_data[zoff + j];
                barrier();

                float alpha_x = alpha_data[i * hv + h] + dtbias_data[h];
                float dt      = alpha_x >= 20.0 ? alpha_x : log(1.0 + exp(alpha_x));   // softplus
                float decay   = exp(dt * ssma_data[h]);
                float b_sc    = 1.0 / (1.0 + exp(-beta_data[i * hv + h]));

                // Pass A: decay S, accumulate p[j] = Σ_i k[i]·S[i,j].
                float p_local = 0.0;
                for (uint ii = 0u; ii < 128u; ii++) {
                    uint off = state_base + ii * 128u + j;
                    float sij = state_data[off] * decay;
                    state_data[off] = sij;
                    p_local += sK[ii] * sij;
                }
                sP[j] = p_local;
                barrier();

                float d_j = b_sc * (sV[j] - sP[j]);
                sD[j] = d_j;
                barrier();

                // Pass B: rank-1 update S[i,j] += k[i]·d[j], fused with readout o[j].
                // #290 ring capture: mirror each post-update element into ring slot i (same value
                // the live state now holds → byte-identical to the per-position device copy).
                bool capture = n_capture > 0u && i < n_capture;
                uint ring_i = ring_scan_off + i * ring_slot_stride + state_base;
                float o_local = 0.0;
                for (uint ii = 0u; ii < 128u; ii++) {
                    uint off = state_base + ii * 128u + j;
                    float sij = state_data[off] + sK[ii] * d_j;
                    state_data[off] = sij;
                    if (capture) ring_data[ring_i + ii * 128u + j] = sij;
                    o_local += sQ[ii] * sij;
                }
                o_local *= inversesqrt(128.0);

                sRed[j] = o_local * o_local;
                barrier();
                [[unroll]] for (uint s = 64u; s > 0u; s >>= 1) {
                    if (j < s) sRed[j] += sRed[j + s];
                    barrier();
                }
                float scale = inversesqrt(sRed[0] / 128.0 + norm_eps);
                float o_normed = o_local * scale * sNormW[j];

                float zv = sZ[j];
                float silu = zv / (1.0 + exp(-zv));
                o_data[i * o_stride + h * d + j] = o_normed * silu;
                barrier();                       // position boundary: next step reloads shared
            }
        }
        """;

    /// <summary>
    /// Chunk-parallel ("FlashQLA-style chunk_gated_delta_rule") GDN prefill scan over
    /// <c>n_tok</c> tokens (issue #356 PR5c). ONE workgroup per v-head, <c>local_size_x = headDim</c>
    /// (HARDCODED 128 — same specialization as <see cref="GdnRecurrenceScan"/>); thread <c>j</c> owns
    /// value-column <c>j</c> of the state, projections, pseudo-values and output. The workgroup walks
    /// the chunk grid (<c>GDN_CHUNK = 64</c> tokens per block) and resolves each block's intra-chunk
    /// delta-rule coupling by forward substitution over a fixed tile, rather than the per-token scan of
    /// <see cref="GdnRecurrenceScan"/>. Mirrors CUDA <c>llm_gdn_chunked_prefill</c> /
    /// CPU <c>GdnKernels.GdnRecurrenceChunkedPrefill</c> op-for-op.
    ///
    /// Numerically EQUAL to the sequential scan up to floating-point reduction order: the chunked form
    /// reorders the FP reductions, so it is argmax-stable but NOT byte-exact against
    /// <see cref="GdnRecurrenceScan"/> (the same parity class as the CUDA/CPU chunked path).
    ///
    /// Same per-head input strides as <see cref="GdnRecurrenceScan"/>: q/k from the tiled
    /// <c>[n_tok, value_dim]</c> buffers (q_stride/k_stride; head h at <c>i*stride + h*d</c>), v from
    /// the silu'd conv output <c>[n_tok, conv_ch]</c> at <c>v_head_off + h*d</c> (v_stride), z from
    /// <c>[n_tok, value_dim]</c> (z_stride), alpha/beta from <c>[n_tok, num_v_heads]</c>, output to
    /// <c>[n_tok, value_dim]</c> (o_stride). NO ring binding — chunked is clean-prefill only.
    ///
    /// Shared memory ≈ 34,560 bytes (<c>sNormW[128] + sCum/sG/sB[64] + sKK/sKQ[64*64] + sRed[128]</c>),
    /// over the 32 KB some GPUs guarantee — the wrapper gates on
    /// <c>VulkanBackend.SupportsGdnChunkedPrefill</c> and falls back to the scan otherwise.
    /// Push constants: { uint hv, uint d, float norm_eps, uint q_stride, uint k_stride, uint v_stride,
    ///   uint v_head_off, uint z_stride, uint o_stride, uint n_tok }.
    /// Bindings: 0=state (in/out), 1=q, 2=k, 3=v, 4=alpha_in, 5=beta, 6=ssm_a, 7=dt_bias,
    ///   8=norm_weight, 9=z (all readonly), 10=output (writeonly).
    /// Dispatch: hv workgroups of 128 threads (groupY=1; loops the chunk grid internally).
    /// </summary>
    internal const string GdnChunkedPrefill = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 128) in;

        layout(binding = 0)            buffer State  { float state_data[]; };
        layout(binding = 1) readonly   buffer Q      { float q_data[]; };
        layout(binding = 2) readonly   buffer K      { float k_data[]; };
        layout(binding = 3) readonly   buffer V      { float v_data[]; };
        layout(binding = 4) readonly   buffer AlphaIn{ float alpha_data[]; };
        layout(binding = 5) readonly   buffer Beta   { float beta_data[]; };
        layout(binding = 6) readonly   buffer SsmA   { float ssma_data[]; };
        layout(binding = 7) readonly   buffer DtBias { float dtbias_data[]; };
        layout(binding = 8) readonly   buffer NormW  { float normw_data[]; };
        layout(binding = 9) readonly   buffer Z      { float z_data[]; };
        layout(binding = 10) writeonly buffer O      { float o_data[]; };

        layout(push_constant) uniform Params {
            uint hv;
            uint d;
            float norm_eps;
            uint q_stride;
            uint k_stride;
            uint v_stride;
            uint v_head_off;
            uint z_stride;
            uint o_stride;
            uint n_tok;
        };

        // GDN_CHUNK = 64 (compile-time). d is fixed at 128 (== local_size_x == headDim).
        shared float sNormW[128];
        shared float sCum[64];          // cumulative log-decay (sequential)
        shared float sG[64];            // exp(cum_t)
        shared float sB[64];            // sigmoid(beta_t)
        shared float sKK[64 * 64];      // K_s·K_t  (lower triangle s<=t)
        shared float sKQ[64 * 64];      // K_s·Q_t  (lower triangle s<=t)
        shared float sRed[128];         // RMSNorm reduction

        void main() {
            uint h = gl_WorkGroupID.x;
            uint j = gl_LocalInvocationID.x;

            sNormW[j] = normw_data[j];          // layer-constant; each thread reads own j
            uint state_base = h * 16384u;       // h * d * d
            float inv_sqrt_d = inversesqrt(128.0);

            float projK[64];
            float projQ[64];
            float u[64];

            for (uint c0 = 0u; c0 < n_tok; c0 += 64u) {
                uint cN = n_tok - c0; if (cN > 64u) cN = 64u;

                // Per-token scalars: cumulative log-decay is sequential → thread 0 fills shared.
                if (j == 0u) {
                    float run = 0.0;
                    for (uint t = 0u; t < cN; t++) {
                        float ax = alpha_data[(c0 + t) * hv + h] + dtbias_data[h];
                        float dt = ax >= 20.0 ? ax : log(1.0 + exp(ax));     // softplus
                        run += dt * ssma_data[h];
                        sCum[t] = run;
                        sG[t]   = exp(run);
                        sB[t]   = 1.0 / (1.0 + exp(-beta_data[(c0 + t) * hv + h]));
                    }
                }
                barrier();

                // K·K and K·Q dot matrices (lower triangle s<=t); shared across all columns j.
                for (uint idx = j; idx < cN * cN; idx += 128u) {
                    uint t = idx / cN;
                    uint s = idx - t * cN;
                    if (s <= t) {
                        uint ks = (c0 + s) * k_stride + h * d;
                        uint kt = (c0 + t) * k_stride + h * d;
                        uint qt = (c0 + t) * q_stride + h * d;
                        float kk = 0.0, kq = 0.0;
                        for (uint i = 0u; i < 128u; i++) {
                            float ksi = k_data[ks + i];
                            kk += ksi * k_data[kt + i];
                            kq += ksi * q_data[qt + i];
                        }
                        sKK[t * 64u + s] = kk;
                        sKQ[t * 64u + s] = kq;
                    }
                }
                barrier();

                // S0 projections (column j): projK[t]=Σ_i K_t[i]·S0[i,j], projQ[t]=Σ_i Q_t[i]·S0[i,j].
                for (uint t = 0u; t < cN; t++) {
                    uint kt = (c0 + t) * k_stride + h * d;
                    uint qt = (c0 + t) * q_stride + h * d;
                    float pk = 0.0, pq = 0.0;
                    for (uint i = 0u; i < 128u; i++) {
                        float sij = state_data[state_base + i * 128u + j];
                        pk += k_data[kt + i] * sij;
                        pq += q_data[qt + i] * sij;
                    }
                    projK[t] = pk;
                    projQ[t] = pq;
                }

                // Forward substitution: u_t = b_t(v_t − g_t·projK_t) − Σ_{s<t} A[t,s] u_s.
                for (uint t = 0u; t < cN; t++) {
                    uint vt = (c0 + t) * v_stride + v_head_off + h * d;
                    float bt = sB[t];
                    float uj = bt * (v_data[vt + j] - sG[t] * projK[t]);
                    for (uint s = 0u; s < t; s++) {
                        float a = bt * exp(sCum[t] - sCum[s]) * sKK[t * 64u + s];
                        uj -= a * u[s];
                    }
                    u[t] = uj;
                }

                // Output + per-head RMSNorm + SiLU(z) gate.
                for (uint t = 0u; t < cN; t++) {
                    float o = sG[t] * projQ[t];
                    for (uint s = 0u; s <= t; s++)
                        o += exp(sCum[t] - sCum[s]) * sKQ[t * 64u + s] * u[s];
                    o *= inv_sqrt_d;

                    sRed[j] = o * o;
                    barrier();
                    [[unroll]] for (uint red = 64u; red > 0u; red >>= 1) {
                        if (j < red) sRed[j] += sRed[j + red];
                        barrier();
                    }
                    float scale = inversesqrt(sRed[0] / 128.0 + norm_eps);
                    float on = o * scale * sNormW[j];
                    float zv = z_data[(c0 + t) * z_stride + h * d + j];
                    float silu = zv / (1.0 + exp(-zv));
                    o_data[(c0 + t) * o_stride + h * d + j] = on * silu;
                    barrier();
                }

                // State carry: S[i,j] = g_{cN-1}·S[i,j] + Σ_s exp(cum_{cN-1}−cum_s)·K_s[i]·u_s.
                float gLast = sG[cN - 1u];
                float cumLast = sCum[cN - 1u];
                for (uint i = 0u; i < 128u; i++) {
                    uint off = state_base + i * 128u + j;
                    float acc = gLast * state_data[off];
                    for (uint s = 0u; s < cN; s++) {
                        uint ks = (c0 + s) * k_stride + h * d;
                        acc += exp(cumLast - sCum[s]) * k_data[ks + i] * u[s];
                    }
                    state_data[off] = acc;
                }
                barrier();   // chunk boundary: next chunk overwrites shared
            }
        }
        """;

    /// <summary>
    /// RoPE NEOX *partial* rotation: rotates only the first <c>rope_dim</c> of each
    /// <c>head_dim</c>-wide head and passes dims <c>[rope_dim, head_dim)</c> through untouched.
    /// qwen35moe/qwen36 Gated-Attention rotates only the first 64 of each 256-dim head; the
    /// frequency exponent uses <c>rope_dim</c> (NOT <c>head_dim</c>) — matching CUDA
    /// <c>llm_rope_neox_partial</c> and CPU <c>SimdKernels.ApplyRoPECachedNeoxPartial</c>.
    /// Pair layout: (i, i + rope_dim/2) for i ∈ [0, rope_dim/2). One thread per pair.
    /// Push constants: { uint num_heads, uint head_dim, uint rope_dim, int position, float theta }.
    /// Bindings: 0=x (in/out). Distinct from <see cref="RoPENeox"/>, which rotates the full head_dim.
    /// </summary>
    internal const string RoPENeoxPartial = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            uint rope_dim;
            int position;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint rope_half = rope_dim / 2u;
            uint total_pairs = num_heads * rope_half;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / rope_half;
            uint i = pair_idx % rope_half;

            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(rope_dim));
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint head_base = h * head_dim;
            uint a_idx = head_base + i;
            uint b_idx = head_base + i + rope_half;
            float x0 = x_data[a_idx];
            float x1 = x_data[b_idx];
            x_data[a_idx] = x0 * cos_a - x1 * sin_a;
            x_data[b_idx] = x0 * sin_a + x1 * cos_a;
            // Dims [rope_dim, head_dim) pass through untouched (never written).
        }
        """;

    /// <summary>
    /// Batched partial NEOX RoPE: the <see cref="RoPENeoxPartial"/> sibling of
    /// <see cref="RoPENeoxBatched"/>. Rotates only the first <c>rope_dim</c> of each
    /// <c>head_dim</c>-wide head over each of <c>n_tok</c> rows of a
    /// <c>[n_tok][num_heads*head_dim]</c> buffer in ONE dispatch; token t (row
    /// <c>gl_WorkGroupID.y</c>) uses position = <c>base_position + t</c>. The frequency exponent
    /// uses <c>rope_dim</c> (NOT <c>head_dim</c>) — matching CUDA <c>llm_rope_neox_partial_batched</c>
    /// and <see cref="RoPENeoxPartial"/>. Bit-identical to <c>n_tok</c> separate
    /// <see cref="RoPENeoxPartial"/> calls at positions base_position, base_position+1, ….
    /// Pair index = <c>gl_GlobalInvocationID.x</c>, token row t = <c>gl_WorkGroupID.y</c> (dispatch
    /// ceil(total_pairs/256) × n_tok groups). Dims <c>[rope_dim, head_dim)</c> pass through untouched.
    /// Push constants: { uint num_heads, uint head_dim, uint rope_dim, int position, float theta }
    /// (position carries base_position; n_tok comes from the dispatched Y group count).
    /// Bindings: 0=x (in/out).
    /// </summary>
    internal const string RoPENeoxPartialBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            uint rope_dim;
            int position;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint token    = gl_WorkGroupID.y;
            uint rope_half = rope_dim / 2u;
            uint total_pairs = num_heads * rope_half;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / rope_half;
            uint i = pair_idx % rope_half;

            int pos = position + int(token);
            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(rope_dim));
            float angle = float(pos) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint head_base = token * num_heads * head_dim + h * head_dim;
            uint a_idx = head_base + i;
            uint b_idx = head_base + i + rope_half;
            float x0 = x_data[a_idx];
            float x1 = x_data[b_idx];
            x_data[a_idx] = x0 * cos_a - x1 * sin_a;
            x_data[b_idx] = x0 * sin_a + x1 * cos_a;
            // Dims [rope_dim, head_dim) pass through untouched (never written).
        }
        """;

    /// <summary>
    /// Strided de-interleave of qwen35moe's Gated-Attention <c>[Q‖G]</c> output. The input
    /// <paramref name="qg"/> is laid out per head as <c>[Q[head_dim] ‖ G[head_dim]]</c>
    /// (stride <c>2*head_dim</c> per head); this splits it into contiguous
    /// <c>q[num_heads*head_dim]</c> and <c>g[num_heads*head_dim]</c>. Mirrors CUDA
    /// <c>llm_split_qg</c> op-for-op. One thread per (h, j).
    /// Push constants: { uint num_heads, uint head_dim }.
    /// Bindings: 0=qg (in), 1=q (out), 2=g (out).
    /// </summary>
    internal const string SplitQG = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer QG { float qg_data[]; };
        layout(binding = 1) writeonly buffer Q  { float q_data[]; };
        layout(binding = 2) writeonly buffer G  { float g_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            uint total = num_heads * head_dim;
            if (idx >= total) return;

            uint h = idx / head_dim;
            uint j = idx % head_dim;
            uint src_base = h * head_dim * 2u;
            // dst index == idx (h*head_dim + j); only the src stride needs the 2x.
            q_data[idx] = qg_data[src_base + j];
            g_data[idx] = qg_data[src_base + head_dim + j];
        }
        """;

    /// <summary>
    /// Batched strided de-interleave: applies <see cref="SplitQG"/> to each of <c>n_tok</c> rows of
    /// the <c>[n_tok][num_heads*head_dim*2]</c> input in ONE dispatch (the <see cref="SplitQG"/>
    /// sibling of the batched ops). The <paramref name="qg"/> row stride is
    /// <c>num_heads*head_dim*2</c>; the q/g row stride is <c>num_heads*head_dim</c>. Token index
    /// t = <c>gl_WorkGroupID.y</c>, (h, j) index = <c>gl_GlobalInvocationID.x</c>. Bit-identical to
    /// <c>n_tok</c> separate <see cref="SplitQG"/> calls. Mirrors CUDA <c>llm_split_qg_batched</c>.
    /// Push constants: { uint num_heads, uint head_dim } (n_tok from the dispatched Y group count).
    /// Bindings: 0=qg (in), 1=q (out), 2=g (out).
    /// </summary>
    internal const string SplitQGBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer QG { float qg_data[]; };
        layout(binding = 1) writeonly buffer Q  { float q_data[]; };
        layout(binding = 2) writeonly buffer G  { float g_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            uint token = gl_WorkGroupID.y;
            uint total = num_heads * head_dim;
            if (idx >= total) return;

            uint h = idx / head_dim;
            uint j = idx % head_dim;
            uint qg_base  = token * num_heads * head_dim * 2u + h * head_dim * 2u;
            uint out_base = token * total + h * head_dim;
            q_data[out_base + j] = qg_data[qg_base + j];
            g_data[out_base + j] = qg_data[qg_base + head_dim + j];
        }
        """;

    /// <summary>
    /// Fused in-place sigmoid-gate: <c>x[i] *= 1/(1+exp(-gate[i]))</c>. Replaces a Sigmoid +
    /// ElementwiseMul pair for the qwen35moe Gated-Attention output gate. Mirrors CUDA
    /// <c>llm_sigmoid_mul_inplace</c>. One thread per element.
    /// Push constants: { uint n }.
    /// Bindings: 0=x (in/out), 1=gate (in).
    /// </summary>
    internal const string SigmoidMulInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };
        layout(binding = 1) readonly buffer Gate { float gate_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            x_data[i] *= 1.0 / (1.0 + exp(-gate_data[i]));
        }
        """;

    /// <summary>
    /// Per-head RMSNorm: applies RMSNorm independently to each head-sized chunk.
    /// data[h*head_dim + i] = data[h*head_dim + i] / rms_h * weight[i]
    /// where rms_h = sqrt(mean(data[h*head_dim .. (h+1)*head_dim]^2) + eps).
    ///
    /// One workgroup per head. Weight is [head_dim] shared across all heads.
    /// Push constants: { uint head_dim, uint num_heads, float eps }.
    /// Bindings: 0=data (in/out), 1=weight (in).
    /// Dispatch: num_heads workgroups of 256 threads.
    /// </summary>
    internal const string HeadNorm = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data   { float data_buf[]; };
        layout(binding = 1) readonly buffer Weight { float weight_data[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
            // 0 = weight is shared across heads (Qwen3 style, len = head_dim).
            // head_dim = per-channel weight (OLMoE style, len = num_heads*head_dim).
            uint weight_stride;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            if (head >= num_heads) return;

            uint base_off = head * head_dim;
            uint w_off    = head * weight_stride;

            // Phase 1: accumulate sum of squares
            float sum = 0.0;
            for (uint i = tid; i < head_dim; i += 256) {
                float v = data_buf[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            // Phase 2: parallel reduction
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            // Phase 3: normalize in-place with weight
            float scale = inversesqrt(sdata[0] / float(head_dim) + eps);
            for (uint i = tid; i < head_dim; i += 256) {
                data_buf[base_off + i] = data_buf[base_off + i] * scale * weight_data[w_off + i];
            }
        }
        """;

    /// <summary>
    /// Batched per-head RMSNorm: applies <see cref="HeadNorm"/> independently to each head of
    /// each of <c>num_tokens</c> rows in a <c>[num_tokens][num_heads*head_dim]</c> buffer, in a
    /// single dispatch. Processes <c>num_tokens * num_heads</c> head-groups: head index
    /// h = <c>gl_WorkGroupID.x</c>, token row r = <c>gl_WorkGroupID.y</c> (dispatch
    /// num_heads × num_tokens groups). The weight (shared <c>[head_dim]</c> for Qwen3, or
    /// per-channel <c>[num_heads*head_dim]</c> for OLMoE via weight_stride) is shared across rows.
    /// Bit-identical to <c>num_tokens</c> separate <see cref="HeadNorm"/> calls.
    /// Push constants: { uint head_dim, uint num_heads, float eps, uint weight_stride, uint num_tokens }.
    /// Bindings: 0=data ([num_tokens][num_heads*head_dim], in/out), 1=weight (in).
    /// </summary>
    internal const string HeadNormBatched = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data   { float data_buf[]; };
        layout(binding = 1) readonly buffer Weight { float weight_data[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
            // 0 = weight shared across heads (Qwen3, len = head_dim).
            // head_dim = per-channel weight (OLMoE, len = num_heads*head_dim).
            uint weight_stride;
            uint num_tokens;
        };

        shared float sdata[256];

        void main() {
            uint tid  = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            uint row  = gl_WorkGroupID.y;
            if (head >= num_heads || row >= num_tokens) return;

            uint row_off  = row * num_heads * head_dim;
            uint base_off = row_off + head * head_dim;
            uint w_off    = head * weight_stride;

            // Phase 1: accumulate sum of squares for this token's head.
            float sum = 0.0;
            for (uint i = tid; i < head_dim; i += 256) {
                float v = data_buf[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            // Phase 2: parallel reduction.
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            // Phase 3: normalize in-place with weight.
            float scale = inversesqrt(sdata[0] / float(head_dim) + eps);
            for (uint i = tid; i < head_dim; i += 256) {
                data_buf[base_off + i] = data_buf[base_off + i] * scale * weight_data[w_off + i];
            }
        }
        """;

    /// <summary>
    /// Per-head RMS normalization without learned weights (L2 normalize).
    /// Used for Llama4TextL2Norm in QK-norm.
    /// Push constants: { uint head_dim, uint num_heads, float eps }.
    /// </summary>
    internal const string HeadNormPure = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data_buf[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            if (head >= num_heads) return;

            uint base_off = head * head_dim;

            float sum = 0.0;
            for (uint i = tid; i < head_dim; i += 256) {
                float v = data_buf[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            float scale = inversesqrt(sdata[0] / float(head_dim) + eps);
            for (uint i = tid; i < head_dim; i += 256) {
                data_buf[base_off + i] = data_buf[base_off + i] * scale;
            }
        }
        """;

    /// <summary>
    /// Element-wise multiply: output[i] = a[i] * b[i]
    /// Push constants: { uint n }.
    /// Bindings: 0=a, 1=b, 2=output.
    /// </summary>
    internal const string ElementwiseMul = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer BufA { float a_data[]; };
        layout(binding = 1) readonly buffer BufB { float b_data[]; };
        layout(binding = 2) writeonly buffer BufC { float c_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            c_data[i] = a_data[i] * b_data[i];
        }
        """;

    /// <summary>
    /// RoPE: interleaved pair rotation. Used by LLaMA, Mistral, SmolLM, etc.
    /// Push constants: { uint num_heads, uint head_dim, int position, float theta }.
    /// Bindings: 0=x (in/out).
    /// Each thread handles one pair (2 elements).
    /// </summary>
    internal const string RoPE = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            int position;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint half_dim = head_dim / 2;
            uint total_pairs = num_heads * half_dim;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / half_dim;
            uint i = pair_idx % half_dim;

            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint base_idx = h * head_dim + 2 * i;
            float x0 = x_data[base_idx];
            float x1 = x_data[base_idx + 1];
            x_data[base_idx]     = x0 * cos_a - x1 * sin_a;
            x_data[base_idx + 1] = x0 * sin_a + x1 * cos_a;
        }
        """;

    /// <summary>
    /// RoPE: NEOX/half rotation (pairs offset by head_dim/2). Used by Qwen, Phi, Gemma, Falcon, etc.
    /// </summary>
    internal const string RoPENeox = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            int position;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint half_dim = head_dim / 2;
            uint total_pairs = num_heads * half_dim;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / half_dim;
            uint i = pair_idx % half_dim;

            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint head_base = h * head_dim;
            uint a_idx = head_base + i;
            uint b_idx = head_base + i + half_dim;
            float x0 = x_data[a_idx];
            float x1 = x_data[b_idx];
            x_data[a_idx] = x0 * cos_a - x1 * sin_a;
            x_data[b_idx] = x0 * sin_a + x1 * cos_a;
        }
        """;

    /// <summary>
    /// Batched interleaved-pair RoPE: applies <see cref="RoPE"/> to each of <c>num_tokens</c>
    /// independent rows of a <c>[num_tokens][num_heads*head_dim]</c> buffer in one dispatch, where
    /// row r uses position = <c>base_pos + r</c> (per-token absolute position). Pair index in
    /// <c>gl_GlobalInvocationID.x</c>, token row r = <c>gl_WorkGroupID.y</c> (dispatch
    /// ceil(total_pairs/256) × num_tokens groups). Each row computes its own cos/sin from
    /// base_pos+r, so it is bit-identical to <c>num_tokens</c> separate <see cref="RoPE"/> calls
    /// with positions base_pos, base_pos+1, ….
    /// Push constants: { uint num_heads, uint head_dim, int base_pos, float theta }.
    /// Bindings: 0=x ([num_tokens][num_heads*head_dim], in/out).
    /// (num_tokens comes from the dispatched Y group count; no separate push-constant needed.)
    /// </summary>
    internal const string RoPEBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            int base_pos;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint row      = gl_WorkGroupID.y;
            uint half_dim = head_dim / 2;
            uint total_pairs = num_heads * half_dim;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / half_dim;
            uint i = pair_idx % half_dim;

            int position = base_pos + int(row);
            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint row_off = row * num_heads * head_dim;
            uint base_idx = row_off + h * head_dim + 2 * i;
            float x0 = x_data[base_idx];
            float x1 = x_data[base_idx + 1];
            x_data[base_idx]     = x0 * cos_a - x1 * sin_a;
            x_data[base_idx + 1] = x0 * sin_a + x1 * cos_a;
        }
        """;

    /// <summary>
    /// Batched NEOX/half-rotation RoPE: the <see cref="RoPENeox"/> sibling of
    /// <see cref="RoPEBatched"/>. Applies NEOX RoPE to each of <c>num_tokens</c> rows of a
    /// <c>[num_tokens][num_heads*head_dim]</c> buffer in one dispatch; row r uses position
    /// = <c>base_pos + r</c>. Bit-identical to <c>num_tokens</c> separate <see cref="RoPENeox"/>
    /// calls. Push constants: { uint num_heads, uint head_dim, int base_pos, float theta }.
    /// Bindings: 0=x (in/out). Pair index = <c>gl_GlobalInvocationID.x</c>, token row =
    /// <c>gl_WorkGroupID.y</c>.
    /// </summary>
    internal const string RoPENeoxBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            int base_pos;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint row      = gl_WorkGroupID.y;
            uint half_dim = head_dim / 2;
            uint total_pairs = num_heads * half_dim;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / half_dim;
            uint i = pair_idx % half_dim;

            int position = base_pos + int(row);
            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint row_off = row * num_heads * head_dim;
            uint head_base = row_off + h * head_dim;
            uint a_idx = head_base + i;
            uint b_idx = head_base + i + half_dim;
            float x0 = x_data[a_idx];
            float x1 = x_data[b_idx];
            x_data[a_idx] = x0 * cos_a - x1 * sin_a;
            x_data[b_idx] = x0 * sin_a + x1 * cos_a;
        }
        """;

    /// <summary>
    /// RoPE NEOX with per-half-dim freq_factors (Gemma 4 global / non-SWA layers). Identical to
    /// <see cref="RoPENeox"/> except each pair's frequency is divided by <c>freq_factors[i]</c>
    /// (binding 1, size head_dim/2), masking the high-frequency tail to ~identity for long
    /// context. Mirrors the CUDA <c>llm_rope_neox_with_factors</c> kernel and the CPU
    /// <c>SimdKernels.BuildRopeTable(..., globalFreqFactors)</c> path. llama.cpp gemma4.cpp:191
    /// applies this only to non-SWA layers; SWA layers use plain <see cref="RoPENeox"/>.
    /// Push constants: { uint num_heads, uint head_dim, int position, float theta }.
    /// </summary>
    internal const string RoPENeoxWithFactors = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };
        layout(binding = 1) readonly buffer FreqFactors { float freq_factors[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            int position;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint half_dim = head_dim / 2;
            uint total_pairs = num_heads * half_dim;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / half_dim;
            uint i = pair_idx % half_dim;

            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
            freq /= freq_factors[i];
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint head_base = h * head_dim;
            uint a_idx = head_base + i;
            uint b_idx = head_base + i + half_dim;
            float x0 = x_data[a_idx];
            float x1 = x_data[b_idx];
            x_data[a_idx] = x0 * cos_a - x1 * sin_a;
            x_data[b_idx] = x0 * sin_a + x1 * cos_a;
        }
        """;

    /// <summary>
    /// Softmax in-place (3-pass: max, exp+sum, normalize).
    /// Uses workgroup shared memory for reductions.
    /// Push constants: { uint n }.
    /// Bindings: 0=x (in/out).
    /// Dispatch: 1 workgroup of 256 threads.
    /// </summary>
    internal const string Softmax = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params { uint n; };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;

            // Pass 1: find max
            float local_max = -1.0/0.0; // -inf
            for (uint i = tid; i < n; i += 256)
                local_max = max(local_max, x_data[i]);
            sdata[tid] = local_max;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            // No extra barrier needed — last reduction iteration's barrier
            // guarantees sdata[0] is visible to all threads

            // Pass 2: exp(x - max) and sum
            float local_sum = 0.0;
            for (uint i = tid; i < n; i += 256) {
                float e = exp(x_data[i] - max_val);
                x_data[i] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float sum_val = sdata[0];

            // Pass 3: normalize
            float inv_sum = 1.0 / sum_val;
            for (uint i = tid; i < n; i += 256)
                x_data[i] *= inv_sum;
        }
        """;

    /// <summary>
    /// Element-wise sigmoid in-place: x[i] = 1 / (1 + exp(-x[i])).
    /// Used for Llama-4 MoE router gating.
    /// Push constants: { uint n }.
    /// Bindings: 0=x (in/out).
    /// </summary>
    internal const string Sigmoid = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            x_data[i] = 1.0 / (1.0 + exp(-x_data[i]));
        }
        """;

    /// <summary>
    /// Embedding lookup: copy one row from F32 embedding table to output.
    /// Push constants: { uint token_id, uint emb_dim }.
    /// Bindings: 0=embedding_table[vocab_size*emb_dim], 1=output[emb_dim].
    /// </summary>
    internal const string EmbedLookup = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer EmbTable { float emb_table[]; };
        layout(binding = 1) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint token_id;
            uint emb_dim;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= emb_dim) return;
            output_data[i] = emb_table[token_id * emb_dim + i];
        }
        """;

    /// <summary>
    /// Embedding lookup from Q4_K quantized table: dequantize one row to F32 output.
    /// 256 threads cooperate: each processes one block (256 elements) sequentially,
    /// with each thread handling one element per block.
    ///
    /// Push constants: { uint token_id, uint emb_dim }.
    /// Bindings: 0=quantized_table (uint8 via uint32[]), 1=output[emb_dim].
    /// Dispatch: 1 workgroup.
    /// </summary>
    internal const string EmbedLookupQ4K = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer EmbTable { uint emb_data[]; };
        layout(binding = 1) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint token_id;
            uint emb_dim;
        };

        shared uint blk[36]; // 144 bytes = one Q4_K block in shared memory

        uint sReadByte(uint byteOffset) {
            return (blk[byteOffset >> 2] >> ((byteOffset & 3) * 8)) & 0xFF;
        }

        float sReadHalf(uint byteOffset) {
            uint lo = sReadByte(byteOffset);
            uint hi = sReadByte(byteOffset + 1);
            return unpackHalf2x16(lo | (hi << 8)).x;
        }

        void sGetScaleMin(uint j, out float sc, out float m) {
            if (j < 4) {
                sc = float(sReadByte(4 + j) & 63);
                m  = float(sReadByte(4 + j + 4) & 63);
            } else {
                sc = float((sReadByte(4 + j + 4) & 0xF) | ((sReadByte(4 + j - 4) >> 6) << 4));
                m  = float((sReadByte(4 + j + 4) >> 4) | ((sReadByte(4 + j) >> 6) << 4));
            }
        }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint num_blocks = emb_dim >> 8; // emb_dim / 256

            // Byte offset to the start of this token's row
            uint bytes_per_row = num_blocks * 144;
            uint row_base = token_id * (bytes_per_row >> 2); // in uint32 units

            for (uint block = 0; block < num_blocks; block++) {
                // Cooperatively load 36 uint32s (144 bytes) into shared memory
                uint blk_word_base = row_base + (block * 144 >> 2);
                if (tid < 36)
                    blk[tid] = emb_data[blk_word_base + tid];
                barrier();

                // Each thread dequantizes its element
                uint chunk = tid >> 6;
                uint sub = tid & 63;
                bool is_upper = sub >= 32;
                uint byte_pos = sub & 31;

                float d = sReadHalf(0);
                float dmin = sReadHalf(2);

                uint si = chunk * 2 + (is_upper ? 1u : 0u);
                float sc, mn;
                sGetScaleMin(si, sc, mn);

                uint qbyte = sReadByte(16 + chunk * 32 + byte_pos);
                uint nibble = is_upper ? (qbyte >> 4) : (qbyte & 0xF);

                output_data[block * 256 + tid] = d * sc * float(nibble) - dmin * mn;

                barrier();
            }
        }
        """;

    /// <summary>
    /// Embedding lookup from a Q6_K quantized table: dequantize one row to F32 output.
    /// Mirrors the CUDA <c>llm_embed_lookup_q6k</c> kernel (and thus <see cref="MatVecQ6K"/>
    /// and the CPU <c>DequantQ6K</c>) — keeps a large Q6_K tied embedding (e.g. Gemma 4 12B,
    /// [3840, 262144] ≈ 787 MiB raw) off the F32 dequant path that would burn ~4 GB of VRAM.
    ///
    /// 256 threads cooperate: each processes one block (256 elements) sequentially, thread
    /// <c>tid</c> emitting element <c>tid</c> of each 256-element super-block.
    /// Q6_K block (210 bytes per 256 elements):
    ///   [0:128]   ql — lower 4 bits
    ///   [128:192] qh — upper 2 bits
    ///   [192:208] 16 int8 scales
    ///   [208:210] FP16 d (super-block scale)
    ///
    /// Push constants: { uint token_id, uint emb_dim }.
    /// Bindings: 0=quantized_table (uint8 via uint32[]), 1=output[emb_dim].
    /// Dispatch: 1 workgroup.
    /// </summary>
    internal const string EmbedLookupQ6K = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer EmbTable { uint emb_data[]; };
        layout(binding = 1) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint token_id;
            uint emb_dim;
        };

        // Read directly from global memory with absolute byte offsets (no shared
        // memory): Q6_K blocks are 210 bytes, so a block's start is not necessarily
        // 4-byte aligned, which would break a uint32-indexed shared-memory copy.
        // Same byte-addressing approach as MatVecQ6K's gByte.
        uint gByte(uint b) { return (emb_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        int  gInt8(uint b) { int v = int(gByte(b)); return v >= 128 ? v - 256 : v; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint num_blocks = emb_dim >> 8; // emb_dim / 256

            // Byte offset to the start of this token's row (210 bytes/block).
            uint bytes_per_row = num_blocks * 210;
            uint row_byte_base = token_id * bytes_per_row;

            uint lane = tid & 31u;          // 0..31
            uint g    = tid >> 5;           // group 0..7
            uint isc  = lane >> 4;          // 0 or 1 (scale half)

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = row_byte_base + block * 210;

                float d = unpackHalf2x16(gByte(b0 + 208) | (gByte(b0 + 209) << 8)).x;
                float scale = d * float(gInt8(b0 + 192 + 2u * g + isc));

                // ql byte: groups {0,2}->ql0, {1,3}->ql1, {4,6}->ql2, {5,7}->ql3 (+lane).
                uint ql_index = (g < 4u) ? (g & 1u) : (2u + (g & 1u));
                uint ql_byte  = gByte(b0 + ql_index * 32u + lane);
                uint high     = (g >> 1) & 1u;  // groups 2,3,6,7 use the high nibble
                uint nib      = (high != 0u) ? (ql_byte >> 4) : (ql_byte & 0xFu);

                // qh: groups 0-3 from qh0 (offset 128), 4-7 from qh1 (160); 2-bit field per group.
                uint qh_byte = (g < 4u) ? gByte(b0 + 128 + lane) : gByte(b0 + 160 + lane);
                uint shift   = 2u * (g & 3u);
                int q = int(nib | (((qh_byte >> shift) & 3u) << 4)) - 32;

                output_data[block * 256 + tid] = scale * float(q);
            }
        }
        """;

    /// <summary>
    /// Copy K and V vectors into the KV cache at the given position.
    /// Push constants: { uint kv_dim, uint position, uint max_seq_len }.
    /// Bindings: 0=k_input[kv_dim], 1=v_input[kv_dim], 2=k_cache[max_seq_len*kv_dim], 3=v_cache[max_seq_len*kv_dim].
    /// </summary>
    internal const string KvAppend = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { float k_cache[]; };
        layout(binding = 3) buffer VCache { float v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;
            uint max_seq_len;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= kv_dim) return;
            uint offset = position * kv_dim + i;
            k_cache[offset] = k_input[i];
            v_cache[offset] = v_input[i];
        }
        """;

    /// <summary>
    /// Batched fp32 <see cref="KvAppend"/> (issue #308): appends K rows of K/V into the cache in ONE
    /// dispatch. 2D grid <c>(ceil(kv_dim/256), K)</c>: column = <c>gl_GlobalInvocationID.x</c> (guarded
    /// against <c>kv_dim</c>), token row = <c>gl_WorkGroupID.y</c>. Row r is written at cache slot
    /// <c>base_pos + r</c>, reading input row r at <c>r * kv_dim</c>. Bit-identical to K separate
    /// <see cref="KvAppend"/> calls at positions base_pos, base_pos+1, … (same element addressing,
    /// no ring modulo). Push constants reuse the <see cref="KvAppend"/> layout (<c>position</c>
    /// carries base_pos). Bindings: 0=k_input[K*kv_dim], 1=v_input[K*kv_dim], 2=k_cache, 3=v_cache.
    /// </summary>
    internal const string KvAppendBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { float k_cache[]; };
        layout(binding = 3) buffer VCache { float v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;     // base_pos; row r writes slot base_pos + r
            uint max_seq_len;
        };

        void main() {
            uint col = gl_GlobalInvocationID.x;
            uint row = gl_WorkGroupID.y;
            if (col >= kv_dim) return;
            // Same element address as the single KvAppend: (position + row) * kv_dim + col.
            uint cache_off = (position + row) * kv_dim + col;
            uint in_off    = row * kv_dim + col;
            k_cache[cache_off] = k_input[in_off];
            v_cache[cache_off] = v_input[in_off];
        }
        """;

    /// <summary>
    /// Scaled dot-product attention with GQA support.
    /// One workgroup per query head. Each workgroup computes:
    ///   scores[t] = Q_h · K[t, kvHead] / sqrt(headDim) for t=0..seqLen
    ///   softmax(scores)
    ///   output[h] = sum(scores[t] * V[t, kvHead])
    ///
    /// For seq_len &lt;= 4096: stores all scores in shared memory — single Q·K pass,
    /// then softmax, then value accumulation. Matches the TqAttention approach.
    /// For seq_len &gt; 4096: triple-pass with Q·K recomputation (correctness over performance).
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim, uint seq_len, uint max_seq_len }.
    /// Bindings: 0=Q[num_heads*head_dim], 1=K_cache[max_seq_len*kv_dim], 2=V_cache[max_seq_len*kv_dim], 3=output[num_heads*head_dim].
    /// </summary>
    internal const string Attention = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { float k_cache[]; };
        layout(binding = 2) readonly buffer VCache { float v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };
        layout(binding = 4) buffer ScoresScratch   { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint max_seq_len;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        // Score-storage strategy mirrors the CUDA `llm_attention` kernel:
        //   • seq_len ≤ MAX_SHARED_SCORES (4096): fast path uses shared memory.
        //   • seq_len > 4096: spills to scores_scratch[h*max_seq_len .. +seq_len).
        // The fast path does not touch the scratch buffer, but Vulkan descriptors
        // require it to be bound — callers pass a 1-float placeholder when the
        // whole context is guaranteed to fit in shared memory.
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint q_off = h * head_dim;
            uint out_off = h * head_dim;

            // Sliding-window bound (Gemma SWA layers): mirror the CPU ForwardPass.Attention
            // start_seq = window > 0 ? max(0, seq_len - window) : 0. Computed with the uint
            // underflow guard (window < seq_len) so window==0 OR window>=seq_len ⇒ full attention.
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;

            bool use_shared = (seq_len <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position Q·K scores over [start_seq, seq_len) ───
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                float score = dot * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            // Pad the shared tail so the max scan ignores stale slots. The masked-off head
            // ([0, start_seq)) is never read because every scan below starts at start_seq.
            if (use_shared) {
                for (uint t = seq_len + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [start_seq, seq_len) ───
            float local_max = -1.0/0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                local_max = max(local_max, s);
            }
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                float e = exp(s - max_val);
                if (use_shared) scores[t] = e;
                else            scores_scratch[scratch_base + t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum over [start_seq, seq_len). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = start_seq; t < seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += weight * v_cache[v_off + d];
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// Batched fp32 attention (issue #308): the spec-decode batched-verify crux. Runs K queries in
    /// ONE dispatch over a 2D grid <c>num_heads × num_queries</c> workgroups, where query qi (at
    /// absolute position <c>base_pos + qi</c>) attends causally over <c>[0, base_pos + qi]</c> — i.e.
    /// query qi's <c>seq_len_i = base_pos + qi + 1</c>. This reproduces the causal-among-K behavior of
    /// K separate single-query <see cref="Attention"/> calls at seqLens base_pos+1 … base_pos+K
    /// WITHOUT the per-token gather/scatter.
    ///
    /// CRITICAL — bit-exactness: each <c>(h, qi)</c> workgroup is an INDEPENDENT copy of the
    /// single-query <see cref="Attention"/> ≤4096 shared-memory fast path with <c>seq_len = seq_len_i</c>
    /// and <c>window = 0</c> (no SWA — spec verify never windows). Score iteration order, the
    /// <c>sdata[256]</c> tree reduce, the <c>exp</c>/<c>inv_sum</c> softmax, and the Phase-3 V-sum
    /// order are kept VERBATIM, so the result is bit-identical to the single-query shader. The shared
    /// <c>scores[]</c> tail is padded with -inf up to <c>seq_len_i</c> (per-row bound, NOT a fixed
    /// seqLen) so the max scan ignores stale slots. There is no split-KV / scratch fallback here:
    /// the caller restricts the batched attention to <c>base_pos + K ≤ 4096</c>.
    ///
    /// Q is read from <c>q_data</c> at <c>qi*(num_heads*head_dim) + h*head_dim</c> and output written
    /// to <c>out_data</c> at the same offset (no gather/scatter). K/V are read from the cache exactly
    /// like the single-query shader (<c>t*kv_dim + kv_head*head_dim + d</c>, GQA
    /// <c>kv_head = h/(num_heads/num_kv_heads)</c>, scale <c>inversesqrt(head_dim)</c>).
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, base_pos, max_seq_len, num_queries }.
    /// Bindings: 0=q_data[K*num_heads*head_dim], 1=K_cache, 2=V_cache, 3=out_data[K*num_heads*head_dim].
    /// </summary>
    /// <summary>
    /// Flash-attention-style batched attention: the weight-amortizing sibling of
    /// <see cref="AttentionBatched"/>, fixing the two defects that made long-prompt Vulkan prefill
    /// collapse (docs/perf-loop-progress.md iteration 30 measured 75.4 -> 6.5 t/s from 43 to 3218
    /// prompt tokens).
    ///
    /// <para><b>1. K/V is read once per HEAD, not once per query.</b> <see cref="AttentionBatched"/>
    /// dispatches a numHeads x numQueries grid, so every query workgroup independently walks the
    /// entire K cache and then the entire V cache — 16x redundant traffic at a 16-token prefill
    /// chunk. Here one workgroup owns a head and serves all its queries from a single streaming
    /// pass, so each K/V element is fetched once and reused across every query.</para>
    ///
    /// <para><b>2. No materialised score array.</b> <see cref="AttentionBatched"/> allocates a fixed
    /// <c>shared float scores[4096]</c> (16 KB) regardless of sequence length, capping GCN
    /// occupancy at ~3 workgroups/CU. Online softmax keeps only a running max and sum per query, so
    /// shared usage here is ~4.7 KB and independent of sequence length.</para>
    ///
    /// <para>Online softmax per tile: m_new = max(m_old, max(tile)); rescale = exp(m_old - m_new);
    /// l_new = l_old*rescale + sum(p); acc_new = acc_old*rescale + sum(p*V); output = acc / l.
    /// Accumulators live in REGISTERS (each thread owns up to MAX_ACC of the numQueries*head_dim
    /// slots, strided by the workgroup size) rather than shared memory, which is what keeps the
    /// shared footprint small.</para>
    ///
    /// <para>Causal masking is PER QUERY — query qi sits at absolute position basePos+qi and may
    /// attend [0, basePos+qi] — so different queries in the same workgroup mask the same tile
    /// differently. Every query attends at least position 0, so l is never zero. A fully-masked
    /// later tile yields m_tile = -inf, m_new = m_old (finite), rescale = 1 and p = 0, which is why
    /// the -inf guard below only needs to cover m_new itself being -inf.</para>
    ///
    /// Bindings: 0 = Q [numQueries][numHeads*headDim], 1 = K cache, 2 = V cache,
    /// 3 = out [numQueries][numHeads*headDim]. Grid: numHeads workgroups. fp32 KV only.
    /// Limits: headDim &lt;= 128, numQueries &lt;= 16 (both enforced by the caller).
    /// </summary>
    internal const string AttentionBatchedFlash =
        AttentionBatchedFlashHead + "\n" + AttentionBatchedFlashAccessFp32 + "\n" + AttentionBatchedFlashBody;

    /// <summary>
    /// <see cref="AttentionBatchedFlash"/> reading a 16-bit-narrowed KV cache (the
    /// <c>STINGRAY_KV_DTYPE=bf16</c> store, which is physically <c>unpackHalf2x16</c> pairs —
    /// see <see cref="AttentionBatchedBf16"/>). Identical kernel otherwise.
    ///
    /// <para><b>Why this exists.</b> <c>GpuForwardPass</c> gated the flash kernel on
    /// <c>_kvDType == Float32</c>, so a narrowed KV cache silently fell back to
    /// <see cref="AttentionBatchedBf16"/> — the pre-flash kernel whose per-(head, query) dispatch
    /// re-reads the whole K/V cache for every query and whose 16 KB <c>scores[4096]</c> caps
    /// occupancy. Measured cost of that fallback on the reference part: prefill 84.6 → 61.8 t/s at
    /// 320 prompt tokens, and 49.3 → 20.3 t/s at 2898 — a gap that widens with context exactly as
    /// an O(N²) kernel against a tiled one should. bf16 was buying a 48% decode win and paying for
    /// it with a 2.4x prefill loss, purely because this variant did not exist.</para>
    ///
    /// <para>The flash kernel stages K and V into shared memory before any arithmetic, so the
    /// narrowing touches exactly two loads; everything downstream reads the same
    /// <c>shared float kvs[]</c>. That is why the two variants share their entire body.</para>
    /// </summary>
    internal const string AttentionBatchedFlashBf16 =
        AttentionBatchedFlashHead + "\n" + AttentionBatchedFlashAccessBf16 + "\n" + AttentionBatchedFlashBody;

    // Shared prologue/body/accessor split (see MatVecBatchedQ4KInt8 for the same technique):
    // `private` const = fragment, `internal` const = complete shader. SpirvGen and
    // VulkanPrecompiledShaderTests both key off that distinction.
    private const string AttentionBatchedFlashHead = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define TILE     32u
        #define MAX_Q    16u
        #define MAX_HD   128u
        #define WG       256u
        #define MAX_ACC  8

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q      { float q_data[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };
        """;

    private const string AttentionBatchedFlashAccessFp32 = """
        layout(binding = 1) readonly buffer KCache { float k_cache[]; };
        layout(binding = 2) readonly buffer VCache { float v_cache[]; };

        float readK(uint i) { return k_cache[i]; }
        float readV(uint i) { return v_cache[i]; }
        """;

    // Two 16-bit values per uint, low half first — the exact layout KvAppendBf16 writes and
    // AttentionBatchedBf16 reads. Element i lives in word i>>1, component i&1. kv_dim and
    // head_dim are both even (the constructor rejects odd head dims), so a head never starts
    // mid-word and this indexing cannot straddle.
    private const string AttentionBatchedFlashAccessBf16 = """
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };

        float readK(uint i) { return unpackHalf2x16(k_cache[i >> 1u])[i & 1u]; }
        float readV(uint i) { return unpackHalf2x16(v_cache[i >> 1u])[i & 1u]; }
        """;

    private const string AttentionBatchedFlashBody = """
        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint base_pos;      // query qi is at absolute position base_pos + qi
            uint max_seq_len;
            uint num_queries;   // K
        };

        // ~4.7 KB total, independent of sequence length.
        shared float kvs[TILE * MAX_HD];   // K tile, then reused for the V tile
        shared float sc[MAX_Q * TILE];     // tile scores, overwritten in place with probabilities
        shared float mrun[MAX_Q];          // running max per query
        shared float lrun[MAX_Q];          // running sum per query
        shared float resc[MAX_Q];          // this tile's rescale factor per query

        void main() {
            uint h   = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head    = h / (num_heads / num_kv_heads);
            uint kv_dim     = num_kv_heads * head_dim;
            uint row_stride = num_heads * head_dim;
            float scale     = inversesqrt(float(head_dim));

            uint kv_len   = base_pos + num_queries;
            uint accSlots = num_queries * head_dim;

            const float NEG_INF = -1.0 / 0.0;

            float accReg[MAX_ACC];
            [[unroll]] for (uint i = 0u; i < MAX_ACC; i++) accReg[i] = 0.0;

            if (tid < num_queries) {
                mrun[tid] = NEG_INF;
                lrun[tid] = 0.0;
            }
            barrier();

            for (uint tile_start = 0u; tile_start < kv_len; tile_start += TILE) {
                // Load the K tile once for the whole head; every query below reuses it.
                for (uint idx = tid; idx < TILE * head_dim; idx += WG) {
                    uint t = idx / head_dim;
                    uint d = idx - t * head_dim;
                    uint t_abs = tile_start + t;
                    kvs[idx] = (t_abs < kv_len)
                        ? readK(t_abs * kv_dim + kv_head * head_dim + d)
                        : 0.0;
                }
                barrier();

                // Scores for every (query, tile position), causally masked per query.
                for (uint idx = tid; idx < num_queries * TILE; idx += WG) {
                    uint q = idx / TILE;
                    uint t = idx - q * TILE;
                    uint t_abs = tile_start + t;
                    float sv = NEG_INF;
                    if (t_abs <= base_pos + q) {
                        float dot = 0.0;
                        uint qo = q * row_stride + h * head_dim;
                        for (uint d = 0u; d < head_dim; d++)
                            dot += q_data[qo + d] * kvs[t * head_dim + d];
                        sv = dot * scale;
                    }
                    sc[idx] = sv;
                }
                barrier();

                // Online softmax update, one thread per query.
                if (tid < num_queries) {
                    float mt = NEG_INF;
                    for (uint t = 0u; t < TILE; t++)
                        mt = max(mt, sc[tid * TILE + t]);

                    float mo = mrun[tid];
                    float mn = max(mo, mt);
                    // mn is -inf only if nothing has been seen yet AND this tile is fully masked,
                    // which cannot happen (every query attends position 0). Guarded anyway so the
                    // -inf minus -inf that would yield NaN is never evaluated.
                    float rs = (mn == NEG_INF) ? 1.0 : exp(mo - mn);

                    float ls = 0.0;
                    for (uint t = 0u; t < TILE; t++) {
                        float pv = (mn == NEG_INF) ? 0.0 : exp(sc[tid * TILE + t] - mn);
                        sc[tid * TILE + t] = pv;
                        ls += pv;
                    }
                    mrun[tid] = mn;
                    lrun[tid] = lrun[tid] * rs + ls;
                    resc[tid] = rs;
                }
                barrier();

                // Reuse the tile buffer for V — the scores have already been consumed into sc.
                for (uint idx = tid; idx < TILE * head_dim; idx += WG) {
                    uint t = idx / head_dim;
                    uint d = idx - t * head_dim;
                    uint t_abs = tile_start + t;
                    kvs[idx] = (t_abs < kv_len)
                        ? readV(t_abs * kv_dim + kv_head * head_dim + d)
                        : 0.0;
                }
                barrier();

                // acc = acc*rescale + sum_t p[q][t] * V[t][d]; accumulators stay in registers.
                [[unroll]] for (uint i = 0u; i < MAX_ACC; i++) {
                    uint sIdx = tid + i * WG;
                    if (sIdx < accSlots) {
                        uint q = sIdx / head_dim;
                        uint d = sIdx - q * head_dim;
                        float a = accReg[i] * resc[q];
                        for (uint t = 0u; t < TILE; t++)
                            a += sc[q * TILE + t] * kvs[t * head_dim + d];
                        accReg[i] = a;
                    }
                }
                barrier();   // before the next tile overwrites kvs
            }

            [[unroll]] for (uint i = 0u; i < MAX_ACC; i++) {
                uint sIdx = tid + i * WG;
                if (sIdx < accSlots) {
                    uint q = sIdx / head_dim;
                    uint d = sIdx - q * head_dim;
                    out_data[q * row_stride + h * head_dim + d] = accReg[i] / lrun[q];
                }
            }
        }
        """;

    internal const string AttentionBatched = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { float k_cache[]; };
        layout(binding = 2) readonly buffer VCache { float v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint base_pos;      // query qi is at absolute position base_pos + qi
            uint max_seq_len;
            uint num_queries;   // K
        };

        // seq_len_i = base_pos + qi + 1 ≤ base_pos + K, and the caller guarantees base_pos + K ≤ 4096,
        // so the whole causal range always fits in shared memory — no scratch-spill path here.
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        void main() {
            uint h   = gl_WorkGroupID.x;
            uint qi  = gl_WorkGroupID.y;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads || qi >= num_queries) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint row_stride = num_heads * head_dim;
            uint q_off   = qi * row_stride + h * head_dim;
            uint out_off = qi * row_stride + h * head_dim;

            // Per-query causal length: query qi (abs pos base_pos+qi) attends [0, base_pos+qi].
            // window = 0 (no SWA), start_seq = 0.
            uint seq_len_i = base_pos + qi + 1u;

            // ─── Phase 1: per-position Q·K scores over [0, seq_len_i) ───
            for (uint t = tid; t < seq_len_i; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                scores[t] = dot * scale;
            }
            // No tail padding needed: every later phase (max scan, exp/sum, V-aggregate) is
            // strictly bounded by seq_len_i, so scores[t >= seq_len_i] is never read.
            barrier();

            // ─── Phase 2: in-place softmax over [0, seq_len_i) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < seq_len_i; t += 256)
                local_max = max(local_max, scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = tid; t < seq_len_i; t += 256) {
                float e = exp(scores[t] - max_val);
                scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            for (uint t = tid; t < seq_len_i; t += 256)
                scores[t] *= inv_sum;
            barrier();

            // ─── Phase 3: weighted V sum over [0, seq_len_i). ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = 0; t < seq_len_i; t++) {
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += scores[t] * v_cache[v_off + d];
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// bf16 (issue #308 follow-up / #332) variant of <see cref="AttentionBatched"/>: control flow,
    /// per-query causal range (<c>seq_len_i = base_pos + qi + 1</c>), no-tail-pad, and 2D grid
    /// (<c>num_heads × num_queries</c>) are IDENTICAL to the fp32 <see cref="AttentionBatched"/>; the
    /// only difference is the K/V cache buffers (bindings 1, 2) are <c>uint[]</c> holding IEEE fp16
    /// packed two-per-uint (<c>unpackHalf2x16</c> on read), using the SAME read idiom as the
    /// single-query <see cref="AttentionBf16"/>. All scores / softmax / value accumulation stay fp32,
    /// so each <c>(h, qi)</c> workgroup is bit-identical to a single-query <see cref="AttentionBf16"/>
    /// call at <c>seq_len = base_pos + qi + 1</c>. No scratch-spill (caller restricts base_pos+K ≤ 4096).
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, base_pos, max_seq_len, num_queries }.
    /// Bindings: 0=q_data (float), 1=K_cache (uint, packed fp16×2), 2=V_cache (uint, packed fp16×2),
    ///           3=out_data (float).
    /// </summary>
    internal const string AttentionBatchedBf16 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint base_pos;      // query qi is at absolute position base_pos + qi
            uint max_seq_len;
            uint num_queries;   // K
        };

        // seq_len_i = base_pos + qi + 1 ≤ base_pos + K ≤ 4096 ⇒ the whole causal range fits in
        // shared memory; no scratch-spill path (matches the fp32 AttentionBatched).
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        void main() {
            uint h   = gl_WorkGroupID.x;
            uint qi  = gl_WorkGroupID.y;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads || qi >= num_queries) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint row_stride = num_heads * head_dim;
            uint q_off   = qi * row_stride + h * head_dim;
            uint out_off = qi * row_stride + h * head_dim;

            // Per-query causal length: query qi (abs pos base_pos+qi) attends [0, base_pos+qi].
            uint seq_len_i = base_pos + qi + 1u;

            // ─── Phase 1: per-position Q·K scores over [0, seq_len_i) ───
            for (uint t = tid; t < seq_len_i; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                // Read each packed fp16 word once (two K elements at a time) — same idiom as the
                // single-query AttentionBf16. k_off is even (head_dim even, see GpuForwardPass guard).
                uint k_off_half = k_off >> 1;
                for (uint dh = 0; dh < (head_dim >> 1); dh++) {
                    uint d = dh << 1;
                    vec2 kv = unpackHalf2x16(k_cache[k_off_half + dh]);
                    dot += q_data[q_off + d] * kv.x + q_data[q_off + d + 1u] * kv.y;
                }
                scores[t] = dot * scale;
            }
            // No tail padding needed: every later phase is strictly bounded by seq_len_i.
            barrier();

            // ─── Phase 2: in-place softmax over [0, seq_len_i) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < seq_len_i; t += 256)
                local_max = max(local_max, scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = tid; t < seq_len_i; t += 256) {
                float e = exp(scores[t] - max_val);
                scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            for (uint t = tid; t < seq_len_i; t += 256)
                scores[t] *= inv_sum;
            barrier();

            // ─── Phase 3: weighted V sum over [0, seq_len_i). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                // Each thread owns ONE output dim d (threads 256 apart, so adjacent d can't be
                // paired). Hoist the per-d word/component selection out of the t-loop and walk the
                // V row word base incrementally — same idiom as the single-query AttentionBf16.
                uint d_half = d >> 1;
                uint component = d & 1u;
                uint v_off_half = (kv_head * head_dim) >> 1;   // t = 0 row word base
                uint kv_dim_half = kv_dim >> 1;
                float sum = 0.0;
                for (uint t = 0; t < seq_len_i; t++) {
                    float vv = unpackHalf2x16(v_cache[v_off_half + d_half])[component];
                    sum += scores[t] * vv;
                    v_off_half += kv_dim_half;
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// q8_0 (issue #308 follow-up / #332) variant of <see cref="AttentionBatched"/>: control flow,
    /// per-query causal range (<c>seq_len_i = base_pos + qi + 1</c>), no-tail-pad, and 2D grid
    /// (<c>num_heads × num_queries</c>) are IDENTICAL to the fp32 <see cref="AttentionBatched"/>; the
    /// only difference is the K/V cache buffers (bindings 1, 2) are <c>uint[]</c> holding ggml
    /// <c>block_q8_0</c> (34 bytes/block: fp16 scale + 32 int8), read via the SAME byte-gather +
    /// dequant idiom as the single-query <see cref="AttentionQ8_0"/>. All scores / softmax / value
    /// accumulation stay fp32, so each <c>(h, qi)</c> workgroup is bit-identical to a single-query
    /// <see cref="AttentionQ8_0"/> call at <c>seq_len = base_pos + qi + 1</c>. No scratch-spill (caller
    /// restricts base_pos+K ≤ 4096). kv_dim%32==0, so blocks never straddle a KV row.
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, base_pos, max_seq_len, num_queries }.
    /// Bindings: 0=q_data (float), 1=K_cache (uint, block_q8_0), 2=V_cache (uint, block_q8_0),
    ///           3=out_data (float).
    /// </summary>
    internal const string AttentionBatchedQ8_0 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint base_pos;      // query qi is at absolute position base_pos + qi
            uint max_seq_len;
            uint num_queries;   // K
        };

        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        // Sign-extend a single int8 byte in one bitfieldExtract (no ternary branch) — same as
        // the single-query AttentionQ8_0.
        int gInt8K(uint b) { return bitfieldExtract(int(k_cache[b >> 2]), int((b & 3u) * 8u), 8); }
        int gInt8V(uint b) { return bitfieldExtract(int(v_cache[b >> 2]), int((b & 3u) * 8u), 8); }

        float loadK(uint e) {
            uint blk = e >> 5; uint lane = e & 31u; uint b0 = blk * 34u;
            uint w = k_cache[b0 >> 2];
            float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
            return dsc * float(gInt8K(b0 + 2u + lane));
        }
        float loadV(uint e) {
            uint blk = e >> 5; uint lane = e & 31u; uint b0 = blk * 34u;
            uint w = v_cache[b0 >> 2];
            float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
            return dsc * float(gInt8V(b0 + 2u + lane));
        }

        void main() {
            uint h   = gl_WorkGroupID.x;
            uint qi  = gl_WorkGroupID.y;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads || qi >= num_queries) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint row_stride = num_heads * head_dim;
            uint q_off   = qi * row_stride + h * head_dim;
            uint out_off = qi * row_stride + h * head_dim;

            uint seq_len_i = base_pos + qi + 1u;

            // ─── Phase 1: per-position Q·K scores over [0, seq_len_i) ───
            for (uint t = tid; t < seq_len_i; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * loadK(k_off + d);
                scores[t] = dot * scale;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [0, seq_len_i) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < seq_len_i; t += 256)
                local_max = max(local_max, scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = tid; t < seq_len_i; t += 256) {
                float e = exp(scores[t] - max_val);
                scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            for (uint t = tid; t < seq_len_i; t += 256)
                scores[t] *= inv_sum;
            barrier();

            // ─── Phase 3: weighted V sum over [0, seq_len_i). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = 0; t < seq_len_i; t++) {
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += scores[t] * loadV(v_off + d);
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// bf16 (issue #308 follow-up / #332) variant of <see cref="KvAppendBatched"/>: appends K rows of
    /// K/V from the packed <c>[K][kvDim]</c> inputs into the cache in ONE dispatch (row r at slot
    /// base_pos + r), storing IEEE fp16 packed two-per-uint (<c>packHalf2x16</c>) — the SAME write
    /// idiom as the single-token <see cref="KvAppendBf16"/>. 2D grid (<c>ceil((kvDim/2)/256), K</c>),
    /// one thread per 2 elements (kv_dim even). Bit-identical to K separate <see cref="KvAppendBf16"/>
    /// calls. Indexes the cache identically to fp32 (<c>(base_pos + row) * kv_dim + i</c>, word-granular).
    ///
    /// Push constants: { uint kv_dim, position (base_pos), max_seq_len }.
    /// Bindings: 0=k_input[K*kv_dim] (float), 1=v_input[K*kv_dim] (float),
    ///           2=k_cache (uint, packed fp16×2), 3=v_cache (uint, packed fp16×2).
    /// </summary>
    internal const string KvAppendBatchedBf16 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { uint k_cache[]; };
        layout(binding = 3) buffer VCache { uint v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;     // base_pos; row r writes slot base_pos + r
            uint max_seq_len;
        };

        void main() {
            uint w = gl_GlobalInvocationID.x;
            uint row = gl_WorkGroupID.y;
            uint half_dim = kv_dim >> 1;   // kv_dim is even (numKvHeads*headDim)
            if (w >= half_dim) return;
            uint i = w << 1;
            // Same element address as fp32 ((position + row) * kv_dim + i), expressed in words.
            uint row_word = (position + row) * half_dim;
            uint in_elem  = row * kv_dim + i;   // first of the 2 source float elements (element-granular)
            k_cache[row_word + w] = packHalf2x16(vec2(k_input[in_elem], k_input[in_elem + 1u]));
            v_cache[row_word + w] = packHalf2x16(vec2(v_input[in_elem], v_input[in_elem + 1u]));
        }
        """;

    /// <summary>
    /// q8_0 (issue #308 follow-up / #332) variant of <see cref="KvAppendBatched"/>: appends K rows of
    /// K/V from the packed <c>[K][kvDim]</c> inputs into the cache in ONE dispatch (row r at slot
    /// base_pos + r), block-quantizing into ggml <c>block_q8_0</c> (34 bytes/block) with the SAME
    /// amax→quant + masked-atomic-byte-store idiom as the single-token <see cref="KvAppendQ8_0"/>. 2D
    /// grid (<c>ceil((kvDim/32)/256), K</c>), one thread per 32-element block. Bit-identical to K
    /// separate <see cref="KvAppendQ8_0"/> calls: every thread (across ALL blocks AND rows) owns a
    /// DISJOINT set of destination bytes; the only sharing is at seam uint words (between adjacent
    /// blocks within a row and, when blocks_per_row is odd, between the last block of one row and the
    /// first of the next), which the masked atomicAnd+atomicOr byte writer makes correct under any
    /// interleaving. So the result is independent of dispatch order. Indexes the cache identically to fp32
    /// (<c>(base_pos + row) * kv_dim + i</c>, expressed in blocks). kv_dim%32==0.
    ///
    /// Push constants: { uint kv_dim, position (base_pos), max_seq_len }.
    /// Bindings: 0=k_input[K*kv_dim] (float), 1=v_input[K*kv_dim] (float),
    ///           2=k_cache (uint, block_q8_0), 3=v_cache (uint, block_q8_0).
    /// </summary>
    internal const string KvAppendBatchedQ8_0 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { uint k_cache[]; };
        layout(binding = 3) buffer VCache { uint v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;     // base_pos; row r writes slot base_pos + r
            uint max_seq_len;
        };

        // Masked-atomic byte writers: clear the target byte, then OR in the value. Disjoint bytes
        // within a shared uint stay correct under any interleaving — same as KvAppendQ8_0.
        void sByteK(uint b, uint val) {
            uint w = b >> 2; uint sh = (b & 3u) * 8u;
            atomicAnd(k_cache[w], ~(0xFFu << sh));
            atomicOr (k_cache[w], (val & 0xFFu) << sh);
        }
        void sByteV(uint b, uint val) {
            uint w = b >> 2; uint sh = (b & 3u) * 8u;
            atomicAnd(v_cache[w], ~(0xFFu << sh));
            atomicOr (v_cache[w], (val & 0xFFu) << sh);
        }

        void main() {
            uint blk = gl_GlobalInvocationID.x;
            uint row = gl_WorkGroupID.y;
            uint blocks_per_row = kv_dim >> 5;   // kv_dim % 32 == 0
            if (blk >= blocks_per_row) return;

            // Same element address as fp32 ((position + row) * kv_dim + i), expressed in blocks.
            uint dst_block = (position + row) * blocks_per_row + blk;
            uint b0 = dst_block * 34u;
            uint src = row * kv_dim + (blk << 5);   // first source element of this block

            // ── K block ──
            {
                float amax = 0.0;
                for (uint j = 0u; j < 32u; j++)
                    amax = max(amax, abs(k_input[src + j]));
                float d = amax / 127.0;
                float invd = (d < 1e-30) ? 0.0 : (1.0 / d);
                uint dh = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                sByteK(b0, dh & 0xFFu);
                sByteK(b0 + 1u, dh >> 8);
                for (uint j = 0u; j < 32u; j++) {
                    int q = clamp(int(round(k_input[src + j] * invd)), -127, 127);
                    sByteK(b0 + 2u + j, uint(q & 0xFF));
                }
            }
            // ── V block ──
            {
                float amax = 0.0;
                for (uint j = 0u; j < 32u; j++)
                    amax = max(amax, abs(v_input[src + j]));
                float d = amax / 127.0;
                float invd = (d < 1e-30) ? 0.0 : (1.0 / d);
                uint dh = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                sByteV(b0, dh & 0xFFu);
                sByteV(b0 + 1u, dh >> 8);
                for (uint j = 0u; j < 32u; j++) {
                    int q = clamp(int(round(v_input[src + j] * invd)), -127, 127);
                    sByteV(b0 + 2u + j, uint(q & 0xFF));
                }
            }
        }
        """;

    /// <summary>
    /// bf16 (issue #311) variant of <see cref="KvAppend"/>: the K/V cache buffers store
    /// IEEE fp16 packed two-per-uint via core-GLSL <c>packHalf2x16</c> (no device extension).
    /// The user-facing <c>--kv-type bf16</c> means "half-width KV"; Vulkan stores fp16
    /// because for the small-magnitude KV values fp16 is more precise than bf16. Arithmetic
    /// elsewhere stays fp32 — only the stored value is narrowed.
    ///
    /// CRITICAL: this indexes the cache IDENTICALLY to the fp32 <see cref="KvAppend"/>
    /// (<c>position * kv_dim + i</c> element addressing, just expressed in words because
    /// each word holds 2 elements). There is NO <c>% max_seq_len</c> ring modulo, matching
    /// the fp32 shader exactly. kv_dim is always even (numKvHeads*headDim), so we dispatch
    /// one thread per 2 elements (word granular).
    ///
    /// Push constants: { uint kv_dim, uint position, uint max_seq_len } — unchanged.
    /// Bindings: 0=k_input[kv_dim] (float), 1=v_input[kv_dim] (float),
    ///           2=k_cache (uint, packed fp16×2), 3=v_cache (uint, packed fp16×2).
    /// </summary>
    internal const string KvAppendBf16 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { uint k_cache[]; };
        layout(binding = 3) buffer VCache { uint v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;
            uint max_seq_len;
        };

        void main() {
            uint w = gl_GlobalInvocationID.x;
            uint half_dim = kv_dim >> 1;   // kv_dim is even (numKvHeads*headDim)
            if (w >= half_dim) return;
            uint i = w << 1;
            // Same element address as fp32 (position * kv_dim + i), expressed in words.
            uint row_word = position * half_dim;
            k_cache[row_word + w] = packHalf2x16(vec2(k_input[i], k_input[i + 1u]));
            v_cache[row_word + w] = packHalf2x16(vec2(v_input[i], v_input[i + 1u]));
        }
        """;

    /// <summary>
    /// bf16 (issue #311) variant of <see cref="Attention"/>: control flow is IDENTICAL to
    /// the fp32 shader; the only difference is that the K/V cache buffers (bindings 1, 2)
    /// are <c>uint[]</c> holding IEEE fp16 packed two-per-uint (<c>packHalf2x16</c> on
    /// write, <c>unpackHalf2x16</c> on read), so every element read becomes an unpack +
    /// lane-select. All scores / softmax / value accumulation stay fp32 — the arithmetic is
    /// bit-identical to the fp32 Attention; only the stored K/V mantissa is narrowed.
    /// scores_scratch (binding 4) stays fp32. The <c>inversesqrt(head_dim)</c> scale is kept
    /// exactly as the fp32 shader has it.
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, seq_len, max_seq_len } — unchanged.
    /// Bindings: 0=Q (float), 1=K_cache (uint, packed fp16×2), 2=V_cache (uint, packed fp16×2),
    ///           3=output (float), 4=scores_scratch (float).
    /// </summary>
    internal const string AttentionBf16 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };
        layout(binding = 4) buffer ScoresScratch   { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint max_seq_len;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        // Score-storage strategy mirrors the fp32 Attention shader.
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint q_off = h * head_dim;
            uint out_off = h * head_dim;

            // SWA bound — mirrors the fp32 Attention shader (CPU ForwardPass.Attention).
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;

            bool use_shared = (seq_len <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position Q·K scores over [start_seq, seq_len) ───
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                // Read each packed fp16 word once (two K elements at a time). k_off is even
                // (head_dim is even — see the GpuForwardPass guard) so k_off>>1 is the exact
                // word base and consecutive d,d+1 are the two halves of word k_off_half+dh.
                uint k_off_half = k_off >> 1;
                for (uint dh = 0; dh < (head_dim >> 1); dh++) {
                    uint d = dh << 1;
                    vec2 kv = unpackHalf2x16(k_cache[k_off_half + dh]);
                    dot += q_data[q_off + d] * kv.x + q_data[q_off + d + 1u] * kv.y;
                }
                float score = dot * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            if (use_shared) {
                for (uint t = seq_len + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [start_seq, seq_len) ───
            float local_max = -1.0/0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                local_max = max(local_max, s);
            }
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                float e = exp(s - max_val);
                if (use_shared) scores[t] = e;
                else            scores_scratch[scratch_base + t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum over [start_seq, seq_len). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                // Each thread owns ONE output dim d (threads are 256 apart, so adjacent d can't
                // be paired). Hoist the per-d word/component selection out of the t-loop and walk
                // the V row word base incrementally. v_off = t*kv_dim + kv_head*head_dim is even
                // (head_dim is even — see the GpuForwardPass guard), so v_off>>1 is the exact word.
                uint d_half = d >> 1;
                uint component = d & 1u;
                uint v_off_half = ((start_seq * kv_dim) + kv_head * head_dim) >> 1;   // t = start_seq row word base
                uint kv_dim_half = kv_dim >> 1;
                float sum = 0.0;
                for (uint t = start_seq; t < seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                    float vv = unpackHalf2x16(v_cache[v_off_half + d_half])[component];
                    sum += weight * vv;
                    v_off_half += kv_dim_half;
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// q8_0 (issue #325) variant of <see cref="KvAppend"/>: block-quantizes the K/V vectors
    /// into the cache as ggml <c>block_q8_0</c> (34 bytes = fp16 scale + 32 int8, per 32
    /// elements; ~4× smaller than fp32). Mirrors the CUDA <c>llm_kv_append_q8_0</c> /
    /// <c>opentail-llm_q8_append_one</c> ground truth: per 32-element block, <c>amax = max(|x|)</c>,
    /// <c>d = amax / 127</c>, <c>invd = (d &lt; 1e-30) ? 0 : 1/d</c> (the 1e-30 guard avoids
    /// 0*inf=NaN — replicated verbatim), <c>q = clamp(round(x*invd), -127, 127)</c>.
    ///
    /// Dispatched ONE THREAD PER 32-ELEMENT BLOCK (not a subgroup — subgroup width is
    /// hardware-dependent; one-thread-per-block sidesteps that). Each thread owns all 34
    /// bytes of its destination block. Because the cache is bound as <c>uint[]</c> and a
    /// 34-byte block is not 4-aligned, adjacent blocks share the seam <c>uint</c> word but
    /// write DISJOINT bytes; the masked <c>atomicAnd</c>+<c>atomicOr</c> byte writer makes the
    /// disjoint-bitfield RMW correct under any thread interleaving (the atomicAnd clears the
    /// byte first, so ring-reuse overwrites cleanly — no zero-init needed).
    ///
    /// CRITICAL: indexes the cache IDENTICALLY to the fp32 <see cref="KvAppend"/>
    /// (<c>position * kv_dim + i</c> element addressing, expressed in blocks). No
    /// <c>% max_seq_len</c> ring modulo, matching the fp32/bf16 shaders. kv_dim is always a
    /// multiple of 32 (enforced in GpuForwardPass), so a KV row's blocks never straddle a row.
    ///
    /// Push constants: { uint kv_dim, position, max_seq_len } — unchanged.
    /// Bindings: 0=k_input[kv_dim] (float), 1=v_input[kv_dim] (float),
    ///           2=k_cache (uint, packed block_q8_0), 3=v_cache (uint, packed block_q8_0).
    /// </summary>
    internal const string KvAppendQ8_0 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { uint k_cache[]; };
        layout(binding = 3) buffer VCache { uint v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;
            uint max_seq_len;
        };

        // Masked-atomic byte writers: clear the target byte, then OR in the value.
        // Disjoint bytes within a shared uint stay correct under any interleaving.
        void sByteK(uint b, uint val) {
            uint w = b >> 2; uint sh = (b & 3u) * 8u;
            atomicAnd(k_cache[w], ~(0xFFu << sh));
            atomicOr (k_cache[w], (val & 0xFFu) << sh);
        }
        void sByteV(uint b, uint val) {
            uint w = b >> 2; uint sh = (b & 3u) * 8u;
            atomicAnd(v_cache[w], ~(0xFFu << sh));
            atomicOr (v_cache[w], (val & 0xFFu) << sh);
        }

        void main() {
            uint blk = gl_GlobalInvocationID.x;
            uint blocks_per_row = kv_dim >> 5;   // kv_dim % 32 == 0
            if (blk >= blocks_per_row) return;

            // Same element address as fp32 (position * kv_dim + i), expressed in blocks.
            uint dst_block = position * blocks_per_row + blk;
            uint b0 = dst_block * 34u;
            uint src = blk << 5;                 // first source element of this block

            // ── K block ──
            {
                float amax = 0.0;
                for (uint j = 0u; j < 32u; j++)
                    amax = max(amax, abs(k_input[src + j]));
                float d = amax / 127.0;
                float invd = (d < 1e-30) ? 0.0 : (1.0 / d);
                uint dh = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                sByteK(b0, dh & 0xFFu);
                sByteK(b0 + 1u, dh >> 8);
                for (uint j = 0u; j < 32u; j++) {
                    int q = clamp(int(round(k_input[src + j] * invd)), -127, 127);
                    sByteK(b0 + 2u + j, uint(q & 0xFF));
                }
            }
            // ── V block ──
            {
                float amax = 0.0;
                for (uint j = 0u; j < 32u; j++)
                    amax = max(amax, abs(v_input[src + j]));
                float d = amax / 127.0;
                float invd = (d < 1e-30) ? 0.0 : (1.0 / d);
                uint dh = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                sByteV(b0, dh & 0xFFu);
                sByteV(b0 + 1u, dh >> 8);
                for (uint j = 0u; j < 32u; j++) {
                    int q = clamp(int(round(v_input[src + j] * invd)), -127, 127);
                    sByteV(b0 + 2u + j, uint(q & 0xFF));
                }
            }
        }
        """;

    /// <summary>
    /// q8_0 (issue #325) variant of <see cref="Attention"/>: control flow is IDENTICAL to the
    /// fp32 shader; the only difference is that the K/V cache buffers (bindings 1, 2) are
    /// <c>uint[]</c> holding ggml <c>block_q8_0</c> (34 bytes/block: fp16 scale + 32 int8), so
    /// every element read becomes a byte-gather + dequant <c>value = fp16(d) * int8</c>. All
    /// scores / softmax / value accumulation stay fp32 — only the stored K/V is narrowed.
    /// scores_scratch (binding 4) stays fp32. The <c>inversesqrt(head_dim)</c> scale is kept
    /// exactly as the fp32 shader has it. Element addressing (<c>off = t*kv_dim + kv_head*head_dim</c>,
    /// <c>e = off + d</c>) is identical to fp32/bf16; per element <c>blk=e&gt;&gt;5</c>,
    /// <c>lane=e&amp;31</c>, <c>b0=blk*34</c>. kv_dim%32==0, so blocks never straddle a KV row.
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, seq_len, max_seq_len } — unchanged.
    /// Bindings: 0=Q (float), 1=K_cache (uint, block_q8_0), 2=V_cache (uint, block_q8_0),
    ///           3=output (float), 4=scores_scratch (float).
    /// </summary>
    internal const string AttentionQ8_0 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };
        layout(binding = 4) buffer ScoresScratch   { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint max_seq_len;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        // Score-storage strategy mirrors the fp32 Attention shader.
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        // Sign-extend a single int8 byte in one bitfieldExtract (no ternary branch).
        int gInt8K(uint b) { return bitfieldExtract(int(k_cache[b >> 2]), int((b & 3u) * 8u), 8); }
        int gInt8V(uint b) { return bitfieldExtract(int(v_cache[b >> 2]), int((b & 3u) * 8u), 8); }

        float loadK(uint e) {
            uint blk = e >> 5; uint lane = e & 31u; uint b0 = blk * 34u;
            // b0 = blk*34 is even, so the two scale bytes [b0, b0+1] live in the same uint word.
            uint w = k_cache[b0 >> 2];
            float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
            return dsc * float(gInt8K(b0 + 2u + lane));
        }
        float loadV(uint e) {
            uint blk = e >> 5; uint lane = e & 31u; uint b0 = blk * 34u;
            uint w = v_cache[b0 >> 2];
            float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
            return dsc * float(gInt8V(b0 + 2u + lane));
        }

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint q_off = h * head_dim;
            uint out_off = h * head_dim;

            // SWA bound — mirrors the fp32 Attention shader (CPU ForwardPass.Attention).
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;

            bool use_shared = (seq_len <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position Q·K scores over [start_seq, seq_len) ───
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * loadK(k_off + d);
                float score = dot * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            if (use_shared) {
                for (uint t = seq_len + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [start_seq, seq_len) ───
            float local_max = -1.0/0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                local_max = max(local_max, s);
            }
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                float e = exp(s - max_val);
                if (use_shared) scores[t] = e;
                else            scores_scratch[scratch_base + t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum over [start_seq, seq_len). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = start_seq; t < seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += weight * loadV(v_off + d);
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// SnapKV (issue #59) — per-head attention scoring across a layer's K cache.
    /// Mirrors the CUDA `llm_snapkv_score` kernel: one workgroup per query head,
    /// 256 threads. Phase 1 computes causal-masked dot(q_head, k_cache[t, kvHead, :]) * scale
    /// for every t in [0, prompt_len); Phase 2 runs an in-place softmax over the
    /// valid prefix; Phase 3 atomicAdds the post-softmax weights into a global
    /// per-position accumulator.
    ///
    /// Vulkan core has no native float atomicAdd, so binding 2 is bound twice —
    /// once as f32 ScoreAccum for readers, once as u32 ScoreAccumAtomic for the
    /// compare-and-swap loop. The two views share the same VkBuffer (same bit
    /// pattern; only the binding type differs).
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, prompt_len, q_abs_pos, max_seq_len }.
    /// Bindings:
    ///   0 = Q (readonly)
    ///   1 = K cache (readonly)
    ///   2 = score_accum, f32 view (coherent, atomic CAS via the u32 alias on the same buffer)
    ///   3 = scores_scratch (writeonly, only used when prompt_len &gt; 4096)
    /// </summary>
    internal const string SnapKvScore = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q      { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { float k_cache[]; };
        // The CUDA reference does a direct float atomicAdd here; Vulkan core
        // exposes only integer atomics, so we keep the storage f32 (callers
        // download it as floats) but mutate it through a uint alias via
        // atomicCompSwap. Same buffer bound twice — the bit pattern of one
        // view IS the bit pattern of the other.
        layout(binding = 2) coherent buffer ScoreAccumAtomic { uint accum_uint[]; };
        // Spill buffer for the > 4096 path: written in Phase 1, re-read in
        // Phase 2 (max-reduce + softmax) and Phase 3 (atomicAdd), so no
        // writeonly qualifier here.
        layout(binding = 3) buffer ScoresScratch   { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint prompt_len;
            uint q_abs_pos;
            uint max_seq_len;
        };

        const uint MAX_STORED_SCORES = 4096u;
        shared float scores[MAX_STORED_SCORES];
        shared float sdata[256];

        void atomicAddFloat(uint idx, float value) {
            // Compare-and-swap loop on the uint reinterpretation of the f32 word.
            // The CUDA path uses native float atomicAdd; this matches its semantics
            // (last-writer-wins associative accumulate) on Vulkan core.
            uint oldBits = accum_uint[idx];
            while (true) {
                float oldVal = uintBitsToFloat(oldBits);
                float newVal = oldVal + value;
                uint newBits = floatBitsToUint(newVal);
                uint prev = atomicCompSwap(accum_uint[idx], oldBits, newBits);
                if (prev == oldBits) return;
                oldBits = prev;
            }
        }

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim  = num_kv_heads * head_dim;
            float scale  = inversesqrt(float(head_dim));
            uint q_off   = h * head_dim;

            bool use_shared = (prompt_len <= MAX_STORED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position causal-masked Q·K dot ───
            for (uint t = tid; t < prompt_len; t += 256) {
                float score;
                if (t > q_abs_pos) {
                    score = -1.0/0.0;
                } else {
                    float dot = 0.0;
                    uint k_off = t * kv_dim + kv_head * head_dim;
                    for (uint d = 0; d < head_dim; d++)
                        dot += q_data[q_off + d] * k_cache[k_off + d];
                    score = dot * scale;
                }
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            // Pad shared tail so max-reduce ignores stale slots. Scratch reads
            // iterate only [0, prompt_len), so no padding needed there.
            if (use_shared) {
                for (uint t = prompt_len + tid; t < MAX_STORED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2a: max over [0, prompt_len) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < prompt_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                local_max = max(local_max, s);
            }
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            // ─── Phase 2b: exp(s - max), sum, normalize ───
            float local_sum = 0.0;
            for (uint t = tid; t < prompt_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                float e = (s == -1.0/0.0) ? 0.0 : exp(s - max_val);
                if (use_shared) scores[t] = e;
                else            scores_scratch[scratch_base + t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            // ─── Phase 3: atomicAdd softmax weight into global accumulator ───
            for (uint t = tid; t < prompt_len; t += 256) {
                if (t > q_abs_pos) continue;
                float w = (use_shared ? scores[t] : scores_scratch[scratch_base + t]) * inv_sum;
                atomicAddFloat(t, w);
            }
        }
        """;

    /// <summary>
    /// SnapKV (issue #59) — gather kept positions of one KV ring (K or V) into a
    /// dense <c>[K * kv_dim]</c> prefix of <c>dst</c>. <c>src</c> and <c>dst</c> MUST be
    /// different buffers; the destination is later copied back over the ring's
    /// <c>[0, K * kv_dim)</c> region by the caller.
    ///
    /// Each thread copies one float from src[keep[blockIdx.y] * kv_dim + d] to
    /// dst[blockIdx.y * kv_dim + d]. Grid = (ceil(kv_dim/256), K, 1), block 256.
    ///
    /// Push constants: { uint K, kv_dim }.
    /// Bindings: 0=src (readonly), 1=dst (writeonly), 2=keep_positions (readonly int32).
    /// </summary>
    internal const string KvCompact = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Src  { float src_data[]; };
        layout(binding = 1) writeonly buffer Dst  { float dst_data[]; };
        layout(binding = 2) readonly  buffer Keep { int   keep_positions[]; };

        layout(push_constant) uniform Params {
            uint K;
            uint kv_dim;
        };

        void main() {
            uint i = gl_WorkGroupID.y;
            if (i >= K) return;
            uint d = gl_GlobalInvocationID.x;
            if (d >= kv_dim) return;

            uint src_pos = uint(keep_positions[i]);
            uint src_off = src_pos * kv_dim + d;
            uint dst_off = i       * kv_dim + d;
            dst_data[dst_off] = src_data[src_off];
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q6_K dequantization.
    /// Same pattern as Q4_K but different block layout.
    /// Q6_K block (210 bytes per 256 elements):
    ///   [0:128]   ql — lower 4 bits
    ///   [128:192] qh — upper 2 bits
    ///   [192:208] 16 int8 scales
    ///   [208:210] FP16 d (super-block scale)
    /// </summary>
    internal const string MatVecQ6K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Q6_K block layout (210 bytes per 256 elements):
        //   [0:128]   ql — lower 4-bit nibbles (two 64-byte halves)
        //   [128:192] qh — upper 2-bit pairs (two 32-byte halves)
        //   [192:208] 16 int8 scale values
        //   [208:210] FP16 super-block scale d
        // Thread layout: each lane handles 8 elements (lane, lane+32, ..., lane+224)
        // which all share l = lane within their respective groups — no shared memory needed.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        int  gInt8(uint b) { int v = int(gByte(b)); return v >= 128 ? v - 256 : v; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;
            uint boff_base = row * num_blocks * 210;

            float acc = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 210;

                float d = unpackHalf2x16(gByte(b0 + 208) | (gByte(b0 + 209) << 8)).x;

                // Precompute the 8 scale floats needed by this lane.
                // isc = lane>>4 selects lower (0) or upper (1) sub-scale row per group.
                uint isc = lane >> 4;
                float sc0 = d * float(gInt8(b0 + 192 + isc));
                float sc1 = d * float(gInt8(b0 + 194 + isc));
                float sc2 = d * float(gInt8(b0 + 196 + isc));
                float sc3 = d * float(gInt8(b0 + 198 + isc));
                float sc4 = d * float(gInt8(b0 + 200 + isc));
                float sc5 = d * float(gInt8(b0 + 202 + isc));
                float sc6 = d * float(gInt8(b0 + 204 + isc));
                float sc7 = d * float(gInt8(b0 + 206 + isc));

                // Load the 6 quantized bytes needed by this lane.
                // Byte layout: groups 0,1 share nibbles from the same byte; 2,3 use upper nibble.
                uint ql0 = gByte(b0 + lane);          // half=0, ql[lane]
                uint ql1 = gByte(b0 + 32 + lane);     // half=0, ql[32+lane]
                uint ql2 = gByte(b0 + 64 + lane);     // half=1, ql[64+lane]
                uint ql3 = gByte(b0 + 96 + lane);     // half=1, ql[96+lane]
                uint qh0 = gByte(b0 + 128 + lane);    // half=0, qh[lane]
                uint qh1 = gByte(b0 + 160 + lane);    // half=1, qh[32+lane]

                uint base_elem = block * 256;

                acc += sc0 * float(int((ql0 & 0xF)        | (((qh0 >> 0) & 3) << 4)) - 32) * input_data[base_elem +       lane];
                acc += sc1 * float(int((ql1 & 0xF)        | (((qh0 >> 2) & 3) << 4)) - 32) * input_data[base_elem +  32 + lane];
                acc += sc2 * float(int(((ql0 >> 4) & 0xF) | (((qh0 >> 4) & 3) << 4)) - 32) * input_data[base_elem +  64 + lane];
                acc += sc3 * float(int(((ql1 >> 4) & 0xF) | (((qh0 >> 6) & 3) << 4)) - 32) * input_data[base_elem +  96 + lane];
                acc += sc4 * float(int((ql2 & 0xF)        | (((qh1 >> 0) & 3) << 4)) - 32) * input_data[base_elem + 128 + lane];
                acc += sc5 * float(int((ql3 & 0xF)        | (((qh1 >> 2) & 3) << 4)) - 32) * input_data[base_elem + 160 + lane];
                acc += sc6 * float(int(((ql2 >> 4) & 0xF) | (((qh1 >> 4) & 3) << 4)) - 32) * input_data[base_elem + 192 + lane];
                acc += sc7 * float(int(((ql3 >> 4) & 0xF) | (((qh1 >> 6) & 3) << 4)) - 32) * input_data[base_elem + 224 + lane];
            }

            sdata[tid] = acc;
            barrier();
            [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                if (lane < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            if (lane == 0)
                output_data[row] = sdata[tid];
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with F32 weights.
    /// Each workgroup computes one output row.
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=weights (float), 1=input (float), 2=output (float).
    /// </summary>
    internal const string MatVecF32 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { float weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            float acc = 0.0;
            uint base_off = row * cols;
            for (uint i = lane; i < cols; i += THREADS_PER_ROW)
                acc += weights_data[base_off + i] * input_data[i];

            sdata[tid] = acc;
            barrier();
            [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                if (lane < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            if (lane == 0)
                output_data[row] = sdata[tid];
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q4_K dequantization.
    /// Each workgroup computes one output row.
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=quantized weights (uint8), 1=input vector (float), 2=output (float).
    ///
    /// Q4_K block layout (144 bytes per 256 elements):
    ///   [0:2]   FP16 d (super-block scale)
    ///   [2:4]   FP16 dmin (super-block minimum)
    ///   [4:16]  12 bytes packed 6-bit scales/mins
    ///   [16:144] 128 bytes 4-bit quantized values
    /// </summary>
    internal const string MatVecQ4K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Register-based scale precomputation. Shared memory reduction.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        // Input aliased as vec4: with the coalesced weight read below, each lane's 4 elements are
        // 4 CONSECUTIVE floats, so they are exactly one 16-byte vec4 load instead of four scalar
        // loads. cols is always a multiple of 256, so the array divides evenly and every index
        // used here is 4-aligned by construction.
        layout(binding = 1) readonly buffer Input   { vec4 input_vec4[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        // One block's partial dot for this lane. Split out so the two-blocks-per-iteration main
        // loop and the odd-block tail share a single implementation rather than duplicating it.
        //
        // Coalesced weight read: the strided form this replaces had each lane extract ONE byte from
        // a dword shared with 3 neighbours (byte_pos >> 2 collapses 4 lanes onto one address), so
        // 32 lanes touched only 8 distinct dwords = 32 B per instruction. A Q4_K block stores its
        // 256 elements as exactly 32 nibble-dwords and a row group has exactly 32 lanes, so lane L
        // owns dword L outright: 32 lanes touch 32 distinct dwords = 128 B. The mapping is a
        // re-derivation, not an approximation — with c = lane >> 3 and j = lane & 7 the old address
        // term chunk*8 + (byte_pos >> 2) equals lane identically. Lane L's dword holds byte_pos
        // 4j..4j+3 of chunk c; each byte's low nibble is element c*64 + byte_pos (scale index c*2)
        // and its high nibble is element c*64 + 32 + byte_pos (scale c*2+1).
        //
        // qw is passed in rather than loaded here so the caller can issue several blocks' weight
        // loads back-to-back before any of them is consumed.
        float blockDot(uint word_base, uint qw, uint block, uint c, uint j) {
            vec2 dm = unpackHalf2x16(weights_data[word_base]);
            float d = dm.x;
            float dmin = dm.y;

            // Preload scale/min into registers (3 global reads instead of ~32)
            uint sm0 = weights_data[word_base + 1];
            uint sm1 = weights_data[word_base + 2];
            uint sm2 = weights_data[word_base + 3];

            // Unpack ONLY the two sub-block scales this lane uses (si = 2c and 2c+1) instead of
            // materialising all 8 pairs. Two reasons, the second being the real one:
            //   * it drops ~14 of 16 scale unpacks per lane per block, and
            //   * dsc/dmn were indexed by a RUNTIME value (c = lane >> 3), and a dynamically
            //     indexed local array is not register-addressable — AMD backs it with scratch
            //     memory, so every block iteration wrote 16 floats to scratch to read 4 back.
            // Encodings match the ggml Q4_K layout the array form spelled out:
            //   si < 4  : 6-bit scale/min packed at byte si of sm0 / sm1
            //   si >= 4 : low 4 bits at nibble (si-4) of sm2, high 2 bits from sm0 / sm1
            uint siLo = c * 2u, siHi = c * 2u + 1u;
            float sLo, mLo, sHi, mHi;
            if (siLo < 4u) {
                uint shL = siLo * 8u, shH = siHi * 8u;
                sLo = d    * float((sm0 >> shL) & 63u);
                mLo = dmin * float((sm1 >> shL) & 63u);
                sHi = d    * float((sm0 >> shH) & 63u);
                mHi = dmin * float((sm1 >> shH) & 63u);
            } else {
                uint tL = (siLo - 4u) * 8u, tH = (siHi - 4u) * 8u;
                sLo = d    * float(((sm2 >> tL)        & 0xFu) | (((sm0 >> (6u + tL)) & 3u) << 4));
                mLo = dmin * float(((sm2 >> (4u + tL)) & 0xFu) | (((sm1 >> (6u + tL)) & 3u) << 4));
                sHi = d    * float(((sm2 >> tH)        & 0xFu) | (((sm0 >> (6u + tH)) & 3u) << 4));
                mHi = dmin * float(((sm2 >> (4u + tH)) & 0xFu) | (((sm1 >> (6u + tH)) & 3u) << 4));
            }

            // (block*256 + c*64 + j*4) / 4 — the vec4 index of this lane's 4 low elements; its 4
            // high elements sit 32 floats = 8 vec4s later. With the coalesced weight read each
            // lane's 4 elements are consecutive, so they are one 16-byte load rather than four.
            uint vi = block * 64 + c * 16 + j;
            vec4 vLo = input_vec4[vi];
            vec4 vHi = input_vec4[vi + 8];

            float s = 0.0;
            [[unroll]] for (uint b = 0; b < 4; b++) {
                uint qbyte = (qw >> (b * 8)) & 0xFF;
                s += (sLo * float(qbyte & 0xFu) - mLo) * vLo[b];
                s += (sHi * float(qbyte >> 4)   - mHi) * vHi[b];
            }
            return s;
        }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;
            uint word_row_base = row * num_blocks * 36;

            // Loop-invariant: depend only on the lane, so hoist out of the block loop.
            uint c = lane >> 3;                 // Q4_K sub-chunk 0..3
            uint j = lane & 7;                  // dword within that chunk

            float acc = 0.0;

            // TWO blocks per iteration, with both weight loads issued BEFORE either is consumed.
            // At the QKV/O shape a row is only 8 blocks, so the single-block loop had one
            // outstanding weight load at a time and the wave spent most of its life waiting on
            // memory latency rather than bandwidth — the shape measured ~60% of ceiling while the
            // 4x-longer FFN rows reached 94%. Two independent chains give the memory system
            // something to overlap. (A plain [[unroll]] hint does NOT achieve this: num_blocks
            // comes from a runtime push constant, so the compiler cannot unroll the loop and
            // instead only raises register pressure — measured as a ~12% isolated LOSS.)
            uint blk = 0;
            for (; blk + 2u <= num_blocks; blk += 2u) {
                uint wb0 = word_row_base + blk * 36;
                uint wb1 = wb0 + 36;
                uint qw0 = weights_data[wb0 + 4 + lane];
                uint qw1 = weights_data[wb1 + 4 + lane];
                acc += blockDot(wb0, qw0, blk,      c, j);
                acc += blockDot(wb1, qw1, blk + 1u, c, j);
            }
            // Odd trailing block, if any.
            for (; blk < num_blocks; blk++) {
                uint wb = word_row_base + blk * 36;
                acc += blockDot(wb, weights_data[wb + 4 + lane], blk, c, j);
            }

            // Shared-memory tree reduction rather than a subgroup op: this device reports
            // minSubgroupSize == maxSubgroupSize == 64, so issue #318's requiredSubgroupSize=32
            // pin cannot apply and a plain subgroupAdd would sum ACROSS two rows (the original
            // Wave64 bug). A subgroupClusteredAdd(acc, 32) variant WAS built and measured as the
            // obvious fix for the barrier cost — it was bit-identical but gave NO speedup
            // (6.82/8.38/8.62 vs 7.07/8.61/8.60 GB/s), so these barriers are not what limits this
            // kernel. Reverted rather than carry an extra subgroup-extension requirement for
            // nothing. See docs/perf-loop-progress.md iteration 27.
            sdata[tid] = acc;
            barrier();
            [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                if (lane < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            if (lane == 0)
                output_data[row] = sdata[tid];
        }
        """;

    /// <summary>
    /// Batched (weight-stationary) matrix-vector multiply with Q4_K dequantization —
    /// the core weight-amortization for Vulkan speculative decoding (issue #308).
    ///
    /// Computes <c>nTok</c> independent matvecs against the SAME Q4_K weight matrix. The
    /// expensive part (reading + unpacking each weight nibble from VRAM) is done ONCE per
    /// output element and then multiplied into <c>nTok</c> accumulators (one per input
    /// vector), so the weight is read from VRAM once for all K tokens instead of K times.
    /// Only the per-token input reads are repeated.
    ///
    /// Bindings: 0=quantized weights (uint8), 1=inputs (float, row-major [nTok][cols]),
    /// 2=outputs (float, row-major [nTok][rows]). Push constants: { uint rows, uint cols, uint nTok }.
    ///
    /// BIT-EXACT vs nTok separate single-row <see cref="MatVecQ4K"/> calls: the element
    /// iteration order, the per-element dequant, and the shared-memory reduction are IDENTICAL
    /// to the single-row shader — only the k (token) dimension is added on top. The same
    /// floating-point accumulation order is therefore preserved per (row, token).
    ///
    /// local_size_x = 256 (8 rows × 32 lanes) using workgroup shared-memory tree reduction,
    /// robust across all subgroup sizes (Wave16, Wave32, Wave64, Wave128).
    ///
    /// nTok is capped at 8 (the acc[] register array size; matches the spec-decode draft cap).
    /// </summary>
    internal const string MatVecBatchedQ4K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Register-based scale precomputation. Shared memory reduction.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32
        #define MAX_NTOK 16

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        // Input aliased as vec4 (see MatVecQ4K): with the coalesced weight read each lane's 4
        // elements are 4 consecutive floats = one 16-byte load. cols is a multiple of 256, so
        // k*cols is 4-aligned and every index below divides evenly.
        layout(binding = 1) readonly buffer Input   { vec4 input_vec4[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;
            uint word_row_base = row * num_blocks * 36;

            float acc[MAX_NTOK];
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) acc[k] = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint word_base = word_row_base + block * 36;

                vec2 dm = unpackHalf2x16(weights_data[word_base]);
                float d = dm.x;
                float dmin = dm.y;

                // Preload scale/min into registers (3 global reads instead of ~32)
                uint sm0 = weights_data[word_base + 1];
                uint sm1 = weights_data[word_base + 2];
                uint sm2 = weights_data[word_base + 3];

                // Unpack ONLY this lane's two sub-block scales — same reasoning as MatVecQ4K:
                // dsc/dmn were indexed by the runtime value c = lane >> 3, and a dynamically
                // indexed local array is not register-addressable, so AMD backs it with scratch
                // memory (16 floats written per block to read 4 back). Measured on the single-row
                // kernel: 20.5 -> 30.5 GB/s at the FFN shape.
                uint c = lane >> 3;                 // Q4_K sub-chunk 0..3
                uint siLo = c * 2u, siHi = c * 2u + 1u;
                float sLo, mLo, sHi, mHi;
                if (siLo < 4u) {
                    uint shL = siLo * 8u, shH = siHi * 8u;
                    sLo = d    * float((sm0 >> shL) & 63u);
                    mLo = dmin * float((sm1 >> shL) & 63u);
                    sHi = d    * float((sm0 >> shH) & 63u);
                    mHi = dmin * float((sm1 >> shH) & 63u);
                } else {
                    uint tL = (siLo - 4u) * 8u, tH = (siHi - 4u) * 8u;
                    sLo = d    * float(((sm2 >> tL)        & 0xFu) | (((sm0 >> (6u + tL)) & 3u) << 4));
                    mLo = dmin * float(((sm2 >> (4u + tL)) & 0xFu) | (((sm1 >> (6u + tL)) & 3u) << 4));
                    sHi = d    * float(((sm2 >> tH)        & 0xFu) | (((sm0 >> (6u + tH)) & 3u) << 4));
                    mHi = dmin * float(((sm2 >> (4u + tH)) & 0xFu) | (((sm1 >> (6u + tH)) & 3u) << 4));
                }

                // Coalesced weight read (lane L owns dword L) PLUS vec4 input loads. Both halves
                // are required; two earlier shapes of this loop were each a ~2x end-to-end LOSS
                // (prefill 21.9 -> 10.9 t/s both times) and are worth not re-deriving:
                //   1. Coalesced weights with SCALAR input loads. Giving lane L dword L forces its
                //      elements to c*64 + j*4 + b, so scalar loads stride 4 floats across lanes;
                //      this kernel re-reads the input nTok times, so uncoalescing it dominates.
                //   2. vec4 input, but indexed INSIDE the byte loop (input_vec4[vi][b]). That
                //      reloads each vec4 four times, which costs more than the coalescing saves.
                // The version below fixes both: unpack the dword's 8 weights once, then hoist the
                // two vec4 loads OUT of the byte loop so each token costs exactly two 16-byte
                // accesses. Net +64% prefill (21.9 -> 36.0 t/s). See perf-loop iteration 28.
                uint j = lane & 7;                  // dword within that chunk
                uint qw = weights_data[word_base + 4 + lane];
                uint viBase = block * 64 + c * 16 + j;   // vec4 index within one token's row

                // Unpack this dword's 8 weights ONCE (shared across all nTok inputs), then loop
                // tokens with the input hoisted OUT of the byte loop: each token costs exactly two
                // 16-byte vec4 loads, not eight scalar loads and not four redundant vec4 reloads.
                float wLo[4], wHi[4];
                [[unroll]] for (uint b = 0; b < 4; b++) {
                    uint qbyte = (qw >> (b * 8)) & 0xFF;
                    wLo[b] = sLo * float(qbyte & 0xFu) - mLo;
                    wHi[b] = sHi * float(qbyte >> 4)   - mHi;
                }
                // Iterate to the COMPILE-TIME bound and predicate, rather than to the runtime
                // nTok. acc[] is a per-lane register array; indexing it with a runtime k makes it
                // dynamically addressed, which is not register-addressable, so the backend spills
                // the whole accumulator set to scratch memory — in the innermost loop of the
                // kernel. Unrolling on MAX_NTOK makes every acc[] index a constant. nTok is a push
                // constant, hence uniform across the workgroup, so the predicate is uniform too.
                [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) {
                    if (k >= nTok) continue;
                    uint vi = k * (cols >> 2) + viBase;
                    vec4 vLo = input_vec4[vi];
                    vec4 vHi = input_vec4[vi + 8];
                    [[unroll]] for (uint b = 0; b < 4; b++)
                        acc[k] += wLo[b] * vLo[b] + wHi[b] * vHi[b];
                }
            }

            // Same reason as the accumulation loop: a constant k keeps acc[] in registers. The
            // reduction runs for all MAX_NTOK slots so every barrier() stays in unconditional,
            // fully uniform control flow; only the store is predicated. Prefill runs at the full
            // nTok = MAX_NTOK in the common case, so the extra reductions cost nothing there, and
            // a reduction is cheap next to the block loop that feeds it.
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) {
                sdata[tid] = acc[k];
                barrier();
                [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                    if (lane < s) sdata[tid] += sdata[tid + s];
                    barrier();
                }
                if (lane == 0 && k < nTok)
                    output_data[k * rows + row] = sdata[tid];
                barrier();
            }
        }
        """;

    /// <summary>
    /// Quantize FP32 activations → Q8_1 (per 32-element sub-block), int8 path for the DP4A
    /// batched Q4_K matvec (issue #308 P0/P1). Mirrors CUDA's <c>llm_quantize_q8_1</c> exactly:
    /// per 32-element sub-block compute <c>amax = max|x|</c> over the 32 lanes, <c>d = amax/127</c>,
    /// <c>q = clamp(round(x/d), -127, 127)</c> (int8), and <c>qsum = Σq</c>. Each 32-element
    /// sub-block emits ONE 36-byte Q8_1 block:
    ///   bytes [0:2]  = fp16 d
    ///   bytes [2:4]  = fp16 (d · qsum)   (the min-bias scale `s` — only the Q4_K MMQ reads it)
    ///   bytes [4:36] = 32 × int8 quants
    /// Input is row-major <c>[nTok][cols]</c> FP32; output is row-major
    /// <c>[nTok][cols/32 × 36 bytes]</c> (one block per 32 input elements). 36 % 4 == 0, so each
    /// 36-byte block is exactly 9 word-aligned, mutually disjoint uints — the header is word 0
    /// ({d, s}) and the 32 int8 quants fill words 1..8. The output binds as a <c>uint[]</c> SSBO
    /// and every word is written PLAINLY (no atomics, no pre-zero dependency): lanes 0..7 each
    /// assemble one quant word from 4 lanes' int8s via shared memory, and lane 0 writes
    /// the header. Each output word is written by exactly one lane.
    ///
    /// local_size_x = 256 → 8 sub-blocks per workgroup, 32 lanes each, using workgroup shared memory
    /// for max/sum reductions and packing, robust across all subgroup sizes (Wave16, Wave32, Wave64).
    ///
    /// Bindings: 0 = input (float, [nTok][cols]), 1 = output (uint, Q8_1 packed bytes).
    /// Push constants: { uint rows, uint cols, uint nTok } — `rows` is unused (kept for the shared
    /// MatVecBatchedParams push-constant struct); the dispatch covers nTok·(cols/32) sub-blocks.
    /// </summary>
    internal const string QuantizeQ8_1 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        // 8 sub-blocks per workgroup, 32 lanes per sub-block = 256 threads.
        #define SUBBLOCKS_PER_WG 8

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[]; };
        layout(binding = 1) writeonly buffer Output { uint  out_data[];   };

        layout(push_constant) uniform Params {
            uint rows;   // unused (shared param struct)
            uint cols;
            uint nTok;
        };

        shared float sdata_f[256];
        shared int   sdata_i[256];
        shared uint  sdata_u[256];

        void main() {
            uint tid  = gl_LocalInvocationID.x;
            uint lane = tid & 31u;

            uint sub_blocks_per_tok = cols >> 5;            // cols / 32
            uint total_sub_blocks   = nTok * sub_blocks_per_tok;
            uint sb = gl_WorkGroupID.x * SUBBLOCKS_PER_WG + (tid >> 5);
            if (sb >= total_sub_blocks) return;

            uint tok    = sb / sub_blocks_per_tok;
            uint sb_tok = sb - tok * sub_blocks_per_tok;    // sub-block index within the token

            float val = input_data[tok * cols + sb_tok * 32u + lane];

            // amax / d / q / qsum over the 32-lane sub-block via shared memory reduction
            sdata_f[tid] = abs(val);
            barrier();
            [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                if (lane < s) sdata_f[tid] = max(sdata_f[tid], sdata_f[tid + s]);
                barrier();
            }
            float a    = sdata_f[tid & ~31u];
            float d    = a / 127.0;
            float invd = (d == 0.0) ? 0.0 : (1.0 / d);
            int   q    = clamp(int(round(val * invd)), -127, 127);

            sdata_i[tid] = q;
            barrier();
            [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                if (lane < s) sdata_i[tid] += sdata_i[tid + s];
                barrier();
            }
            int   qsum = sdata_i[tid & ~31u];

            // 36-byte Q8_1 block = 9 aligned, disjoint words: word 0 = {fp16 d, fp16 d·qsum},
            // words 1..8 = 32 int8 quants. Each output word written by exactly one lane.
            uint word_base = sb * 9u;
            uint qb = uint(q) & 0xFFu;                       // this lane's int8 quant (low byte)

            sdata_u[tid] = qb;
            barrier();

            uint sb_base = tid & ~31u;
            uint src = (lane * 4u) & 31u;                    // in-range for all lanes; only lane<8 stores
            uint b0 = sdata_u[sb_base + src + 0u];
            uint b1 = sdata_u[sb_base + src + 1u];
            uint b2 = sdata_u[sb_base + src + 2u];
            uint b3 = sdata_u[sb_base + src + 3u];
            if (lane < 8u)
                out_data[word_base + 1u + lane] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);

            if (lane == 0u) {
                uint d_bits = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                uint s_bits = packHalf2x16(vec2(d * float(qsum), 0.0)) & 0xFFFFu;
                out_data[word_base] = d_bits | (s_bits << 16u);
            }
        }
        """;

    /// <summary>
    /// Minimal correctness probe for the <c>GL_EXT_integer_dot_product</c>
    /// <c>dotPacked4x8AccSatEXT</c> intrinsic (perf-loop task #6).
    ///
    /// <para>The int8 matvec kernels replaced the intrinsic with a hand-written
    /// <c>dot4x8</c> loop because it returned materially wrong results on this codebase's
    /// AMD GCN/Vega reference driver. That workaround is unconditional, so hardware where the
    /// intrinsic <i>is</i> correct pays for a driver bug it does not have. This shader exists to
    /// answer the prerequisite question empirically: <b>is a cheap standalone probe able to detect
    /// the fault at all?</b> If a trivial dispatch of the intrinsic returns correct values on the
    /// very device whose real kernels it corrupts, then no isolated probe can gate the fast path
    /// and the gate must run the actual kernel instead.</para>
    ///
    /// <para>Bindings: 0 = operand pairs (uint, [2·count]: <c>[2i]</c> = packed weight bytes,
    /// <c>[2i+1]</c> = packed activation bytes), 1 = results (float, [count]). Push constants:
    /// { uint count }. Results are written as float because every value this probe can produce
    /// (|dot| ≤ 4·127·127 = 64516) is exactly representable, so the float round-trip is lossless
    /// and the readback can reuse the ordinary <c>Download</c> path.</para>
    ///
    /// <para>The signed×signed <c>(int, int, int)</c> overload is the one the kernels used: their
    /// weight operand holds either 4-bit nibbles (0..15) or already-biased <c>q6−32</c> bytes, and
    /// the activation operand holds signed int8 — so signed interpretation is correct for both.</para>
    /// </summary>
    internal const string IntegerDotProbe = """
        #version 450
        #extension GL_EXT_integer_dot_product : require

        layout(local_size_x = 64) in;

        layout(binding = 0) readonly buffer Ops     { uint ops[];        };
        layout(binding = 1) writeonly buffer Result { float results[];   };

        layout(push_constant) uniform Params { uint count; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= count) return;
            results[i] = float(dotPacked4x8AccSatEXT(int(ops[2u * i]), int(ops[2u * i + 1u]), 0));
        }
        """;

    /// <summary>
    /// Batched (weight-stationary) Q4_K matvec via int8-activation DP4A — the make-or-break
    /// weight-amortization for Vulkan speculative decoding (issue #308 P1). Drop-in replacement
    /// for <see cref="MatVecBatchedQ4K"/> when <c>VK_KHR_shader_integer_dot_product</c> is present;
    /// the FP variant remains the fallback. Mirrors CUDA's <c>llm_matvec_q4k_ws_n</c> exactly.
    ///
    /// The expensive per-weight work (read the Q4_K nibble word, unpack the 6-bit (sc, mn) pair,
    /// fold super_d·sc / super_dmin·mn — all token-INVARIANT) is hoisted ONCE per output element.
    /// The per-token inner cost collapses from 8 FP loads+FMAs/weight-word to: load one int8
    /// activation word + its fp16 scale, then two 4-term dot products
    ///   dot = ⟨nibbles, q_act⟩,   sum = ⟨0x01010101, q_act⟩ (the Σq min-bias),
    /// and fold the scales onto the int32 dot. Identity (per 32-element sub-block):
    ///   Σ w·a = (super_d · sc · d8) · Σ(nibble · q)  −  (super_dmin · mn · d8) · Σq.
    /// The activation is read from the Q8_1 buffer (<see cref="QuantizeQ8_1"/>), NOT FP32.
    ///
    /// LOSSY (int8 activation quant) but ARGMAX-STABLE vs <see cref="MatVecBatchedQ4K"/> — the same
    /// trade-off as the CUDA DP4A path.
    ///
    /// <para><b>Uses <c>manualDot4x8u</c>, NOT the <c>GL_EXT_integer_dot_product</c>
    /// <c>dotPacked4x8AccSatEXT</c> intrinsic.</b> The intrinsic is broken for this kernel's call
    /// pattern on this codebase's AMD GCN/Vega reference driver, exactly as it is in the sibling
    /// <see cref="MatVecBatchedQ6KInt8"/>. It was measured at <b>4-8% relative error</b> against the
    /// FP <c>MatVecQ4K</c> path at the trunk's real shapes (2048×2048, 8192×2048, 2048×8192) —
    /// versus the ~0.4% that int8 activation quantization alone should cost. Compounded across 24
    /// layers that produced completely wrong logits, which silently broke Vulkan speculative-decode
    /// verify (<c>GpuForwardPass.BatchVerifyBatched</c>) and any other <c>MatMulBatched</c> consumer.
    /// The pre-existing parity test did not catch it because its <c>maxAbs &lt; 1.0</c> tolerance on
    /// small, well-conditioned synthetic weights cannot distinguish a 4-8% relative error; the
    /// regression test added alongside this fix asserts a RELATIVE bound at the real shapes instead.</para>
    ///
    /// Lane→element layout is IDENTICAL to <see cref="MatVecBatchedQ4K"/> / the single-row
    /// MatVecQ4K (chunk = lane>>3, the 8 weight uints per chunk, sub-block 2·chunk = low nibbles,
    /// 2·chunk+1 = high nibbles), so the dp4a sum reproduces the same weight·activation pairing.
    ///
    /// Bindings: 0 = Q4_K weights (uint8), 1 = Q8_1 activations (uint, [nTok][cols/32 × 36 B]),
    /// 2 = outputs (float, row-major [nTok][rows]). Push constants: { uint rows, uint cols, uint nTok }.
    /// local_size_x = 256 (8 rows × 32 lanes) using workgroup shared-memory tree reduction,
    /// robust across all subgroup sizes (Wave16, Wave32, Wave64, Wave128).
    /// </summary>
    internal const string MatVecBatchedQ4KInt8 = Q4KInt8Head + "\n" + Q4KInt8DotManual + "\n" + Q4KInt8Body;

    /// <summary>
    /// <see cref="MatVecBatchedQ4KInt8"/> with the hand-written dot replaced by the
    /// <c>dotPacked4x8AccSatEXT</c> intrinsic — byte-for-byte the same kernel otherwise
    /// (perf-loop task #6).
    ///
    /// <para>Exists so the two can be compared on real hardware at real shapes instead of by
    /// argument. The isolated probe (<see cref="IntegerDotProbe"/>) showed that on the AMD Vega
    /// reference part the intrinsic's fault requires a sign-extended weight byte, which THIS
    /// kernel's operands (4-bit nibbles, and the <c>0x01010101</c> ones vector) can never have —
    /// so the intrinsic ought to be exactly equivalent here. Whether it actually is, inside a
    /// kernel with this one's register pressure, is a question only a measurement can answer.</para>
    /// </summary>
    internal const string MatVecBatchedQ4KInt8Dp4a = Q4KInt8Head + "\n" + Q4KInt8DotIntrinsic + "\n" + Q4KInt8Body;

    // Shared prologue. Split out so the manual and intrinsic variants cannot drift apart: the ONLY
    // difference between them is which dot4x8u body gets concatenated in the middle. Const-string
    // concatenation stays a compile-time constant, so SpirvGen's `IsLiteral` reflection still sees
    // both variants and precompiles each to SPIR-V.
    private const string Q4KInt8Head = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        """;

    // Manual 4-term signed dot — see the doc comment on MatVecBatchedQ4KInt8. Operand A holds
    // UNSIGNED payloads at every call site here (nibbles 0..15, or the ones vector), so it is NOT
    // sign-extended; operand B is the signed int8 activations, which is. That asymmetry mirrors
    // what the intrinsic's operands actually mean at these call sites.
    private const string Q4KInt8DotManual = """
        int dot4x8u(uint packedW, uint packedA) {
            int sum = 0;
            [[unroll]] for (uint t = 0u; t < 4u; t++) {
                int w = int((packedW >> (t * 8u)) & 0xFFu);
                int a = int((packedA >> (t * 8u)) & 0xFFu); if (a >= 128) a -= 256;
                sum += w * a;
            }
            return sum;
        }
        """;

    // The GL_EXT_integer_dot_product intrinsic. Sign-extends BOTH operands, which is a no-op for
    // this kernel's weight operands (all bytes < 0x80), so it is mathematically identical to the
    // manual version above at every call site in this shader.
    private const string Q4KInt8DotIntrinsic = """
        #extension GL_EXT_integer_dot_product : require

        int dot4x8u(uint packedW, uint packedA) {
            return dotPacked4x8AccSatEXT(int(packedW), int(packedA), 0);
        }
        """;

    private const string Q4KInt8Body = """
        #define N_ROWS 8
        #define THREADS_PER_ROW 32
        #define MAX_NTOK 16

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Acts    { uint act_data[];     }; // Q8_1 packed
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        shared float sdata[256];

        // Read a uint at an arbitrary BYTE offset from the (uint-typed) Q8_1 buffer. The Q8_1
        // 36-byte stride keeps every header 4-aligned, but the 4-int8 activation reads at
        // byte_off ∈ {0,4,…,28} land at base+4, which is 4-aligned too (block base is a multiple
        // of 36 → base%4 == 0). So a direct word index suffices; assert via the >>2.
        uint actWord(uint byteAddr) { return act_data[byteAddr >> 2]; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;                 // 256-element super-blocks per row
            uint word_row_base = row * num_blocks * 36u; // 36 uints per super-block

            // Q8_1 activation row stride: (cols/32) sub-blocks × 36 bytes.
            uint tok_byte_stride = (cols >> 5) * 36u;

            uint chunk     = lane >> 3;                  // 0..3
            uint byte_off  = (lane & 7u) * 4u;           // 0,4,…,28
            uint q4_offset = 4u + chunk * 8u + (lane & 7u);

            float acc[MAX_NTOK];
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) acc[k] = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint word_base = word_row_base + block * 36u;

                vec2 dm = unpackHalf2x16(weights_data[word_base]);
                float super_d    = dm.x;
                float super_dmin = dm.y;

                uint sm0 = weights_data[word_base + 1];
                uint sm1 = weights_data[word_base + 2];
                uint sm2 = weights_data[word_base + 3];

                // Unpack this lane's two 6-bit (sc, mn) pairs: lo = sub-block 2·chunk, hi = 2·chunk+1.
                uint sc_lo, mn_lo, sc_hi, mn_hi;
                if (chunk == 0u) {
                    sc_lo = (sm0)        & 63u; mn_lo = (sm1)        & 63u;
                    sc_hi = (sm0 >>  8u) & 63u; mn_hi = (sm1 >>  8u) & 63u;
                } else if (chunk == 1u) {
                    sc_lo = (sm0 >> 16u) & 63u; mn_lo = (sm1 >> 16u) & 63u;
                    sc_hi = (sm0 >> 24u) & 63u; mn_hi = (sm1 >> 24u) & 63u;
                } else if (chunk == 2u) {
                    sc_lo = (sm2         & 0xFu) | (((sm0 >>  6u) & 3u) << 4u);
                    mn_lo = ((sm2 >>  4u) & 0xFu) | (((sm1 >>  6u) & 3u) << 4u);
                    sc_hi = ((sm2 >>  8u) & 0xFu) | (((sm0 >> 14u) & 3u) << 4u);
                    mn_hi = ((sm2 >> 12u) & 0xFu) | (((sm1 >> 14u) & 3u) << 4u);
                } else {
                    sc_lo = ((sm2 >> 16u) & 0xFu) | (((sm0 >> 22u) & 3u) << 4u);
                    mn_lo = ((sm2 >> 20u) & 0xFu) | (((sm1 >> 22u) & 3u) << 4u);
                    sc_hi = ((sm2 >> 24u) & 0xFu) | (((sm0 >> 30u) & 3u) << 4u);
                    mn_hi = ((sm2 >> 28u) & 0xFu) | (((sm1 >> 30u) & 3u) << 4u);
                }

                // Load this lane's weight word once; split into 4 low + 4 high nibbles.
                uint wq    = weights_data[word_base + q4_offset];
                uint wq_lo = wq & 0x0F0F0F0Fu;          // 4 low nibbles  → sub-block 2·chunk
                uint wq_hi = (wq >> 4u) & 0x0F0F0F0Fu;  // 4 high nibbles → sub-block 2·chunk+1

                // Token-invariant folded scales (weight read amortized across all nTok tokens).
                float sd_sc_lo = super_d    * float(sc_lo);
                float sm_mn_lo = super_dmin * float(mn_lo);
                float sd_sc_hi = super_d    * float(sc_hi);
                float sm_mn_hi = super_dmin * float(mn_hi);

                // Q8_1 byte base for the two sub-blocks (within a token's activation row).
                uint q81_base_lo = (block * 8u + chunk * 2u)      * 36u;
                uint q81_base_hi = (block * 8u + chunk * 2u + 1u) * 36u;

                for (uint k = 0; k < nTok; k++) {
                    uint tok_base = k * tok_byte_stride;

                    // fp16 activation scale d8 (low 16 bits of each block header).
                    float d8_lo = unpackHalf2x16(actWord(tok_base + q81_base_lo)).x;
                    float d8_hi = unpackHalf2x16(actWord(tok_base + q81_base_hi)).x;

                    // 4 int8 activations per sub-block at byte offset (4 + byte_off).
                    uint act_lo = actWord(tok_base + q81_base_lo + 4u + byte_off);
                    uint act_hi = actWord(tok_base + q81_base_hi + 4u + byte_off);

                    // dp4a (unsigned nibbles × signed int8 acts): dot(4 nibbles, 4 int8 acts) plus
                    // Σq via dot(0x01010101, acts). dot4x8u is supplied by the concatenated variant
                    // prologue — either the hand-written loop or the dotPacked4x8AccSatEXT
                    // intrinsic — so both variants share this body verbatim.
                    int dot_lo = dot4x8u(wq_lo,       act_lo);
                    int dot_hi = dot4x8u(wq_hi,       act_hi);
                    int sum_lo = dot4x8u(0x01010101u, act_lo);
                    int sum_hi = dot4x8u(0x01010101u, act_hi);

                    acc[k] += (sd_sc_lo * d8_lo) * float(dot_lo) - (sm_mn_lo * d8_lo) * float(sum_lo);
                    acc[k] += (sd_sc_hi * d8_hi) * float(dot_hi) - (sm_mn_hi * d8_hi) * float(sum_hi);
                }
            }

            for (uint k = 0; k < nTok; k++) {
                sdata[tid] = acc[k];
                barrier();
                [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                    if (lane < s) sdata[tid] += sdata[tid + s];
                    barrier();
                }
                if (lane == 0)
                    output_data[k * rows + row] = sdata[tid];
                barrier();
            }
        }
        """;

    /// <summary>
    /// Batched (M=K) weight-stationary matrix-vector multiply with Q6_K dequantization —
    /// the Q6_K sibling of <see cref="MatVecBatchedQ4K"/>. Q4_K_M models pack most weights
    /// as Q4_K but keep ffn_down and token_embd/output as Q6_K, so the batched trunk needs
    /// a Q6_K batched matvec too (issue #308). Computes <c>output[k][row] = Σ_c W[row][c] *
    /// input[k][c]</c> for k ∈ [0, nTok). The expensive part (reading + unpacking each weight
    /// from VRAM) is done ONCE per output element and then multiplied into <c>nTok</c>
    /// accumulators (one per input vector), so the weight is read from VRAM once for all K
    /// tokens instead of K times. Only the per-token input reads are repeated.
    ///
    /// Bindings: 0=quantized weights (uint8), 1=inputs (float, row-major [nTok][cols]),
    /// 2=outputs (float, row-major [nTok][rows]). Push constants: { uint rows, uint cols, uint nTok }.
    ///
    /// BIT-EXACT vs nTok separate single-row <see cref="MatVecQ6K"/> calls: the element
    /// iteration order (the 8 explicit per-lane elements at lane, lane+32, …, lane+224), the
    /// per-element Q6_K dequant, and the shared-memory reduction are IDENTICAL to the single-row
    /// shader — only the k (token) dimension is added on top. The same floating-point
    /// accumulation order is therefore preserved per (row, token).
    ///
    /// local_size_x = 256 (8 rows × 32 lanes) using workgroup shared-memory tree reduction,
    /// robust across all subgroup sizes (Wave16, Wave32, Wave64, Wave128).
    ///
    /// nTok is capped at 8 (the acc[] register array size; matches the spec-decode draft cap).
    /// </summary>
    internal const string MatVecBatchedQ6K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Q6_K block layout (210 bytes per 256 elements):
        //   [0:128]   ql — lower 4-bit nibbles (two 64-byte halves)
        //   [128:192] qh — upper 2-bit pairs (two 32-byte halves)
        //   [192:208] 16 int8 scale values
        //   [208:210] FP16 super-block scale d
        // Thread layout: each lane handles 8 elements (lane, lane+32, ..., lane+224).
        #define N_ROWS 8
        #define THREADS_PER_ROW 32
        #define MAX_NTOK 16

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        shared float sdata[256];

        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        int  gInt8(uint b) { int v = int(gByte(b)); return v >= 128 ? v - 256 : v; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;
            uint boff_base = row * num_blocks * 210;

            float acc[MAX_NTOK];
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) acc[k] = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 210;

                float d = unpackHalf2x16(gByte(b0 + 208) | (gByte(b0 + 209) << 8)).x;

                // Precompute the 8 scale floats needed by this lane.
                // isc = lane>>4 selects lower (0) or upper (1) sub-scale row per group.
                uint isc = lane >> 4;
                float sc0 = d * float(gInt8(b0 + 192 + isc));
                float sc1 = d * float(gInt8(b0 + 194 + isc));
                float sc2 = d * float(gInt8(b0 + 196 + isc));
                float sc3 = d * float(gInt8(b0 + 198 + isc));
                float sc4 = d * float(gInt8(b0 + 200 + isc));
                float sc5 = d * float(gInt8(b0 + 202 + isc));
                float sc6 = d * float(gInt8(b0 + 204 + isc));
                float sc7 = d * float(gInt8(b0 + 206 + isc));

                // Load the 6 quantized bytes needed by this lane.
                // Byte layout: groups 0,1 share nibbles from the same byte; 2,3 use upper nibble.
                uint ql0 = gByte(b0 + lane);          // half=0, ql[lane]
                uint ql1 = gByte(b0 + 32 + lane);     // half=0, ql[32+lane]
                uint ql2 = gByte(b0 + 64 + lane);     // half=1, ql[64+lane]
                uint ql3 = gByte(b0 + 96 + lane);     // half=1, ql[96+lane]
                uint qh0 = gByte(b0 + 128 + lane);    // half=0, qh[lane]
                uint qh1 = gByte(b0 + 160 + lane);    // half=1, qh[32+lane]

                uint base_elem = block * 256;

                // Each weight value w is dequantized ONCE here, then multiplied into all nTok
                // input accumulators. Same element order + same w as single-row MatVecQ6K.
                float w0 = sc0 * float(int((ql0 & 0xF)        | (((qh0 >> 0) & 3) << 4)) - 32);
                float w1 = sc1 * float(int((ql1 & 0xF)        | (((qh0 >> 2) & 3) << 4)) - 32);
                float w2 = sc2 * float(int(((ql0 >> 4) & 0xF) | (((qh0 >> 4) & 3) << 4)) - 32);
                float w3 = sc3 * float(int(((ql1 >> 4) & 0xF) | (((qh0 >> 6) & 3) << 4)) - 32);
                float w4 = sc4 * float(int((ql2 & 0xF)        | (((qh1 >> 0) & 3) << 4)) - 32);
                float w5 = sc5 * float(int((ql3 & 0xF)        | (((qh1 >> 2) & 3) << 4)) - 32);
                float w6 = sc6 * float(int(((ql2 >> 4) & 0xF) | (((qh1 >> 4) & 3) << 4)) - 32);
                float w7 = sc7 * float(int(((ql3 >> 4) & 0xF) | (((qh1 >> 6) & 3) << 4)) - 32);

                for (uint k = 0; k < nTok; k++) {
                    uint in_base = k * cols + base_elem + lane;
                    acc[k] += w0 * input_data[in_base];
                    acc[k] += w1 * input_data[in_base +  32];
                    acc[k] += w2 * input_data[in_base +  64];
                    acc[k] += w3 * input_data[in_base +  96];
                    acc[k] += w4 * input_data[in_base + 128];
                    acc[k] += w5 * input_data[in_base + 160];
                    acc[k] += w6 * input_data[in_base + 192];
                    acc[k] += w7 * input_data[in_base + 224];
                }
            }

            for (uint k = 0; k < nTok; k++) {
                sdata[tid] = acc[k];
                barrier();
                [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                    if (lane < s) sdata[tid] += sdata[tid + s];
                    barrier();
                }
                if (lane == 0)
                    output_data[k * rows + row] = sdata[tid];
                barrier();
            }
        }
        """;

    /// <summary>
    /// Batched (weight-stationary) Q6_K matvec via int8-activation DP4A — the Q6_K sibling of
    /// <see cref="MatVecBatchedQ4KInt8"/> (issue #308 P2). Q4_K_M models keep ffn_down and
    /// token_embd/output as Q6_K, so the Q4_K-only int8 path of P1 left ~⅓ of the trunk on the
    /// slow FP <see cref="MatVecBatchedQ6K"/>; this shader pushes Q6_K onto the same DP4A path so
    /// the WHOLE spec-decode trunk amortizes the weight read across all nTok draft tokens. Drop-in
    /// replacement for <see cref="MatVecBatchedQ6K"/> when <c>VK_KHR_shader_integer_dot_product</c>
    /// is present; the FP variant remains the fallback. Mirrors CUDA's Q6_K decode-MMQ int8 dot.
    ///
    /// The expensive per-weight work (read the ql/qh bytes, reconstruct the 6-bit quant, fold the
    /// int8 sub-scale and super-block d — all token-INVARIANT) is hoisted ONCE per output element.
    /// The per-token inner cost collapses to: load one int8 activation word + its fp16 scale, then
    /// one 4-term signed dot product via <c>manualDot4x8</c> below. Q6_K has NO min/dmin term
    /// (unlike Q4_K), so the identity is simpler — no Σq bias, no 0x01010101 dot:
    ///   Σ w·a = (d · scale · d8) · Σ((q6 − 32) · q8)   over each group of 4 elements.
    /// The activation is read from the SAME Q8_1 buffer as the Q4_K int8 path
    /// (<see cref="QuantizeQ8_1"/>) — Q6_K reuses the identical int8 activations, no new quant.
    ///
    /// LOSSY (int8 activation quant) but ARGMAX-STABLE vs <see cref="MatVecBatchedQ6K"/> — the same
    /// trade-off as the CUDA DP4A path and the Q4_K int8 sibling. Spec-decode verify accepts on
    /// argmax, so greedy spec stays lossless; the parity test relaxes to argmax-match + maxAbs &lt; 1.0.
    ///
    /// The int8 weight is <c>(q6 − 32) ∈ [−32, 31]</c>, which fits signed int8 — packed 4 per uint
    /// for the signed dp4a. Lane→element layout: each lane owns 8 CONTIGUOUS columns
    /// <c>lane·8 .. lane·8+7</c> of the 256-element super-block (32 lanes × 8 = 256), split into two
    /// dp4a groups of 4 contiguous columns. Each group lands wholly inside one 32-element Q8_1
    /// sub-block (its 4 int8 activations are one aligned word) and inside one 16-element Q6_K scale
    /// group (scale index <c>lane/2</c>, shared by both groups of the lane). This differs from the
    /// FP shader's strided per-lane element order, but it pairs each weight column with its OWN
    /// activation column — the products Σ w[c]·a[c] are identical (only the FP reduction order
    /// changes, which argmax-stability permits). The per-column (q6 − 32) reconstruction reuses the
    /// exact ql/qh nibble + qh-pair-shift recipe of <see cref="MatVecBatchedQ6K"/> / MatVecQ6K
    /// (column c → l = c%32, j = c/32), so the dequantized quant matches the FP path bit-for-bit.
    ///
    /// Bindings: 0 = Q6_K weights (uint8), 1 = Q8_1 activations (uint, [nTok][cols/32 × 36 B]),
    /// 2 = outputs (float, row-major [nTok][rows]). Push constants: { uint rows, uint cols, uint nTok }.
    /// local_size_x = 256 (8 rows × 32 lanes) using workgroup shared-memory tree reduction,
    /// robust across all subgroup sizes (Wave16, Wave32, Wave64, Wave128).
    ///
    /// Uses a manual 4-term scalar dot (<c>manualDot4x8</c>) instead of the
    /// <c>GL_EXT_integer_dot_product</c> <c>dotPacked4x8AccSatEXT</c> intrinsic. On this codebase's
    /// AMD GCN/Vega reference hardware, the intrinsic produced wildly wrong results for this
    /// kernel's call pattern specifically — confirmed by a hand-computed C# reference
    /// reimplementation matching the FP path to within int8 quant noise (~0.1 abs on values in the
    /// hundreds) while the intrinsic's output differed by 10-40%. The sibling
    /// <see cref="MatVecBatchedQ4KInt8"/> kernel calls the same intrinsic four times with a
    /// different operand pattern. It was ORIGINALLY BELIEVED UNAFFECTED because its parity test
    /// passed — that conclusion was WRONG. The test's <c>maxAbs &lt; 1.0</c> tolerance on small,
    /// well-conditioned synthetic weights cannot see a relative error, and re-measuring at the real
    /// trunk shapes showed 4-8% relative error there too. Both kernels now use a manual dot.
    /// Treat <c>dotPacked4x8AccSatEXT</c> as UNUSABLE on this driver rather than as a
    /// per-call-site quirk, and gate any future use behind a relative-error test at production
    /// shapes, not an absolute tolerance on a toy matrix.
    /// </summary>
    /// <summary>
    /// Tiled Q4_K matrix-MATRIX multiply (Path 2) — both operands staged in shared memory.
    /// </summary>
    /// <remarks>
    /// <para><b>Why this exists.</b> <see cref="MatVecBatchedQ4K"/> is a matrix-<i>vector</i> kernel
    /// with register blocking over tokens: <c>acc[MAX_NTOK]</c> is one VGPR per token, which caps it
    /// at 16 and forces prefill to re-stream the whole model every 16 tokens. Measured on a
    /// 931-token prompt: 53.3 GiB of weight traffic for a 1 GiB model, 59 passes. Sweeping the chunk
    /// size fits <c>time ≈ weight_GiB/12.33 + 8.04s</c>, so weights are ~35% of prefill and the
    /// ceiling from removing them entirely is 1.54x.</para>
    ///
    /// <para><b>Shape follows llama.cpp's <c>mul_mm.comp</c>.</b> That backend switches from matvec
    /// to a tiled GEMM at <c>n &gt; 8</c> (<c>mul_mat_vec_max_cols = 8</c>, ggml-vulkan.cpp:9771), so
    /// all of its prefill takes this shape and none of it takes ours. It stages both operands
    /// (<c>buf_a</c>/<c>buf_b</c>), dequantizes the quantized operand into shared memory once, and
    /// computes a per-thread <c>TM x TN</c> register tile. The <c>+1</c> row padding below is its
    /// <c>SHMEM_STRIDE = BK/2 + 1</c> trick: without it every thread in the M direction reads
    /// <c>r*BK + kk</c> with <c>BK</c> a multiple of 32, so all rows land in the same LDS bank.</para>
    ///
    /// <para><b>Tiling.</b> BM=64 rows x BN=16 tokens per workgroup, stepped BK=64 elements at a
    /// time. BK=64 is not arbitrary: it is exactly one Q4_K "c-chunk" — 8 dwords whose low nibbles
    /// are sub-block 2c and whose high nibbles are sub-block 2c+1, sharing one pair of 6-bit
    /// scale/min values. That makes the dequant in the staging loop identical in structure to the
    /// incumbent kernel's, so the two can be compared without a second variable.</para>
    ///
    /// <para><b>Not bit-exact vs Path 1</b>, and cannot be: the K-accumulation order differs (one
    /// running sum per output here, versus a per-lane partial sum plus a shared-memory tree
    /// reduction there). Path 1 remains the default until this is measured better end-to-end AND no
    /// worse on perplexity — the same bar the CPU's Path 2 had to clear.</para>
    ///
    /// <para>Bindings: 0 = Q4_K weights (uint), 1 = inputs (vec4, row-major [nTok][cols]),
    /// 2 = outputs (float, row-major [nTok][rows]). Push constants { rows, cols, nTok }.
    /// Dispatch: ceil(rows / BM) workgroups. Requires nTok &lt;= BN and cols % 256 == 0.</para>
    /// </remarks>
    internal const string MatMulTiledQ4K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define BM 64        // output rows per workgroup
        #define BN 16        // tokens per workgroup (must be >= nTok)
        #define BK 64        // elements per K-step = one Q4_K c-chunk (2 sub-blocks)
        #define TM 2         // rows per thread
        #define TN 2         // tokens per thread (BN/TN must be 256/(BM/TM) = 8)
        #define STRIDE (BK + 1)   // +1: LDS bank-conflict padding (see mul_mm.comp SHMEM_STRIDE)

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { vec4 input_vec4[]; };
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        shared float buf_a[BM * STRIDE];   // dequantized weights [row][k]
        shared float buf_b[BN * STRIDE];   // activations         [token][k]

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_base = gl_WorkGroupID.x * BM;

            uint num_blocks = cols >> 8;          // 256-element Q4_K blocks
            uint num_ksteps = num_blocks << 2;    // 4 c-chunks of 64 per block

            // 32 threads span M (64/TM), 8 span N (16/TN) = 256.
            uint tm_i = tid % (BM / TM);
            uint tn_i = tid / (BM / TM);

            float acc[TM][TN];
            [[unroll]] for (uint i = 0; i < TM; i++)
                [[unroll]] for (uint j = 0; j < TN; j++) acc[i][j] = 0.0;

            for (uint ks = 0; ks < num_ksteps; ks++) {
                uint block = ks >> 2;
                uint c     = ks & 3u;

                // ---- stage A: BM rows x BK dequantized weights (512 (row,dword) pairs / 256 thr) ----
                [[unroll]] for (uint pass = 0; pass < 2u; pass++) {
                    uint t = tid + pass * 256u;
                    uint r = t >> 3;              // row within the tile, 0..63
                    uint j = t & 7u;              // dword within the c-chunk, 0..7
                    uint row = row_base + r;
                    float sLo = 0.0, mLo = 0.0, sHi = 0.0, mHi = 0.0;
                    uint qw = 0u;
                    if (row < rows) {
                        uint wb = row * num_blocks * 36u + block * 36u;
                        vec2 dm = unpackHalf2x16(weights_data[wb]);
                        float d = dm.x, dmin = dm.y;
                        uint sm0 = weights_data[wb + 1u];
                        uint sm1 = weights_data[wb + 2u];
                        uint sm2 = weights_data[wb + 3u];
                        // Same 6-bit scale/min unpack as MatVecBatchedQ4K, for sub-blocks 2c / 2c+1.
                        uint siLo = c * 2u, siHi = c * 2u + 1u;
                        if (siLo < 4u) {
                            uint shL = siLo * 8u, shH = siHi * 8u;
                            sLo = d    * float((sm0 >> shL) & 63u);
                            mLo = dmin * float((sm1 >> shL) & 63u);
                            sHi = d    * float((sm0 >> shH) & 63u);
                            mHi = dmin * float((sm1 >> shH) & 63u);
                        } else {
                            uint tL = (siLo - 4u) * 8u, tH = (siHi - 4u) * 8u;
                            sLo = d    * float(((sm2 >> tL)        & 0xFu) | (((sm0 >> (6u + tL)) & 3u) << 4));
                            mLo = dmin * float(((sm2 >> (4u + tL)) & 0xFu) | (((sm1 >> (6u + tL)) & 3u) << 4));
                            sHi = d    * float(((sm2 >> tH)        & 0xFu) | (((sm0 >> (6u + tH)) & 3u) << 4));
                            mHi = dmin * float(((sm2 >> (4u + tH)) & 0xFu) | (((sm1 >> (6u + tH)) & 3u) << 4));
                        }
                        qw = weights_data[wb + 4u + c * 8u + j];
                    }
                    // Rows past the end stage zeros, so the compute loop needs no bounds test.
                    [[unroll]] for (uint b = 0; b < 4u; b++) {
                        uint qb = (qw >> (b * 8u)) & 0xFFu;
                        buf_a[r * STRIDE + j * 4u + b]        = sLo * float(qb & 0xFu) - mLo;
                        buf_a[r * STRIDE + 32u + j * 4u + b]  = sHi * float(qb >> 4)   - mHi;
                    }
                }

                // ---- stage B: BN tokens x BK activations (BN*16 vec4 over 256 threads) ----
                // BN=32 needs two passes; at BN=16 this was exactly one. Tokens past nTok stage
                // zeros so the compute loop needs no bounds test.
                [[unroll]] for (uint pass = 0; pass < (BN * 16u) / 256u; pass++) {
                    uint t = tid + pass * 256u;
                    uint k = t >> 4;              // token, 0..BN-1
                    uint v = t & 15u;             // vec4 within the 64-element chunk
                    vec4 val = (k < nTok) ? input_vec4[k * (cols >> 2) + ks * 16u + v] : vec4(0.0);
                    uint o = k * STRIDE + v * 4u;
                    buf_b[o]      = val.x;
                    buf_b[o + 1u] = val.y;
                    buf_b[o + 2u] = val.z;
                    buf_b[o + 3u] = val.w;
                }

                barrier();

                [[unroll]] for (uint kk = 0; kk < BK; kk++) {
                    float a[TM], bv[TN];
                    [[unroll]] for (uint i = 0; i < TM; i++) a[i]  = buf_a[(tm_i * TM + i) * STRIDE + kk];
                    [[unroll]] for (uint j = 0; j < TN; j++) bv[j] = buf_b[(tn_i * TN + j) * STRIDE + kk];
                    [[unroll]] for (uint i = 0; i < TM; i++)
                        [[unroll]] for (uint j = 0; j < TN; j++) acc[i][j] += a[i] * bv[j];
                }

                barrier();
            }

            [[unroll]] for (uint i = 0; i < TM; i++) {
                uint row = row_base + tm_i * TM + i;
                if (row < rows) {
                    [[unroll]] for (uint j = 0; j < TN; j++) {
                        uint k = tn_i * TN + j;
                        if (k < nTok) output_data[k * rows + row] = acc[i][j];
                    }
                }
            }
        }
        """;

    /// <summary>
    /// Tiled Q6_K GEMM — the Q6_K twin of <see cref="MatMulTiledQ4K"/>, same BM/BN/BK/TM/TN and the
    /// same LDS layout, differing only in the dequant that fills <c>buf_a</c>.
    /// <para>Exists to stop Path 2 declining <c>ffn_down</c> (Q6_K, 1 dispatch per layer). While it
    /// declined, every raise of the token tile was capped by the Path 1 fallback's nTok &lt;= 16
    /// throw, so BN could not grow past 16 and the tiled kernel amortized no more weight traffic
    /// than the matvec it replaced.</para>
    /// <para><b>BK=64 lands exactly on a Q6_K scale-group pair.</b> Element <c>lane + 32*j</c>
    /// (lane 0..31, j 0..7) means group j owns the contiguous run [32j, 32j+32), so a 64-element
    /// k-step c covers groups 2c and 2c+1 — no group is ever split across a k-step, and each thread
    /// needs one ql byte pair, one qh byte and two scales. The Q4_K kernel gets the same property
    /// from its c-chunks; that the two layouts agree at BK=64 is why one tile shape serves both.</para>
    /// Push constants: { uint rows, uint cols, uint nTok }.
    /// Bindings: 0=weights (Q6_K), 1=activations (nTok x cols, fp32), 2=output (nTok x rows).
    /// </summary>
    internal const string MatMulTiledQ6K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define BM 64        // output rows per workgroup
        #define BN 16        // tokens per workgroup (must be >= nTok)
        #define BK 64        // elements per K-step = one Q6_K scale-group PAIR
        #define TM 2         // rows per thread
        #define TN 2         // tokens per thread (BN/TN must be 256/(BM/TM) = 8)
        #define STRIDE (BK + 1)   // +1: LDS bank-conflict padding (see mul_mm.comp SHMEM_STRIDE)

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { vec4 input_vec4[]; };
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        shared float buf_a[BM * STRIDE];   // dequantized weights [row][k]
        shared float buf_b[BN * STRIDE];   // activations         [token][k]

        // Q6_K rows are 210 bytes and therefore NOT dword-aligned; every access goes through a
        // byte gather. Same helpers as MatVecBatchedQ6K so the two cannot drift.
        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        int  gInt8(uint b) { int v = int(gByte(b)); return v >= 128 ? v - 256 : v; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_base = gl_WorkGroupID.x * BM;

            uint num_blocks = cols >> 8;          // 256-element Q6_K blocks
            uint num_ksteps = num_blocks << 2;    // 4 group-pairs of 64 per block

            uint tm_i = tid % (BM / TM);
            uint tn_i = tid / (BM / TM);

            float acc[TM][TN];
            [[unroll]] for (uint i = 0; i < TM; i++)
                [[unroll]] for (uint j = 0; j < TN; j++) acc[i][j] = 0.0;

            for (uint ks = 0; ks < num_ksteps; ks++) {
                uint block = ks >> 2;
                uint c     = ks & 3u;
                uint half_ = c >> 1;              // 0 = ql[0:64]/qh[0:32], 1 = ql[64:128]/qh[32:64]
                uint nib   = c & 1u;              // 0 = low nibble, 1 = high nibble
                uint nsh   = nib * 4u;

                // ---- stage A: BM rows x BK dequantized weights (2048 (row,lane) pairs / 256 thr) ----
                // Each (row,lane) yields TWO values: group 2c at position `lane` and group 2c+1 at
                // position `lane + 32`, which is exactly the [0,64) k-step window.
                [[unroll]] for (uint pass = 0; pass < 8u; pass++) {
                    uint t = tid + pass * 256u;
                    uint r = t >> 5;              // row within the tile, 0..63
                    uint lane = t & 31u;          // 0..31
                    uint row = row_base + r;
                    float wLo = 0.0, wHi = 0.0;
                    if (row < rows) {
                        uint b0 = row * num_blocks * 210u + block * 210u;
                        float d = unpackHalf2x16(gByte(b0 + 208u) | (gByte(b0 + 209u) << 8)).x;
                        uint isc = lane >> 4;     // lower/upper sub-scale row, as in MatVecBatchedQ6K
                        float scLo = d * float(gInt8(b0 + 192u + c * 4u + isc));
                        float scHi = d * float(gInt8(b0 + 192u + c * 4u + 2u + isc));

                        uint qlA = gByte(b0 + half_ * 64u + lane);
                        uint qlB = gByte(b0 + half_ * 64u + 32u + lane);
                        uint qh  = gByte(b0 + 128u + half_ * 32u + lane);

                        wLo = scLo * float(int(((qlA >> nsh) & 0xFu) | (((qh >> nsh)        & 3u) << 4)) - 32);
                        wHi = scHi * float(int(((qlB >> nsh) & 0xFu) | (((qh >> (nsh + 2u)) & 3u) << 4)) - 32);
                    }
                    // Rows past the end stage zeros, so the compute loop needs no bounds test.
                    buf_a[r * STRIDE + lane]        = wLo;
                    buf_a[r * STRIDE + 32u + lane]  = wHi;
                }

                // ---- stage B: BN tokens x BK activations (BN*16 vec4 over 256 threads) ----
                // BN=32 needs two passes; at BN=16 this was exactly one. Tokens past nTok stage
                // zeros so the compute loop needs no bounds test.
                [[unroll]] for (uint pass = 0; pass < (BN * 16u) / 256u; pass++) {
                    uint t = tid + pass * 256u;
                    uint k = t >> 4;              // token, 0..BN-1
                    uint v = t & 15u;             // vec4 within the 64-element chunk
                    vec4 val = (k < nTok) ? input_vec4[k * (cols >> 2) + ks * 16u + v] : vec4(0.0);
                    uint o = k * STRIDE + v * 4u;
                    buf_b[o]      = val.x;
                    buf_b[o + 1u] = val.y;
                    buf_b[o + 2u] = val.z;
                    buf_b[o + 3u] = val.w;
                }

                barrier();

                [[unroll]] for (uint kk = 0; kk < BK; kk++) {
                    float a[TM], bv[TN];
                    [[unroll]] for (uint i = 0; i < TM; i++) a[i]  = buf_a[(tm_i * TM + i) * STRIDE + kk];
                    [[unroll]] for (uint j = 0; j < TN; j++) bv[j] = buf_b[(tn_i * TN + j) * STRIDE + kk];
                    [[unroll]] for (uint i = 0; i < TM; i++)
                        [[unroll]] for (uint j = 0; j < TN; j++) acc[i][j] += a[i] * bv[j];
                }

                barrier();
            }

            [[unroll]] for (uint i = 0; i < TM; i++) {
                uint row = row_base + tm_i * TM + i;
                if (row < rows) {
                    [[unroll]] for (uint j = 0; j < TN; j++) {
                        uint k = tn_i * TN + j;
                        if (k < nTok) output_data[k * rows + row] = acc[i][j];
                    }
                }
            }
        }
        """;

    internal const string MatVecBatchedQ6KInt8 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Q6_K block layout (210 bytes per 256 elements):
        //   [0:128]   ql — lower 4-bit nibbles (two 64-byte halves)
        //   [128:192] qh — upper 2-bit pairs (two 32-byte halves)
        //   [192:208] 16 int8 scale values
        //   [208:210] FP16 super-block scale d
        // Lane layout: each lane owns 8 contiguous columns lane*8 .. lane*8+7, split into
        // two dp4a groups of 4 contiguous columns. Both groups share scale index lane/2.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32
        #define MAX_NTOK 16

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Acts    { uint act_data[];     }; // Q8_1 packed
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        shared float sdata[256];

        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        // Read a uint at a 4-aligned BYTE offset of the (uint-typed) Q8_1 buffer (see Q4K int8).
        uint actWord(uint byteAddr) { return act_data[byteAddr >> 2]; }

        int manualDot4x8(uint packedW, uint packedA) {
            int sum = 0;
            [[unroll]] for (uint t = 0u; t < 4u; t++) {
                int w = int((packedW >> (t * 8u)) & 0xFFu); if (w >= 128) w -= 256;
                int a = int((packedA >> (t * 8u)) & 0xFFu); if (a >= 128) a -= 256;
                sum += w * a;
            }
            return sum;
        }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;                 // 256-element super-blocks per row
            uint boff_base  = row * num_blocks * 210u;   // Q6_K weights are byte-addressed (210 B/block)

            // Q8_1 activation row stride: (cols/32) sub-blocks × 36 bytes.
            uint tok_byte_stride = (cols >> 5) * 36u;

            // This lane's 8 contiguous columns within a block: groupA = base0..+3, groupB = base1..+3.
            uint base0   = lane * 8u;        // first column of group A within the block
            uint base1   = base0 + 4u;       // first column of group B
            uint isc     = lane >> 1;        // shared Q6_K scale index (lane/2) for both groups

            // Group A column j = base0/32; group B column j = base1/32 (constant within a 4-group).
            uint jA = base0 >> 5;            // 0..7
            uint jB = base1 >> 5;
            uint lA = base0 & 31u;           // first of 4 consecutive ql/qh lanes
            uint lB = base1 & 31u;

            // ql byte base + high-nibble flag + qh byte base + qh 2-bit shift, per j (mirrors the
            // MatVecBatchedQ6K per-element extraction; see /tmp derivation in PR notes).
            //   j: 0->(ql 0,  lo, qh 128, sh0) 1->(ql 32, lo, qh 128, sh2) 2->(ql 0,  hi, qh 128, sh4)
            //      3->(ql 32, hi, qh 128, sh6) 4->(ql 64, lo, qh 160, sh0) 5->(ql 96, lo, qh 160, sh2)
            //      6->(ql 64, hi, qh 160, sh4) 7->(ql 96, hi, qh 160, sh6)
            uint qlbaseA = ((jA & 1u) == 0u) ? ((jA < 4u) ? 0u : 64u) : ((jA < 4u) ? 32u : 96u);
            uint qlbaseB = ((jB & 1u) == 0u) ? ((jB < 4u) ? 0u : 64u) : ((jB < 4u) ? 32u : 96u);
            bool hiA = (jA == 2u) || (jA == 3u) || (jA == 6u) || (jA == 7u);
            bool hiB = (jB == 2u) || (jB == 3u) || (jB == 6u) || (jB == 7u);
            uint qhbaseA = (jA < 4u) ? 128u : 160u;
            uint qhbaseB = (jB < 4u) ? 128u : 160u;
            uint qhshA = (jA & 3u) * 2u;     // 0,2,4,6
            uint qhshB = (jB & 3u) * 2u;

            // Q8_1 byte base for each group's sub-block + the 4-int8 word offset within it.
            uint subA      = base0 >> 5;     // == jA (sub-block index within the block)
            uint subB      = base1 >> 5;
            uint wordOffA  = (base0 & 31u);  // int8 position of the 4-group within its sub-block (0,4,..,28)
            uint wordOffB  = (base1 & 31u);

            float acc[MAX_NTOK];
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) acc[k] = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 210u;

                float d = unpackHalf2x16(gByte(b0 + 208u) | (gByte(b0 + 209u) << 8u)).x;
                int   sc = int(gByte(b0 + 192u + isc));
                sc = (sc >= 128) ? sc - 256 : sc;           // int8 sub-scale
                float dsc = d * float(sc);                   // token-invariant folded scale (both groups)

                // Reconstruct the 4 int8 weights (q6 − 32) ∈ [−32,31] for each group, pack 4/int.
                uint wpackA = 0u, wpackB = 0u;
                [[unroll]] for (uint t = 0u; t < 4u; t++) {
                    uint qlA = gByte(b0 + qlbaseA + lA + t);
                    uint qhA = gByte(b0 + qhbaseA + lA + t);
                    int  q6A = int((hiA ? ((qlA >> 4u) & 0xFu) : (qlA & 0xFu)) | (((qhA >> qhshA) & 3u) << 4u)) - 32;
                    wpackA |= (uint(q6A) & 0xFFu) << (t * 8u);

                    uint qlB = gByte(b0 + qlbaseB + lB + t);
                    uint qhB = gByte(b0 + qhbaseB + lB + t);
                    int  q6B = int((hiB ? ((qlB >> 4u) & 0xFu) : (qlB & 0xFu)) | (((qhB >> qhshB) & 3u) << 4u)) - 32;
                    wpackB |= (uint(q6B) & 0xFFu) << (t * 8u);
                }

                // Q8_1 byte base for the two sub-blocks (within a token's activation row).
                uint q81_base_A = (block * 8u + subA) * 36u;
                uint q81_base_B = (block * 8u + subB) * 36u;

                for (uint k = 0; k < nTok; k++) {
                    uint tok_base = k * tok_byte_stride;

                    // fp16 activation scale d8 (low 16 bits of each sub-block header).
                    float d8A = unpackHalf2x16(actWord(tok_base + q81_base_A)).x;
                    float d8B = unpackHalf2x16(actWord(tok_base + q81_base_B)).x;

                    // 4 int8 activations per group at byte offset (4 + wordOff).
                    uint actA = actWord(tok_base + q81_base_A + 4u + wordOffA);
                    uint actB = actWord(tok_base + q81_base_B + 4u + wordOffB);

                    // Signed×signed dp4a: Σ((q6−32)·q8). NO min term (Q6_K has no dmin). The Sat
                    // overload is inert here (|partial| ≤ 4·32·127 = 16256, far below int32 overflow).
                    int dotA = manualDot4x8(wpackA, actA);
                    int dotB = manualDot4x8(wpackB, actB);

                    acc[k] += (dsc * d8A) * float(dotA) + (dsc * d8B) * float(dotB);
                }
            }

            for (uint k = 0; k < nTok; k++) {
                sdata[tid] = acc[k];
                barrier();
                [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                    if (lane < s) sdata[tid] += sdata[tid + s];
                    barrier();
                }
                if (lane == 0)
                    output_data[k * rows + row] = sdata[tid];
                barrier();
            }
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q5_K dequantization.
    /// Each workgroup computes 8 output rows (8 rows × 32 lanes = 256 threads).
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=quantized weights (uint8), 1=input vector (float), 2=output (float).
    ///
    /// Q5_K block layout (176 bytes per 256 elements):
    ///   [0:2]     FP16 d (super-block scale)
    ///   [2:4]     FP16 dmin (super-block minimum)
    ///   [4:16]    12 bytes packed 6-bit (scale, min) pairs (8 pairs, same packing as Q4_K)
    ///   [16:48]   qh[32] — high bit per element (one bit, 8 polarities × 32 lanes)
    ///   [48:176]  ql[128] — lower 4 bits, two elements per byte
    /// Dequant per chunk c∈0..3, lane l∈0..31 (matches CPU DequantQ5K / CUDA llm_matvec_q5k):
    ///   y[64c+l]    = d*sc[2c]  * ((ql[32c+l]&amp;0xF) + (qh[l]&amp;(1&lt;&lt;2c)   ?16:0)) - dmin*m[2c]
    ///   y[64c+l+32] = d*sc[2c+1]* ((ql[32c+l]>>4)  + (qh[l]&amp;(1&lt;&lt;(2c+1))?16:0)) - dmin*m[2c+1]
    /// The 6-bit (scale, min) unpack reuses the exact Q4_K logic (Q5_K packs scales
    /// identically); Q5_K only adds the qh high bit (+16) per quant. The super-block
    /// d/dmin and the 12 scale/min bytes occupy bytes [0:16] of each 176-byte block,
    /// which is 4-byte aligned, so they're read as four aligned uint words (like
    /// MatVecQ4K); the per-lane qh/ql bytes are byte-granular and use the byte-gather
    /// helper. Mirrors the CUDA llm_matvec_q5k kernel and the CPU DequantQ5K path.
    /// </summary>
    internal const string MatVecQ5K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;            // cols / 256
            uint boff_base = row * num_blocks * 176;

            float acc = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 176;

                // b0 is always a multiple of 176 (hence 4-byte aligned), so the first
                // 16 bytes (d/dmin + 12 scale/min bytes) read as four aligned uint words,
                // exactly like MatVecQ4K — 4 global reads instead of 16 gByte gathers.
                uint word_base = b0 >> 2;
                vec2 dm = unpackHalf2x16(weights_data[word_base]);
                float d    = dm.x;
                float dmin = dm.y;

                // 12 packed scale/min bytes at b0+4 (identical packing to Q4_K).
                // sm0 = scales[0..3], sm1 = scales[4..7], sm2 = scales[8..11].
                uint sm0 = weights_data[word_base + 1];
                uint sm1 = weights_data[word_base + 2];
                uint sm2 = weights_data[word_base + 3];

                float dsc[8], dmn[8];
                dsc[0] = d * float((sm0) & 63);         dmn[0] = dmin * float((sm1) & 63);
                dsc[1] = d * float((sm0 >> 8) & 63);    dmn[1] = dmin * float((sm1 >> 8) & 63);
                dsc[2] = d * float((sm0 >> 16) & 63);   dmn[2] = dmin * float((sm1 >> 16) & 63);
                dsc[3] = d * float((sm0 >> 24) & 63);   dmn[3] = dmin * float((sm1 >> 24) & 63);
                dsc[4] = d * float((sm2 & 0xF) | (((sm0 >> 6) & 3) << 4));
                dmn[4] = dmin * float(((sm2 >> 4) & 0xF) | (((sm1 >> 6) & 3) << 4));
                dsc[5] = d * float(((sm2 >> 8) & 0xF) | (((sm0 >> 14) & 3) << 4));
                dmn[5] = dmin * float(((sm2 >> 12) & 0xF) | (((sm1 >> 14) & 3) << 4));
                dsc[6] = d * float(((sm2 >> 16) & 0xF) | (((sm0 >> 22) & 3) << 4));
                dmn[6] = dmin * float(((sm2 >> 20) & 0xF) | (((sm1 >> 22) & 3) << 4));
                dsc[7] = d * float(((sm2 >> 24) & 0xF) | (((sm0 >> 30) & 3) << 4));
                dmn[7] = dmin * float(((sm2 >> 28) & 0xF) | (((sm1 >> 30) & 3) << 4));

                // High bit for this lane: one qh byte per lane (qh[lane]), bits 2c / 2c+1
                // select the +16 polarity for chunk c low/high nibble respectively.
                uint qh_byte = gByte(b0 + 16 + lane);
                uint base_elem = block * 256;

                [[unroll]] for (uint c = 0; c < 4; c++) {
                    uint ql_byte = gByte(b0 + 48 + c * 32 + lane);
                    uint low4 = ql_byte & 0xF;
                    uint hi4  = (ql_byte >> 4) & 0xF;

                    uint u1 = 1u << (2u * c);
                    uint u2 = u1 << 1;
                    float hLo = (qh_byte & u1) != 0u ? 16.0 : 0.0;
                    float hHi = (qh_byte & u2) != 0u ? 16.0 : 0.0;

                    uint si = 2u * c;
                    uint elem_lo = base_elem + c * 64 + lane;
                    acc += (dsc[si]     * (float(low4) + hLo) - dmn[si])     * input_data[elem_lo];
                    acc += (dsc[si + 1] * (float(hi4)  + hHi) - dmn[si + 1]) * input_data[elem_lo + 32];
                }
            }

            sdata[tid] = acc;
            barrier();
            [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                if (lane < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            if (lane == 0)
                output_data[row] = sdata[tid];
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q8_0 dequantization.
    /// Each workgroup computes 8 output rows (8 rows × 32 lanes = 256 threads).
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=quantized weights (uint8), 1=input vector (float), 2=output (float).
    ///
    /// Q8_0 block layout (34 bytes per 32 elements):
    ///   [0:2]  FP16 d (block scale)
    ///   [2:34] 32 int8 quantized values
    /// Dequant: value = d * int8. One lane handles one element per block.
    /// Mirrors the CUDA llm_matvec_q8_0 kernel and the CPU DequantQ8_0 path.
    /// </summary>
    internal const string MatVecQ8_0 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        int  gInt8(uint b) { int v = int(gByte(b)); return v >= 128 ? v - 256 : v; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 5;            // cols / 32
            uint boff_base = row * num_blocks * 34;

            float acc = 0.0;
            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 34;
                float d = unpackHalf2x16(gByte(b0) | (gByte(b0 + 1) << 8)).x;
                int q = gInt8(b0 + 2 + lane);
                acc += d * float(q) * input_data[block * 32 + lane];
            }

            sdata[tid] = acc;
            barrier();
            [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                if (lane < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            if (lane == 0)
                output_data[row] = sdata[tid];
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q4_0 dequantization.
    /// Each workgroup computes 8 output rows (8 rows × 32 lanes = 256 threads).
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=quantized weights (uint8), 1=input vector (float), 2=output (float).
    ///
    /// Q4_0 block layout (18 bytes per 32 elements):
    ///   [0:2]  FP16 d (block scale)
    ///   [2:18] 16 bytes of packed 4-bit nibbles (two signed nibbles per byte)
    /// Element j (0..15) = low nibble of qs[j]; element j+16 = high nibble of qs[j].
    /// Dequant: value = (nibble - 8) * d. One lane handles one element per block.
    /// Mirrors the CUDA llm_matvec_q4_0 kernel and the CPU DequantQ4_0 path.
    /// </summary>
    internal const string MatVecQ4_0 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 5;            // cols / 32
            uint boff_base = row * num_blocks * 18;

            float acc = 0.0;
            // TWO blocks per iteration. A Q4_0 block has only 16 qs bytes for its 32 elements, so
            // the natural "lane L takes element L" mapping makes lanes L and L+16 read the SAME
            // byte — 32 lanes touch just 16 distinct bytes, half what Q8_0's 32-byte blocks manage,
            // and Q4_0 measured at almost exactly half Q8_0's bandwidth (17.9 vs 31.6 GB/s).
            // Instead each lane owns one byte outright and unpacks BOTH its nibbles, with the wave
            // split across two consecutive blocks: 32 distinct bytes per instruction, and half as
            // many iterations, which also halves the redundant per-lane reload of the block scale
            // d (all 32 lanes were computing the same d from two byte reads every iteration).
            // Byte k of a block holds element k in its low nibble and element k+16 in its high one.
            uint blkHalf = lane >> 4;               // which of the two blocks this lane serves
            uint bidx    = lane & 15;               // byte within that block's 16 qs bytes
            for (uint pair = 0; pair < num_blocks; pair += 2) {
                uint blk = pair + blkHalf;
                if (blk >= num_blocks) continue;    // odd block count: the tail half-wave idles
                uint b0 = boff_base + blk * 18;
                float d = unpackHalf2x16(gByte(b0) | (gByte(b0 + 1) << 8)).x;
                uint qbyte = gByte(b0 + 2 + bidx);
                uint e0 = blk * 32 + bidx;
                acc += d * float(int(qbyte & 0xF) - 8) * input_data[e0];
                acc += d * float(int(qbyte >> 4)  - 8) * input_data[e0 + 16];
            }

            sdata[tid] = acc;
            barrier();
            [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
                if (lane < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            if (lane == 0)
                output_data[row] = sdata[tid];
        }
        """;

    // ================================================================
    //  TurboQuant KV Cache Compression Shaders
    // ================================================================

    /// <summary>
    /// Rotate query vectors for TurboQuant: WHT + sign flip per KV head.
    /// One workgroup per query head. 128 threads per workgroup.
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim }.
    /// Bindings: 0=q_input[num_heads*head_dim], 1=rotated_q[num_heads*head_dim], 2=sign_patterns[num_kv_heads*head_dim].
    /// </summary>
    internal const string TqRotateQuery = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 128) in;

        layout(binding = 0) readonly buffer QIn      { float q_input[]; };
        layout(binding = 1) buffer QOut              { float rotated_q[]; };
        layout(binding = 2) readonly buffer Signs    { float sign_patterns[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
        };

        shared float sdata[128];

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads || tid >= head_dim) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint q_off = h * head_dim;
            uint sign_off = kv_head * head_dim;

            // Load query into shared memory
            sdata[tid] = q_input[q_off + tid];
            barrier();

            // In-place WHT butterfly
            [[unroll]] for (uint stride = 64; stride >= 1; stride >>= 1) {
                barrier();
                uint pair = (tid / stride) * (stride * 2) + (tid % stride);
                float a = sdata[pair];
                float b = sdata[pair + stride];
                sdata[pair] = a + b;
                sdata[pair + stride] = a - b;
            }
            barrier();

            // Normalize and apply sign flip
            float scale = 1.0 / sqrt(float(head_dim));
            rotated_q[q_off + tid] = sdata[tid] * scale * sign_patterns[sign_off + tid];
        }
        """;

    /// <summary>
    /// TurboQuant KV cache append: applies WHT + sign flip + quantization,
    /// then packs into 3-bit compressed format.
    /// Workgroup of 128 threads (one per dimension).
    /// Push constants: { uint kv_dim, uint head_dim, uint position, uint max_seq_len, uint num_kv_heads }.
    /// Bindings: 0=k_input[kv_dim], 1=v_input[kv_dim], 2=k_cache_tq[...], 3=v_cache_tq[...],
    ///           4=sign_patterns[num_kv_heads*head_dim], 5=codebook[8], 6=boundaries[7].
    /// Each workgroup handles one KV head.
    /// </summary>
    internal const string TqKvAppend = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 128) in;

        layout(binding = 0) readonly buffer KIn      { float k_input[]; };
        layout(binding = 1) readonly buffer VIn      { float v_input[]; };
        layout(binding = 2) buffer KCacheTQ          { uint k_cache_tq[]; };
        layout(binding = 3) buffer VCacheTQ          { uint v_cache_tq[]; };
        layout(binding = 4) readonly buffer Signs    { float sign_patterns[]; };
        layout(binding = 5) readonly buffer Codebook { float codebook[8]; };
        layout(binding = 6) readonly buffer Bounds   { float boundaries[7]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint head_dim;
            uint position;
            uint max_seq_len;
            uint num_kv_heads;
            uint block_bytes;    // bytes per compressed block (52 for 3-bit d=128)
        };

        shared float sdata[128];  // shared memory for WHT butterfly

        // Walsh-Hadamard transform (in-place butterfly, 128 elements)
        void wht_128() {
            uint tid = gl_LocalInvocationID.x;
            [[unroll]] for (uint stride = 64; stride >= 1; stride >>= 1) {
                barrier();
                uint pair = (tid / stride) * (stride * 2) + (tid % stride);
                float a = sdata[pair];
                float b = sdata[pair + stride];
                sdata[pair] = a + b;
                sdata[pair + stride] = a - b;
            }
            barrier();
            float scale = 1.0 / sqrt(float(head_dim));
            sdata[tid] *= scale;
            barrier();
        }

        // Find quantization bin for a normalized value
        int find_bin(float val) {
            int bin = 0;
            [[unroll]] for (int i = 0; i < 7; i++) {
                if (val >= boundaries[i]) bin = i + 1;
                else break;
            }
            return bin;
        }

        // Quantize shared memory data and write a compressed block to the key cache.
        void quantize_and_pack_k(uint cache_offset) {
            uint tid = gl_LocalInvocationID.x;

            // Compute L2 norm via parallel reduction
            float val = sdata[tid];
            sdata[tid] = val * val;
            barrier();
            [[unroll]] for (uint s = 64; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float norm = sqrt(sdata[0]);

            // Restore and normalize
            barrier();
            sdata[tid] = val;
            barrier();
            float inv_norm = (norm > 0.0) ? (1.0 / norm) : 0.0;
            float normalized = sdata[tid] * inv_norm;

            // Quantize to 3-bit index
            int idx = find_bin(normalized);

            // Thread 0 writes the FP16 norm
            // Pack 3-bit indices into uint array (each uint holds ~10 indices)
            barrier();

            // We store as: [FP16 norm as uint16 in first 2 bytes][48 bytes of packed 3-bit indices]
            // Using uint buffer: first uint has norm in lower 16 bits + first ~10 indices
            // Simpler approach: pack indices into shared memory, then write cooperatively

            // Each thread contributes its 3-bit index. We pack 10 indices per uint (30 bits).
            // 128 indices / 10 = 13 uints (last has 8 indices).
            // But for simplicity and correctness, pack bit-by-bit.

            // Store indices to shared memory
            sdata[tid] = float(idx);
            barrier();

            // Thread 0 writes the entire block
            if (tid == 0) {
                // Write FP16 norm as the first 2 bytes (stored in first uint, low 16 bits)
                uint norm_bits = packHalf2x16(vec2(norm, 0.0));

                // Pack 128 3-bit indices into 48 bytes = 12 uints
                uint packed[13]; // 13 uints = 52 bytes = our block
                packed[0] = norm_bits & 0xFFFFu; // first 2 bytes are norm

                // Pack bits starting at byte offset 2 (bit offset 16 within packed[0])
                uint bit_pos = 16; // start after norm
                for (uint i = 0; i < 128; i++) {
                    uint index3 = uint(sdata[i]) & 0x7u;
                    uint word_idx = bit_pos / 32;
                    uint bit_off = bit_pos % 32;
                    if (i == 0 && word_idx == 0) {
                        packed[word_idx] |= (index3 << bit_off);
                    } else {
                        if (bit_off == 0 && (i == 0 || (bit_pos % 32) == 0))
                            packed[word_idx] = 0;
                        packed[word_idx] |= (index3 << bit_off);
                    }
                    if (bit_off > 29) { // overflow into next uint
                        uint next_word = word_idx + 1;
                        if (bit_off > 29) packed[next_word] |= (index3 >> (32 - bit_off));
                    }
                    bit_pos += 3;
                }

                // Write packed block to cache buffer
                uint base_idx = cache_offset / 4; // uint offset
                uint num_uints = (block_bytes + 3) / 4;
                for (uint w = 0; w < num_uints; w++)
                    k_cache_tq[base_idx + w] = (w < 13) ? packed[w] : 0u;
            }
        }

        // Quantize shared memory data and write a compressed block to the value cache.
        void quantize_and_pack_v(uint cache_offset) {
            uint tid = gl_LocalInvocationID.x;

            float val = sdata[tid];
            sdata[tid] = val * val;
            barrier();
            [[unroll]] for (uint s = 64; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float norm = sqrt(sdata[0]);

            barrier();
            sdata[tid] = val;
            barrier();
            float inv_norm = (norm > 0.0) ? (1.0 / norm) : 0.0;
            float normalized = sdata[tid] * inv_norm;

            int idx = find_bin(normalized);

            barrier();
            sdata[tid] = float(idx);
            barrier();

            if (tid == 0) {
                uint norm_bits = packHalf2x16(vec2(norm, 0.0));
                uint packed[13];
                packed[0] = norm_bits & 0xFFFFu;

                uint bit_pos = 16;
                for (uint i = 0; i < 128; i++) {
                    uint index3 = uint(sdata[i]) & 0x7u;
                    uint word_idx = bit_pos / 32;
                    uint bit_off = bit_pos % 32;
                    if (i == 0 && word_idx == 0) {
                        packed[word_idx] |= (index3 << bit_off);
                    } else {
                        if (bit_off == 0 && (i == 0 || (bit_pos % 32) == 0))
                            packed[word_idx] = 0;
                        packed[word_idx] |= (index3 << bit_off);
                    }
                    if (bit_off > 29) {
                        uint next_word = word_idx + 1;
                        if (bit_off > 29) packed[next_word] |= (index3 >> (32 - bit_off));
                    }
                    bit_pos += 3;
                }

                uint base_idx = cache_offset / 4;
                uint num_uints = (block_bytes + 3) / 4;
                for (uint w = 0; w < num_uints; w++)
                    v_cache_tq[base_idx + w] = (w < 13) ? packed[w] : 0u;
            }
        }

        void main() {
            uint kv_head = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (kv_head >= num_kv_heads || tid >= head_dim) return;

            uint head_offset = kv_head * head_dim;
            uint byte_offset = position * num_kv_heads * block_bytes + kv_head * block_bytes;

            // --- Compress Key ---
            sdata[tid] = k_input[head_offset + tid];
            barrier();
            wht_128();
            // Apply sign flip
            sdata[tid] *= sign_patterns[head_offset + tid];
            barrier();
            quantize_and_pack_k(byte_offset);

            barrier();

            // --- Compress Value ---
            sdata[tid] = v_input[head_offset + tid];
            barrier();
            wht_128();
            sdata[tid] *= sign_patterns[head_offset + tid];
            barrier();
            quantize_and_pack_v(byte_offset);
        }
        """;

    /// <summary>
    /// TurboQuant attention: fused dequant-dot for compressed KV cache.
    /// One workgroup per query head. Tiles over sequence positions.
    /// Handles both compressed (TQ) positions and FP16 recent window.
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim,
    ///                    uint tq_seq_len, uint fp16_seq_len, uint max_seq_len,
    ///                    uint block_bytes }.
    /// Bindings: 0=Q[num_heads*head_dim], 1=rotated_Q[num_heads*head_dim],
    ///           2=k_cache_tq[...], 3=v_cache_tq[...],
    ///           4=k_cache_fp16[...], 5=v_cache_fp16[...],
    ///           6=output[num_heads*head_dim], 7=codebook[8],
    ///           8=scores_scratch[num_heads * max_seq_len]  (long-context spill).
    ///
    /// Score-storage strategy mirrors the CUDA kernel `llm_tq_attention`:
    ///   • total_seq ≤ MAX_SHARED_SCORES (4096): hot path uses shared memory.
    ///   • total_seq > 4096: spills to `scores_scratch[h*max_seq_len .. +total_seq)`.
    /// The fast path does not touch the scratch buffer, but Vulkan descriptor sets
    /// require it to be bound regardless — the caller passes a 1-float placeholder
    /// when max_seq_len ≤ 4096.
    /// </summary>
    internal const string TqAttention = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q             { float q_data[]; };
        layout(binding = 1) readonly buffer RotQ          { float rotated_q[]; };
        layout(binding = 2) readonly buffer KCacheTQ      { uint k_cache_tq[]; };
        layout(binding = 3) readonly buffer VCacheTQ      { uint v_cache_tq[]; };
        layout(binding = 4) readonly buffer KCacheFP16    { float k_cache_fp16[]; };
        layout(binding = 5) readonly buffer VCacheFP16    { float v_cache_fp16[]; };
        layout(binding = 6) buffer Out                    { float out_data[]; };
        layout(binding = 7) readonly buffer Codebook      { float codebook[8]; };
        layout(binding = 8) buffer ScoresScratch          { float scores_scratch[]; };
        layout(binding = 9) readonly buffer Signs         { float sign_patterns[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint tq_seq_len;      // number of TQ-compressed positions
            uint fp16_seq_len;    // number of FP16 recent positions
            uint max_seq_len;
            uint block_bytes;
        };

        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];    // reduction scratch

        float tq_dequant_dot_k(uint block_base_uint, uint kv_head) {
            float dot = 0.0;
            uint q_off = gl_WorkGroupID.x * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                uint bit_pos = 16u + d * 3u;
                uint word_idx = block_base_uint + bit_pos / 32u;
                uint bit_off = bit_pos & 31u;
                uint raw = k_cache_tq[word_idx] >> bit_off;
                if (bit_off > 29u) raw |= k_cache_tq[word_idx + 1u] << (32u - bit_off);
                int idx = int(raw & 0x7u);
                dot += codebook[idx] * rotated_q[q_off + d];
            }
            return dot;
        }

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint q_off = h * head_dim;
            uint out_off = h * head_dim;
            uint total_seq = tq_seq_len + fp16_seq_len;

            bool use_shared = (total_seq <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1a: per-position scores for TQ-compressed positions ───
            for (uint t = tid; t < tq_seq_len; t += 256) {
                uint block_byte_off = t * num_kv_heads * block_bytes + kv_head * block_bytes;
                uint block_base_uint = block_byte_off / 4u;

                // FP16 per-block norm packed in first 2 bytes of the block.
                uint norm_word = k_cache_tq[block_base_uint];
                float norm = unpackHalf2x16(norm_word).x;

                float dot = tq_dequant_dot_k(block_base_uint, kv_head);
                float score = dot * norm * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }

            // ─── Phase 1b: FP16 recent-window positions ───
            for (uint t = tid; t < fp16_seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache_fp16[k_off + d];
                float score = dot * scale;
                if (use_shared) scores[tq_seq_len + t] = score;
                else            scores_scratch[scratch_base + tq_seq_len + t] = score;
            }

            // Pad the shared tail with -inf so the max scan ignores stale slots.
            // The scratch path's scans iterate only [0, total_seq), so no padding needed.
            if (use_shared) {
                for (uint t = total_seq + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }

            barrier();

            // ─── Phase 2: in-place softmax over [0, total_seq) ───
            // Max.
            float local_max = -1.0/0.0;
            for (uint t = tid; t < total_seq; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                local_max = max(local_max, s);
            }
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            // Exp + sum.
            float local_sum = 0.0;
            for (uint t = tid; t < total_seq; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                float e = exp(s - max_val);
                if (use_shared) scores[t] = e;
                else            scores_scratch[scratch_base + t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            // Normalize → softmax weight per position.
            for (uint t = tid; t < total_seq; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum into output[head, :] ───
            // The TQ V codes are stored ROTATED (TqKvAppend applies WHT·1/sqrt(D)·sign
            // to V), so the compressed-region aggregate is built in the rotated domain,
            // un-rotated ONCE per head (deferred sign flip + inverse WHT — issue #435),
            // then the FP16 recent-window contribution adds on top in the original domain.

            // 3a — compressed-region aggregate in the rotated domain (sdata reused;
            // head_dim ≤ 256 → one slot per output dim).
            for (uint d = tid; d < head_dim; d += 256) {
                float acc = 0.0;
                for (uint t = 0; t < tq_seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];

                    uint block_byte_off = t * num_kv_heads * block_bytes + kv_head * block_bytes;
                    uint block_base_uint = block_byte_off / 4u;
                    uint norm_word = v_cache_tq[block_base_uint];
                    float norm = unpackHalf2x16(norm_word).x;

                    uint bit_pos = 16u + d * 3u;
                    uint word_idx = block_base_uint + bit_pos / 32u;
                    uint bit_off = bit_pos & 31u;
                    uint raw = v_cache_tq[word_idx] >> bit_off;
                    if (bit_off > 29u) raw |= v_cache_tq[word_idx + 1u] << (32u - bit_off);
                    int idx = int(raw & 0x7u);

                    acc += weight * codebook[idx] * norm;
                }
                sdata[d] = acc;
            }
            barrier();

            // 3b — deferred sign flip + inverse WHT (once per head). v = Hn·D·rot_acc:
            // the per-head sign pattern first, then the normalized WHT (an involution;
            // 1/sqrt(D) folded in at readout below).
            uint sign_off = kv_head * head_dim;
            if (tid < head_dim) sdata[tid] *= sign_patterns[sign_off + tid];
            barrier();
            for (uint stride = head_dim >> 1u; stride >= 1u; stride >>= 1u) {
                if (tid < (head_dim >> 1u)) {
                    uint pair = (tid / stride) * (stride * 2u) + (tid % stride);
                    float a = sdata[pair];
                    float b = sdata[pair + stride];
                    sdata[pair] = a + b;
                    sdata[pair + stride] = a - b;
                }
                barrier();
            }
            // Exact 1/sqrt(D) (not inversesqrt) to mirror the CPU WalshHadamard.Transform
            // normalization this un-rotate inverts — same convention as TqRotateQuery.
            float wht_scale = 1.0 / sqrt(float(head_dim));

            // 3c — FP16 recent-window V contribution (original domain) + output write.
            for (uint d = tid; d < head_dim; d += 256) {
                float sum_val = (tq_seq_len > 0u) ? sdata[d] * wht_scale : 0.0;
                for (uint t = 0; t < fp16_seq_len; t++) {
                    float weight = use_shared
                        ? scores[tq_seq_len + t]
                        : scores_scratch[scratch_base + tq_seq_len + t];
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum_val += weight * v_cache_fp16[v_off + d];
                }

                out_data[out_off + d] = sum_val;
            }
        }
        """;

    // ================================================================
    //  DiT / Diffusion Shaders
    // ================================================================

    /// <summary>
    /// Tiled SGEMM: C[M,N] = A[M,K] × B[N,K]^T
    /// A is activations [M rows, K cols], B is weights [N rows, K cols] (row = one output neuron's weights).
    /// Uses 16×16 shared-memory tiles with +1 column padding to avoid bank conflicts.
    ///
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (readonly), 1=B (readonly), 2=C (writeonly).
    /// Dispatch: (ceil(M/16), ceil(N/16), 1) with local_size=(16,16,1).
    /// </summary>
    internal const string SgemmF32 = """
        #version 450

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        layout(push_constant) uniform PC {
            uint M;
            uint N;
            uint K;
        } pc;

        layout(binding = 0) readonly  buffer BufA { float a_data[]; };   // [M, K] activations
        layout(binding = 1) readonly  buffer BufB { float b_data[]; };   // [N, K] weights
        layout(binding = 2) writeonly buffer BufC { float c_data[]; };   // [M, N] output

        shared float tileA[16][17]; // +1 column to avoid bank conflicts
        shared float tileB[16][17];

        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;

            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;

            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;

                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? a_data[row * pc.K + aCol] : 0.0;

                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? b_data[col * pc.K + bCol] : 0.0;

                barrier();

                for (uint k = 0u; k < 16u; k++)
                    acc += tileA[gl_LocalInvocationID.x][k] * tileB[gl_LocalInvocationID.y][k];

                barrier();
            }

            if (row < pc.M && col < pc.N)
                c_data[row * pc.N + col] = acc;
        }
        """;

    /// <summary>
    /// Mixed-precision SGEMM: C[M,N] = A[M,K] × B[N,K]^T
    /// A (activations) is fp32 — avoids activation overflow (e.g. SiLU*gate &gt; 65504).
    /// B (weights) is fp16 — bandwidth savings on large weight matrices.
    /// Accumulation and output C are fp32 — full range, no overflow.
    ///
    /// Requires: VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    ///
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (readonly fp32 activations), 1=B (readonly fp16 weights), 2=C (writeonly fp32).
    /// Dispatch: (ceil(M/16), ceil(N/16), 1) with local_size=(16,16,1).
    /// </summary>
    internal const string SgemmF16 = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        layout(push_constant) uniform PC {
            uint M;
            uint N;
            uint K;
        } pc;

        layout(binding = 0) readonly  buffer BufA { float    a_data[]; };
        layout(binding = 1) readonly  buffer BufB { float16_t b_data[]; };
        layout(binding = 2) writeonly buffer BufC { float    c_data[]; };

        // fp32 shared tiles. A reads fp32, B reads fp16 (converted on load).
        shared float tileA[16][17];
        shared float tileB[16][17];

        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;

            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;

            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;

                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? a_data[row * pc.K + aCol] : 0.0;

                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? float(b_data[col * pc.K + bCol]) : 0.0;

                barrier();

                for (uint k = 0u; k < 16u; k++)
                    acc += tileA[gl_LocalInvocationID.x][k] * tileB[gl_LocalInvocationID.y][k];

                barrier();
            }

            if (row < pc.M && col < pc.N)
                c_data[row * pc.N + col] = acc;
        }
        """;

    /// <summary>
    /// Tiled int8-weight × fp16-activation SGEMM: C[M,N] = A[M,K] × (scale * B)[N,K]^T
    /// A is fp16 activations, B is int8 weights (per-row quantized with fp16 scales).
    /// Accumulation is done in fp32.
    ///
    /// Requires: VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage + VK_KHR_8bit_storage
    ///
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (fp16 activations [M,K]), 1=B (int8 weights [N,K]),
    ///           2=scale (fp16 per-row scales [N]), 3=C (fp16 output [M,N]).
    /// Dispatch: (ceil(M/16), ceil(N/16), 1) with local_size=(16,16,1).
    /// </summary>
    internal const string SgemmInt8Fp16 = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_explicit_arithmetic_types_int8    : require
        #extension GL_EXT_shader_16bit_storage : require
        #extension GL_EXT_shader_8bit_storage  : require

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        layout(push_constant) uniform PC {
            uint M;
            uint N;
            uint K;
        } pc;

        layout(binding = 0) readonly  buffer BufA  { float16_t a_data[]; };
        layout(binding = 1) readonly  buffer BufB  { int8_t    b_data[]; };
        layout(binding = 2) readonly  buffer BufS  { float16_t b_scale[]; };
        layout(binding = 3) writeonly buffer BufC  { float16_t c_data[]; };

        shared float16_t tileA[16][17];
        shared int8_t    tileB[16][17];

        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;

            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;

            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;

                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? a_data[row * pc.K + aCol] : float16_t(0.0);

                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? b_data[col * pc.K + bCol] : int8_t(0);

                barrier();

                for (uint k = 0u; k < 16u; k++)
                    acc += float(tileA[gl_LocalInvocationID.x][k]) *
                           float(tileB[gl_LocalInvocationID.y][k]);

                barrier();
            }

            if (row < pc.M && col < pc.N) {
                float scale = float(b_scale[col]);
                c_data[row * pc.N + col] = float16_t(acc * scale);
            }
        }
        """;

    /// <summary>
    /// Tiled bf16 SGEMM: C[M,N] = A[M,K] × B[N,K]^T
    /// All inputs and output are bfloat16_t. Accumulation in fp32.
    /// Requires: VK_KHR_shader_bfloat16 + VK_KHR_16bit_storage
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (readonly bf16), 1=B (readonly bf16), 2=C (writeonly bf16).
    /// </summary>
    internal const string SgemmBf16 = """
        #version 450
        #extension GL_KHR_shader_bfloat16 : require
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
        layout(push_constant) uniform PC { uint M; uint N; uint K; } pc;
        layout(binding = 0) readonly  buffer BufA { bfloat16_t a_data[]; };
        layout(binding = 1) readonly  buffer BufB { bfloat16_t b_data[]; };
        layout(binding = 2) writeonly buffer BufC { bfloat16_t c_data[]; };
        // fp32 shared tiles to avoid driver issues with bf16 shared memory;
        // global loads/stores remain bf16 so VRAM bandwidth is fully saved.
        shared float tileA[16][17];
        shared float tileB[16][17];
        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;
            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;
                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? float(a_data[row * pc.K + aCol]) : 0.0;
                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? float(b_data[col * pc.K + bCol]) : 0.0;
                barrier();
                for (uint k = 0u; k < 16u; k++)
                    acc += tileA[gl_LocalInvocationID.x][k] * tileB[gl_LocalInvocationID.y][k];
                barrier();
            }
            if (row < pc.M && col < pc.N)
                c_data[row * pc.N + col] = bfloat16_t(acc);
        }
        """;

    /// <summary>
    /// Tiled fp8 × fp16 SGEMM: C[M,N] = A[M,K] × B[N,K]^T
    /// A is fp16 activations, B is fp8 E4M3 weights, C is fp16 output.
    /// Requires: VK_EXT_shader_float8 + VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (fp16), 1=B (fp8 e4m3), 2=C (fp16).
    /// </summary>
    internal const string SgemmFp8 = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float8_e4m3 : require

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
        layout(push_constant) uniform PC { uint M; uint N; uint K; } pc;
        layout(binding = 0) readonly  buffer BufA { float8_e4m3_t a_data[]; };
        layout(binding = 1) readonly  buffer BufB { float8_e4m3_t b_data[]; };
        layout(binding = 2) writeonly buffer BufC { float c_data[]; };
        // fp32 shared tiles: avoids driver issues with fp16/fp8 shared memory
        shared float tileA[16][17];
        shared float tileB[16][17];
        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;
            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;
                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? float(a_data[row * pc.K + aCol]) : 0.0;
                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? float(b_data[col * pc.K + bCol]) : 0.0;
                barrier();
                for (uint k = 0u; k < 16u; k++)
                    acc += tileA[gl_LocalInvocationID.x][k] * tileB[gl_LocalInvocationID.y][k];
                barrier();
            }
            if (row < pc.M && col < pc.N)
                c_data[row * pc.N + col] = acc;
        }
        """;

    /// <summary>
    /// GPU-side Q5_K_M dequantization: one workgroup per block, 256 threads per workgroup.
    /// Q5_K block layout (176 bytes / 256 elements):
    ///   [0:2]   FP16 d (super-block scale)
    ///   [2:4]   FP16 dmin
    ///   [4:16]  12 bytes packed 6-bit scales/mins
    ///   [16:48] 32 bytes qh (1 high bit per element)
    ///   [48:176] 128 bytes ql (4-bit nibbles, 2 per byte)
    /// Requires: VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    /// Push constants: { uint numBlocks }.
    /// Bindings: 0=src (raw uint32 array), 1=dst (fp16 array).
    /// Dispatch: (numBlocks, 1, 1).
    /// </summary>
    internal const string DequantQ5KM = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        layout(push_constant) uniform PC { uint numBlocks; } pc;
        layout(binding = 0) readonly  buffer SrcBuf { uint src[]; };
        layout(binding = 1) writeonly buffer DstBuf { float16_t dst[]; };

        uint byteAt(uint bi) { return (src[bi >> 2u] >> ((bi & 3u) << 3u)) & 0xFFu; }

        void getScaleMinK4(uint j, uint scBase, out uint sc, out uint mn) {
            if (j < 4u) { sc = byteAt(scBase + j) & 63u; mn = byteAt(scBase + j + 4u) & 63u; }
            else {
                sc = (byteAt(scBase + j + 4u) & 0xFu) | ((byteAt(scBase + j - 4u) >> 6u) << 4u);
                mn = (byteAt(scBase + j + 4u) >> 4u)  | ((byteAt(scBase + j)       >> 6u) << 4u);
            }
        }

        void main() {
            uint blockIdx = gl_WorkGroupID.x;
            if (blockIdx >= pc.numBlocks) return;

            uint elem  = gl_LocalInvocationID.x;
            uint bBase = blockIdx * 176u;

            uint dBits    = byteAt(bBase + 0u) | (byteAt(bBase + 1u) << 8u);
            uint dminBits = byteAt(bBase + 2u) | (byteAt(bBase + 3u) << 8u);
            float d    = unpackHalf2x16(dBits).x;
            float dmin = unpackHalf2x16(dminBits).x;

            uint scBase = bBase + 4u;
            uint qhBase = bBase + 16u;
            uint qlBase = bBase + 48u;

            uint grp   = elem / 64u;
            uint loc   = elem % 64u;
            uint lo_hi = loc  / 32u;
            uint l     = loc  % 32u;

            uint scaleIdx = grp * 2u + lo_hi;
            uint sc, mn;
            getScaleMinK4(scaleIdx, scBase, sc, mn);
            float df  = d    * float(sc);
            float dmf = dmin * float(mn);

            uint u      = 1u << (grp * 2u + lo_hi);
            uint hBit   = ((byteAt(qhBase + l) & u) != 0u) ? 16u : 0u;
            uint qlByte = byteAt(qlBase + grp * 32u + l);
            uint q5     = (lo_hi == 0u ? (qlByte & 0xFu) : (qlByte >> 4u)) + hBit;

            dst[blockIdx * 256u + elem] = float16_t(df * float(q5) - dmf);
        }
        """;

    /// <summary>
    /// GPU-side Q4_K_M dequantization: one workgroup per block, 256 threads per workgroup.
    /// Q4_K block layout (144 bytes / 256 elements):
    ///   [0:2]   FP16 d
    ///   [2:4]   FP16 dmin
    ///   [4:16]  12 bytes packed 6-bit scales/mins
    ///   [16:144] 128 bytes ql (4-bit nibbles, 2 per byte)
    /// Requires: VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    /// Push constants: { uint numBlocks }.
    /// Bindings: 0=src (raw uint32 array), 1=dst (fp16 array).
    /// Dispatch: (numBlocks, 1, 1).
    /// </summary>
    internal const string DequantQ4KM = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        layout(push_constant) uniform PC { uint numBlocks; } pc;
        layout(binding = 0) readonly  buffer SrcBuf { uint src[]; };
        layout(binding = 1) writeonly buffer DstBuf { float16_t dst[]; };

        uint byteAt(uint bi) { return (src[bi >> 2u] >> ((bi & 3u) << 3u)) & 0xFFu; }

        void getScaleMinK4(uint j, uint scBase, out uint sc, out uint mn) {
            if (j < 4u) { sc = byteAt(scBase + j) & 63u; mn = byteAt(scBase + j + 4u) & 63u; }
            else {
                sc = (byteAt(scBase + j + 4u) & 0xFu) | ((byteAt(scBase + j - 4u) >> 6u) << 4u);
                mn = (byteAt(scBase + j + 4u) >> 4u)  | ((byteAt(scBase + j)       >> 6u) << 4u);
            }
        }

        void main() {
            uint blockIdx = gl_WorkGroupID.x;
            if (blockIdx >= pc.numBlocks) return;

            uint elem  = gl_LocalInvocationID.x;
            uint bBase = blockIdx * 144u;

            uint dBits    = byteAt(bBase + 0u) | (byteAt(bBase + 1u) << 8u);
            uint dminBits = byteAt(bBase + 2u) | (byteAt(bBase + 3u) << 8u);
            float d    = unpackHalf2x16(dBits).x;
            float dmin = unpackHalf2x16(dminBits).x;

            uint scBase = bBase + 4u;
            uint qlBase = bBase + 16u;

            uint grp   = elem / 64u;
            uint loc   = elem % 64u;
            uint lo_hi = loc  / 32u;
            uint l     = loc  % 32u;

            uint scaleIdx = grp * 2u + lo_hi;
            uint sc, mn;
            getScaleMinK4(scaleIdx, scBase, sc, mn);
            float df  = d    * float(sc);
            float dmf = dmin * float(mn);

            uint qlByte = byteAt(qlBase + grp * 32u + l);
            uint q4     = (lo_hi == 0u) ? (qlByte & 0xFu) : (qlByte >> 4u);

            dst[blockIdx * 256u + elem] = float16_t(df * float(q4) - dmf);
        }
        """;

    // ── Image upscaler ops (RRDBNet) ──────────────────────────────────────

    /// <summary>
    /// 2D convolution: output[outCh, H, W] = conv(input[inCh, H, W], weight[outCh, inCh, k, k]) + bias[outCh]
    /// stride=1, configurable padding (default same).
    /// Each thread computes one output element (oc, oh, ow).
    /// Push constants: { inCh, outCh, height, width, ksize, padding }.
    /// Bindings: 0=input, 1=weight, 2=bias, 3=output.
    /// Dispatch: ceil(outCh * H * W / 256).
    /// </summary>
    /// <summary>
    /// Conv2d shader using a 2D workgroup dispatch: X=outCh, Y=ceil(H*W/256).
    /// All 256 threads in a workgroup share the same output channel, so they
    /// cooperatively load that channel's weight vector into shared memory once —
    /// reducing weight reads from global memory by 256×.
    ///
    /// Dispatch: (outCh, ceil(H*W / 256), 1) — matches VulkanBackend.Conv2d.
    /// Push constants unchanged: { inCh, outCh, height, width, ksize, padding }.
    /// </summary>
    internal const string Conv2d = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) readonly  buffer Weight { float weight_data[]; };
        layout(binding = 2) readonly  buffer Bias   { float bias_data[];   };
        layout(binding = 3) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint inCh;
            uint outCh;
            uint height;
            uint width;
            uint ksize;
            uint padding;
        };

        // Shared memory for one output-channel's weight vector.
        // Max weight per channel: 192 inCh × 3×3 kernel = 1728 floats = 6.75 KB.
        // 2048 slots provides safe alignment margin.
        shared float sWeights[2048];

        void main() {
            uint oc      = gl_WorkGroupID.x;           // output channel index
            uint tileIdx = gl_WorkGroupID.y;           // spatial tile within channel
            uint lid     = gl_LocalInvocationID.x;     // thread within tile (0..255)

            uint hw  = height * width;
            uint pos = tileIdx * 256u + lid;           // output pixel index

            // Cooperatively load all weights for this output channel into shared memory.
            // wLen ≤ 2048 for all configs in RRDBNet; each thread loads ceil(wLen/256) slots.
            uint wLen  = inCh * ksize * ksize;
            uint wBase = oc * wLen;
            for (uint i = lid; i < wLen; i += 256u)
                sWeights[i] = weight_data[wBase + i];

            // Ensure all threads see the fully loaded weights before computing.
            barrier();
            memoryBarrierShared();

            if (oc >= outCh || pos >= hw) return;

            uint oh = pos / width;
            uint ow = pos % width;

            float acc = bias_data[oc];
            for (uint ic = 0u; ic < inCh; ic++) {
                uint iBase   = ic * hw;
                uint wIcBase = ic * ksize * ksize;
                for (uint kh = 0u; kh < ksize; kh++) {
                    for (uint kw = 0u; kw < ksize; kw++) {
                        int ih = int(oh + kh) - int(padding);
                        int iw = int(ow + kw) - int(padding);
                        if (uint(ih) < height && uint(iw) < width)
                            acc += input_data[iBase + uint(ih) * width + uint(iw)]
                                 * sWeights[wIcBase + kh * ksize + kw];
                    }
                }
            }
            output_data[oc * hw + pos] = acc;
        }
        """;

    /// <summary>
    /// LeakyReLU in-place: data[i] = data[i] >= 0 ? data[i] : negSlope * data[i]
    /// Push constants: { n, negSlope }.
    /// Bindings: 0=data (in/out).
    /// </summary>
    internal const string LeakyRelu = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint  n;
            float negSlope;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            float x = data[i];
            data[i] = x >= 0.0 ? x : negSlope * x;
        }
        """;

    /// <summary>
    /// Clamp in-place: data[i] = clamp(data[i], minVal, maxVal)
    /// Push constants: { n, minVal, maxVal }.
    /// Bindings: 0=data (in/out).
    /// </summary>
    internal const string ClampInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint  n;
            float minVal;
            float maxVal;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            data[i] = clamp(data[i], minVal, maxVal);
        }
        """;

    /// <summary>
    /// Channel concatenation: output[(aCh+bCh), hw] from a[aCh, hw] and b[bCh, hw].
    /// Push constants: { aCh, bCh, hw }.
    /// Bindings: 0=a, 1=b, 2=output.
    /// Dispatch: ceil((aCh+bCh)*hw / 256).
    /// </summary>
    internal const string CatChannels = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer A      { float a_data[];      };
        layout(binding = 1) readonly  buffer B      { float b_data[];      };
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint aCh;
            uint bCh;
            uint hw;
        };

        void main() {
            uint idx   = gl_GlobalInvocationID.x;
            uint outCh = aCh + bCh;
            if (idx >= outCh * hw) return;

            uint c   = idx / hw;
            uint pos = idx % hw;
            output_data[idx] = (c < aCh)
                ? a_data[c * hw + pos]
                : b_data[(c - aCh) * hw + pos];
        }
        """;

    /// <summary>
    /// Pixel shuffle: [inCh, H, W] → [inCh/r², H*r, W*r]  (r = upscale)
    /// Push constants: { inCh, h, w, upscale }.
    /// Bindings: 0=input, 1=output.
    /// Dispatch: ceil(outCh*outH*outW / 256).
    /// </summary>
    internal const string PixelShuffle = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint inCh;
            uint h;
            uint w;
            uint upscale;
        };

        void main() {
            uint r2    = upscale * upscale;
            uint outCh = inCh / r2;
            uint outH  = h * upscale;
            uint outW  = w * upscale;

            uint idx = gl_GlobalInvocationID.x;
            if (idx >= outCh * outH * outW) return;

            uint outHW = outH * outW;
            uint oc    = idx / outHW;
            uint pos   = idx % outHW;
            uint oh    = pos / outW;
            uint ow    = pos % outW;

            uint ih = oh / upscale;
            uint iw = ow / upscale;
            uint rh = oh % upscale;
            uint rw = ow % upscale;

            // Input channel: oc * r² + rh * upscale + rw
            uint ic = oc * r2 + rh * upscale + rw;
            output_data[idx] = input_data[ic * h * w + ih * w + iw];
        }
        """;

    /// <summary>
    /// Pixel unshuffle (inverse): [inCh, H*r, W*r] → [inCh*r², H, W]  (r = downscale)
    /// Push constants: { inCh, h (output), w (output), downscale }.
    /// Bindings: 0=input, 1=output.
    /// Dispatch: ceil(inCh*r²*h*w / 256).
    /// </summary>
    internal const string PixelUnshuffle = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint inCh;
            uint h;         // output height = inputH / downscale
            uint w;         // output width  = inputW / downscale
            uint downscale;
        };

        void main() {
            uint d2    = downscale * downscale;
            uint outCh = inCh * d2;
            uint inH   = h * downscale;
            uint inW   = w * downscale;

            uint idx = gl_GlobalInvocationID.x;
            if (idx >= outCh * h * w) return;

            uint hw  = h * w;
            uint oc  = idx / hw;
            uint pos = idx % hw;
            uint oh  = pos / w;
            uint ow  = pos % w;

            uint ic  = oc / d2;
            uint rem = oc % d2;
            uint rh  = rem / downscale;
            uint rw  = rem % downscale;

            uint ih = oh * downscale + rh;
            uint iw = ow * downscale + rw;
            output_data[idx] = input_data[ic * inH * inW + ih * inW + iw];
        }
        """;

    /// <summary>
    /// Nearest-neighbour 2× upsample: [ch, H, W] → [ch, 2H, 2W]
    /// Push constants: { ch, h, w }.
    /// Bindings: 0=input, 1=output.
    /// Dispatch: ceil(ch*2H*2W / 256).
    /// </summary>
    internal const string Upsample2xNearest = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint ch;
            uint h;
            uint w;
        };

        void main() {
            uint idx   = gl_GlobalInvocationID.x;
            uint outHW = 4u * h * w;   // (2h)*(2w)
            if (idx >= ch * outHW) return;

            uint c   = idx / outHW;
            uint pos = idx % outHW;
            uint oh  = pos / (2u * w);
            uint ow  = pos % (2u * w);

            output_data[idx] = input_data[c * h * w + (oh / 2u) * w + (ow / 2u)];
        }
        """;

    /// <summary>
    /// Flash-decoding split-KV partial attention (issue #312) — the Vulkan mirror of the CUDA
    /// <c>llm_attention_splitkv</c> kernel. The single-workgroup <see cref="Attention"/> shader
    /// launches only <c>num_heads</c> workgroups and serially scans the whole KV range, which
    /// collapses decode throughput at very long context (the two earlier single-workgroup
    /// online-softmax attempts regressed for exactly this reason). This kernel splits each head's
    /// causal sequence <c>[0, seq_len)</c> into fixed <c>CHUNK</c>-sized slices and dispatches a
    /// 2D grid of <c>num_heads × n_splits</c> workgroups, so the KV read parallelizes across the
    /// GPU. Each workgroup emits the UN-normalized online-softmax partial for its slice; the
    /// companion <see cref="AttentionSplitKvCombine"/> LSE-merges the per-head partials.
    ///
    /// fp32 K/V (the bf16/q8_0 caches use <see cref="AttentionSplitKvPartialBf16"/> /
    /// <see cref="AttentionSplitKvPartialQ8"/>, which differ only in the K/V read — issue #332).
    /// Scalar (no subgroup ops) — uses plain shared-memory tree reductions, so #318's
    /// subgroup-size pin is irrelevant here.
    ///
    /// Workgroup (h = gl_WorkGroupID.x, s = gl_WorkGroupID.y) handles slice
    /// <c>[s*CHUNK, min((s+1)*CHUNK, seq_len))</c>. Out-of-range splits (s*CHUNK ≥ seq_len, from
    /// the caller's n_splits = ceil(seq_len/CHUNK)) write (m=−inf, l=0) and return so the combine
    /// scale exp(m−gmax)=0 skips them. GQA: kv_head = h / (num_heads/num_kv_heads).
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim, uint seq_len, uint n_splits, uint window }.
    /// Bindings: 0=Q[num_heads*head_dim], 1=K_cache[seq_len*kv_dim], 2=V_cache[seq_len*kv_dim],
    ///           3=partial_o[num_heads*n_splits*head_dim], 4=partial_meta[num_heads*n_splits*2].
    /// </summary>
    internal const string AttentionSplitKvPartial = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q          { float q_data[]; };
        layout(binding = 1) readonly buffer KCache     { float k_cache[]; };
        layout(binding = 2) readonly buffer VCache     { float v_cache[]; };
        layout(binding = 3) buffer PartialO            { float partial_o[]; };
        layout(binding = 4) buffer PartialMeta         { float partial_meta[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint n_splits;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        const uint CHUNK = 256u;
        shared float sk_scores[256];   // per-slice scores (≤ CHUNK)
        shared float sdata[256];       // reduction scratch

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint h = gl_WorkGroupID.x;   // query head
            uint s = gl_WorkGroupID.y;   // KV split
            if (h >= num_heads || s >= n_splits) return;

            uint meta_off = (h * n_splits + s) * 2u;
            // SWA bound — mirrors the fp32 Attention shader (CPU ForwardPass.Attention).
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;
            uint t0 = s * CHUNK;
            uint t1 = t0 + CHUNK; if (t1 > seq_len) t1 = seq_len;
            // Empty for this split: out-of-range (t0 >= seq_len, fixed n_splits) OR entirely below
            // the sliding window (t1 <= start_seq). Mark empty and bail so the combine skips it
            // (scale = exp(−inf − gmax) = 0) and never reads a stale numerator.
            if (t0 >= seq_len || t1 <= start_seq) {
                if (tid == 0u) { partial_meta[meta_off] = -1.0/0.0; partial_meta[meta_off + 1u] = 0.0; }
                return;
            }
            // Clamp the slice's start to the window so positions < start_seq never contribute.
            if (t0 < start_seq) t0 = start_seq;
            uint n = t1 - t0;   // 1 ≤ n ≤ CHUNK

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim  = num_kv_heads * head_dim;
            float scale  = inversesqrt(float(head_dim));
            uint q_off   = h * head_dim;
            uint kv_base = t0 * kv_dim + kv_head * head_dim;   // first row of this (clamped) slice for this kv head

            // ─── Phase 1: scores for the slice → shared (indexed t − t0) ───
            for (uint t = tid; t < n; t += 256u) {
                float dot = 0.0;
                uint k_off = kv_base + t * kv_dim;
                for (uint d = 0u; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                sk_scores[t] = dot * scale;
            }
            barrier();

            // ─── Phase 2: local max over the slice ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < n; t += 256u) local_max = max(local_max, sk_scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] = max(sdata[tid], sdata[tid + r]);
                barrier();
            }
            float m_i = sdata[0];
            barrier();

            // exp(score − m_i) in place + local denom.
            float local_sum = 0.0;
            for (uint t = tid; t < n; t += 256u) {
                float e = exp(sk_scores[t] - m_i);
                sk_scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] += sdata[tid + r];
                barrier();
            }
            float l_i = sdata[0];
            barrier();

            if (tid == 0u) { partial_meta[meta_off] = m_i; partial_meta[meta_off + 1u] = l_i; }

            // ─── Phase 3: UN-normalized weighted-V numerator for this slice ───
            uint o_off = (h * n_splits + s) * head_dim;
            for (uint d = tid; d < head_dim; d += 256u) {
                float acc = 0.0;
                for (uint t = 0u; t < n; t++) {
                    uint v_off = kv_base + t * kv_dim;   // same hoisted base as Phase 1
                    acc += sk_scores[t] * v_cache[v_off + d];
                }
                partial_o[o_off + d] = acc;
            }
        }
        """;

    /// <summary>
    /// Flash-decoding combine (issue #312) — the Vulkan mirror of the CUDA
    /// <c>llm_attention_combine</c> kernel. One workgroup per query head; LSE-merges the
    /// <c>n_splits</c> per-slice partials emitted by <see cref="AttentionSplitKvPartial"/> into
    /// the final attention output with the standard online-softmax rescale:
    ///   <c>m = max_s m_s ; l = Σ_s exp(m_s−m)·l_s ; out[d] = (Σ_s exp(m_s−m)·Õ_s[d]) / l</c>.
    /// Exact modulo FP reduction order. Empty splits carry m_s=−inf → scale 0 → skipped.
    /// MAX_SPLITS bounds the per-head split count (ceil(131072/512)=256).
    ///
    /// Push constants: { uint num_heads, uint head_dim, uint n_splits }.
    /// Bindings: 0=partial_o[num_heads*n_splits*head_dim], 1=partial_meta[num_heads*n_splits*2],
    ///           2=output[num_heads*head_dim].
    /// </summary>
    internal const string AttentionSplitKvCombine = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer PartialO    { float partial_o[]; };
        layout(binding = 1) readonly buffer PartialMeta { float partial_meta[]; };
        layout(binding = 2) buffer Out                  { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            uint n_splits;
        };

        const uint MAX_SPLITS = 256u;
        shared float sh_scale[256];   // per-split rescale exp(m_s − gmax)
        shared float red[256];        // reduction scratch
        shared float sh_gmax;
        shared float sh_denom;

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint h = gl_WorkGroupID.x;
            if (h >= num_heads) return;
            uint base = h * n_splits;

            // Global max over the splits' local maxima.
            float lmax = -1.0/0.0;
            for (uint s = tid; s < n_splits; s += 256u)
                lmax = max(lmax, partial_meta[(base + s) * 2u]);
            red[tid] = lmax;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) red[tid] = max(red[tid], red[tid + r]);
                barrier();
            }
            if (tid == 0u) sh_gmax = red[0];
            barrier();
            float gmax = sh_gmax;

            // Per-split rescale factor exp(m_s − gmax) + global denom Σ exp(m_s−gmax)·l_s.
            float ldenom = 0.0;
            for (uint s = tid; s < n_splits; s += 256u) {
                float m = partial_meta[(base + s) * 2u];
                float l = partial_meta[(base + s) * 2u + 1u];
                float sc = exp(m - gmax);
                sh_scale[s] = sc;
                ldenom += sc * l;
            }
            red[tid] = ldenom;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) red[tid] += red[tid + r];
                barrier();
            }
            if (tid == 0u) sh_denom = red[0];
            barrier();
            float inv = 1.0 / sh_denom;

            // Weighted sum of the per-split numerators across head_dim.
            uint po_base  = base * head_dim;     // first split's row for this head
            uint out_base = h * head_dim;
            for (uint d = tid; d < head_dim; d += 256u) {
                float acc = 0.0;
                for (uint s = 0u; s < n_splits; s++) {
                    float sc = sh_scale[s];
                    if (sc != 0.0) acc += sc * partial_o[po_base + s * head_dim + d];
                }
                out_data[out_base + d] = acc * inv;
            }
        }
        """;

    /// <summary>
    /// bf16 (issue #332) variant of <see cref="AttentionSplitKvPartial"/>: control flow is
    /// IDENTICAL to the fp32 partial; the ONLY difference is that the K/V cache buffers
    /// (bindings 1, 2) hold IEEE fp16 packed two-per-uint and are read via
    /// <c>unpackHalf2x16</c> (same idiom as <see cref="AttentionBf16"/>). The element addressing
    /// (<c>kv_base + t*kv_dim + d</c>) is identical to fp32; per element <c>e</c> the packed word
    /// is <c>e&gt;&gt;1</c> and the component is <c>e&amp;1</c> (head_dim/kv_dim are even — see the
    /// GpuForwardPass guard). All scores / softmax / value accumulation stay fp32; only the
    /// stored K/V mantissa is narrowed. The companion (dtype-agnostic, reads the fp32 partial
    /// buffers) <see cref="AttentionSplitKvCombine"/> is reused unchanged.
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim, uint seq_len, uint n_splits, uint window }.
    /// Bindings: 0=Q[num_heads*head_dim] (float), 1=K_cache (uint, fp16-packed),
    ///           2=V_cache (uint, fp16-packed), 3=partial_o[num_heads*n_splits*head_dim] (float),
    ///           4=partial_meta[num_heads*n_splits*2] (float).
    /// </summary>
    internal const string AttentionSplitKvPartialBf16 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q          { float q_data[]; };
        layout(binding = 1) readonly buffer KCache     { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache     { uint v_cache[]; };
        layout(binding = 3) buffer PartialO            { float partial_o[]; };
        layout(binding = 4) buffer PartialMeta         { float partial_meta[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint n_splits;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        const uint CHUNK = 256u;
        shared float sk_scores[256];   // per-slice scores (≤ CHUNK)
        shared float sdata[256];       // reduction scratch

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint h = gl_WorkGroupID.x;   // query head
            uint s = gl_WorkGroupID.y;   // KV split
            if (h >= num_heads || s >= n_splits) return;

            uint meta_off = (h * n_splits + s) * 2u;
            // SWA bound — mirrors the fp32 split-KV partial.
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;
            uint t0 = s * CHUNK;
            uint t1 = t0 + CHUNK; if (t1 > seq_len) t1 = seq_len;
            // Empty for this split: out-of-range (t0 >= seq_len) OR entirely below the sliding
            // window (t1 <= start_seq). Mark empty and bail so the combine skips it
            // (scale = exp(−inf − gmax) = 0) and never reads a stale numerator.
            if (t0 >= seq_len || t1 <= start_seq) {
                if (tid == 0u) { partial_meta[meta_off] = -1.0/0.0; partial_meta[meta_off + 1u] = 0.0; }
                return;
            }
            // Clamp the slice's start to the window so positions < start_seq never contribute.
            if (t0 < start_seq) t0 = start_seq;
            uint n = t1 - t0;   // 1 ≤ n ≤ CHUNK

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim  = num_kv_heads * head_dim;
            float scale  = inversesqrt(float(head_dim));
            uint q_off   = h * head_dim;
            uint kv_base = t0 * kv_dim + kv_head * head_dim;   // first row of this (clamped) slice for this kv head

            // ─── Phase 1: scores for the slice → shared (indexed t − t0) ───
            // Read each packed fp16 word once (two K elements at a time). kv_base + t*kv_dim is
            // even (head_dim is even — see the GpuForwardPass guard) so >>1 is the exact word base
            // and consecutive d,d+1 are the two halves of word k_off_half+dh — mirrors AttentionBf16.
            for (uint t = tid; t < n; t += 256u) {
                float dot = 0.0;
                uint k_off_half = (kv_base + t * kv_dim) >> 1;
                for (uint dh = 0u; dh < (head_dim >> 1); dh++) {
                    uint d = dh << 1;
                    vec2 kv = unpackHalf2x16(k_cache[k_off_half + dh]);
                    dot += q_data[q_off + d] * kv.x + q_data[q_off + d + 1u] * kv.y;
                }
                sk_scores[t] = dot * scale;
            }
            barrier();

            // ─── Phase 2: local max over the slice ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < n; t += 256u) local_max = max(local_max, sk_scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] = max(sdata[tid], sdata[tid + r]);
                barrier();
            }
            float m_i = sdata[0];
            barrier();

            // exp(score − m_i) in place + local denom.
            float local_sum = 0.0;
            for (uint t = tid; t < n; t += 256u) {
                float e = exp(sk_scores[t] - m_i);
                sk_scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] += sdata[tid + r];
                barrier();
            }
            float l_i = sdata[0];
            barrier();

            if (tid == 0u) { partial_meta[meta_off] = m_i; partial_meta[meta_off + 1u] = l_i; }

            // ─── Phase 3: UN-normalized weighted-V numerator for this slice ───
            // Each thread owns ONE output dim d. Hoist the per-d word/component selection out of the
            // t-loop and walk the V row word base incrementally (kv_base>>1 is this slice's t=0 word
            // base; head_dim is even) — mirrors AttentionBf16's Phase 3.
            uint o_off = (h * n_splits + s) * head_dim;
            for (uint d = tid; d < head_dim; d += 256u) {
                uint d_half = d >> 1;
                uint component = d & 1u;
                uint v_off_half = (kv_base >> 1) + d_half;
                uint kv_dim_half = kv_dim >> 1;
                float acc = 0.0;
                for (uint t = 0u; t < n; t++) {
                    float vv = unpackHalf2x16(v_cache[v_off_half])[component];
                    acc += sk_scores[t] * vv;
                    v_off_half += kv_dim_half;
                }
                partial_o[o_off + d] = acc;
            }
        }
        """;

    /// <summary>
    /// q8_0 (issue #332) variant of <see cref="AttentionSplitKvPartial"/>: control flow is
    /// IDENTICAL to the fp32 partial; the ONLY difference is that the K/V cache buffers
    /// (bindings 1, 2) hold ggml <c>block_q8_0</c> (34 bytes/block = fp16 scale + 32 int8) and
    /// every element read becomes a byte-gather + dequant <c>value = fp16(d) * int8</c> — the
    /// same <c>loadK</c>/<c>loadV</c> idiom as <see cref="AttentionQ8_0"/>. Element addressing
    /// (<c>kv_base + t*kv_dim + d</c>) is identical to fp32; per absolute element <c>e</c>:
    /// <c>blk=e&gt;&gt;5</c>, <c>lane=e&amp;31</c>, <c>b0=blk*34</c>. kv_dim%32==0 (enforced in
    /// GpuForwardPass), so a KV row's blocks never straddle a row. All scores / softmax / value
    /// accumulation stay fp32; only the stored K/V is narrowed. The companion
    /// <see cref="AttentionSplitKvCombine"/> (reads the fp32 partial buffers) is reused unchanged.
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim, uint seq_len, uint n_splits, uint window }.
    /// Bindings: 0=Q[num_heads*head_dim] (float), 1=K_cache (uint, block_q8_0),
    ///           2=V_cache (uint, block_q8_0), 3=partial_o[num_heads*n_splits*head_dim] (float),
    ///           4=partial_meta[num_heads*n_splits*2] (float).
    /// </summary>
    internal const string AttentionSplitKvPartialQ8 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q          { float q_data[]; };
        layout(binding = 1) readonly buffer KCache     { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache     { uint v_cache[]; };
        layout(binding = 3) buffer PartialO            { float partial_o[]; };
        layout(binding = 4) buffer PartialMeta         { float partial_meta[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint n_splits;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        const uint CHUNK = 256u;
        shared float sk_scores[256];   // per-slice scores (≤ CHUNK)
        shared float sdata[256];       // reduction scratch

        // Sign-extend a single int8 byte in one bitfieldExtract (no ternary branch).
        int gInt8K(uint b) { return bitfieldExtract(int(k_cache[b >> 2]), int((b & 3u) * 8u), 8); }
        int gInt8V(uint b) { return bitfieldExtract(int(v_cache[b >> 2]), int((b & 3u) * 8u), 8); }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint h = gl_WorkGroupID.x;   // query head
            uint s = gl_WorkGroupID.y;   // KV split
            if (h >= num_heads || s >= n_splits) return;

            uint meta_off = (h * n_splits + s) * 2u;
            // SWA bound — mirrors the fp32 split-KV partial.
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;
            uint t0 = s * CHUNK;
            uint t1 = t0 + CHUNK; if (t1 > seq_len) t1 = seq_len;
            // Empty for this split: out-of-range (t0 >= seq_len) OR entirely below the sliding
            // window (t1 <= start_seq). Mark empty and bail so the combine skips it
            // (scale = exp(−inf − gmax) = 0) and never reads a stale numerator.
            if (t0 >= seq_len || t1 <= start_seq) {
                if (tid == 0u) { partial_meta[meta_off] = -1.0/0.0; partial_meta[meta_off + 1u] = 0.0; }
                return;
            }
            // Clamp the slice's start to the window so positions < start_seq never contribute.
            // kv_base stays a multiple of 32 (kv_dim & head_dim are multiples of 32), so the
            // block addressing (kv_base >> 5) is still exact after the clamp.
            if (t0 < start_seq) t0 = start_seq;
            uint n = t1 - t0;   // 1 ≤ n ≤ CHUNK

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim  = num_kv_heads * head_dim;
            float scale  = inversesqrt(float(head_dim));
            uint q_off   = h * head_dim;
            uint kv_base = t0 * kv_dim + kv_head * head_dim;   // first row of this (clamped) slice for this kv head

            // ─── Phase 1: scores for the slice → shared (indexed t − t0) ───
            // Load each block's fp16 scale ONCE per 32-element block (head_dim & kv_dim are
            // multiples of 32 — enforced in GpuForwardPass), then dequant the 32 int8 lanes with it.
            // Mirrors AttentionQ8_0's read pattern; scale-once instead of per-element loadK.
            for (uint t = tid; t < n; t += 256u) {
                float dot = 0.0;
                uint k_off = kv_base + t * kv_dim;
                uint blk_start = k_off >> 5;
                for (uint blk = 0u; blk < (head_dim >> 5); blk++) {
                    uint b0 = (blk_start + blk) * 34u;
                    // b0 = blk*34 is even, so the two scale bytes [b0, b0+1] live in the same word.
                    uint w = k_cache[b0 >> 2];
                    float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
                    uint q_blk_off = q_off + blk * 32u;
                    for (uint lane = 0u; lane < 32u; lane++) {
                        dot += q_data[q_blk_off + lane] * (dsc * float(gInt8K(b0 + 2u + lane)));
                    }
                }
                sk_scores[t] = dot * scale;
            }
            barrier();

            // ─── Phase 2: local max over the slice ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < n; t += 256u) local_max = max(local_max, sk_scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] = max(sdata[tid], sdata[tid + r]);
                barrier();
            }
            float m_i = sdata[0];
            barrier();

            // exp(score − m_i) in place + local denom.
            float local_sum = 0.0;
            for (uint t = tid; t < n; t += 256u) {
                float e = exp(sk_scores[t] - m_i);
                sk_scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] += sdata[tid + r];
                barrier();
            }
            float l_i = sdata[0];
            barrier();

            if (tid == 0u) { partial_meta[meta_off] = m_i; partial_meta[meta_off + 1u] = l_i; }

            // ─── Phase 3: UN-normalized weighted-V numerator for this slice ───
            // Each thread owns ONE output dim d. Hoist the block index to a linear recurrence over t
            // (base_blk = this slice's t=0 block for dim d; stride_blk = kv_dim in blocks) so the
            // per-block scale is read once per t — mirrors AttentionQ8_0's Phase 3.
            uint o_off = (h * n_splits + s) * head_dim;
            for (uint d = tid; d < head_dim; d += 256u) {
                uint d_blk = d >> 5;
                uint lane = d & 31u;
                uint base_blk = (kv_base >> 5) + d_blk;
                uint stride_blk = kv_dim >> 5;
                float acc = 0.0;
                for (uint t = 0u; t < n; t++) {
                    uint b0 = (base_blk + t * stride_blk) * 34u;
                    uint w = v_cache[b0 >> 2];
                    float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
                    float vv = dsc * float(gInt8V(b0 + 2u + lane));
                    acc += sk_scores[t] * vv;
                }
                partial_o[o_off + d] = acc;
            }
        }
        """;

    internal const string VisionPixelShuffle2x2 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint gridY;
            uint gridX;
            uint inDim;
        };

        void main() {
            uint outY = gridY / 2u;
            uint outX = gridX / 2u;
            uint totalTokens = outY * outX;

            uint tokenIdx = gl_GlobalInvocationID.x;
            if (tokenIdx >= totalTokens) return;

            uint ty = tokenIdx / outX;
            uint tx = tokenIdx % outX;
            uint py0 = ty * 2u;
            uint px0 = tx * 2u;
            uint outDim = inDim * 4u;

            uint p00 = (py0 * gridX + px0) * inDim;
            uint p01 = (py0 * gridX + px0 + 1u) * inDim;
            uint p10 = ((py0 + 1u) * gridX + px0) * inDim;
            uint p11 = ((py0 + 1u) * gridX + px0 + 1u) * inDim;
            uint dstOff = tokenIdx * outDim;

            for (uint c = 0u; c < inDim; c++) {
                output_data[dstOff + c]              = input_data[p00 + c];
                output_data[dstOff + inDim + c]      = input_data[p01 + c];
                output_data[dstOff + inDim * 2u + c] = input_data[p10 + c];
                output_data[dstOff + inDim * 3u + c] = input_data[p11 + c];
            }
        }
        """;

    internal const string VisionMRoPE = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer QBuffer { float q_data[]; };
        layout(binding = 1) buffer KBuffer { float k_data[]; };

        layout(push_constant) uniform Params {
            uint patchesX;
            uint patchesY;
            uint qHeads;
            uint kvHeads;
            uint headDim;
            float theta;
        };

        void main() {
            uint tokenIdx = gl_GlobalInvocationID.x;
            uint totalTokens = patchesX * patchesY;
            if (tokenIdx >= totalTokens) return;

            uint py = tokenIdx / patchesX;
            uint px = tokenIdx % patchesX;
            uint mropeHalf = headDim / 4u;

            for (uint h = 0u; h < qHeads; h++) {
                uint headOff = (tokenIdx * qHeads + h) * headDim;
                for (uint d = 0u; d < mropeHalf; d++) {
                    float freqX = pow(theta, -2.0 * float(d) / float(headDim));
                    float cosX = cos(float(px) * freqX);
                    float sinX = sin(float(px) * freqX);

                    float q0 = q_data[headOff + d];
                    float q1 = q_data[headOff + d + mropeHalf];
                    q_data[headOff + d]             = q0 * cosX - q1 * sinX;
                    q_data[headOff + d + mropeHalf] = q0 * sinX + q1 * cosX;

                    uint ySecOff = headOff + 2u * mropeHalf;
                    float freqY = pow(theta, -2.0 * float(d) / float(headDim));
                    float cosY = cos(float(py) * freqY);
                    float sinY = sin(float(py) * freqY);

                    float qY0 = q_data[ySecOff + d];
                    float qY1 = q_data[ySecOff + d + mropeHalf];
                    q_data[ySecOff + d]             = qY0 * cosY - qY1 * sinY;
                    q_data[ySecOff + d + mropeHalf] = qY0 * sinY + qY1 * cosY;
                }
            }

            for (uint h = 0u; h < kvHeads; h++) {
                uint headOff = (tokenIdx * kvHeads + h) * headDim;
                for (uint d = 0u; d < mropeHalf; d++) {
                    float freqX = pow(theta, -2.0 * float(d) / float(headDim));
                    float cosX = cos(float(px) * freqX);
                    float sinX = sin(float(px) * freqX);

                    float k0 = k_data[headOff + d];
                    float k1 = k_data[headOff + d + mropeHalf];
                    k_data[headOff + d]             = k0 * cosX - k1 * sinX;
                    k_data[headOff + d + mropeHalf] = k0 * sinX + k1 * cosX;

                    uint ySecOff = headOff + 2u * mropeHalf;
                    float freqY = pow(theta, -2.0 * float(d) / float(headDim));
                    float cosY = cos(float(py) * freqY);
                    float sinY = sin(float(py) * freqY);

                    float kY0 = k_data[ySecOff + d];
                    float kY1 = k_data[ySecOff + d + mropeHalf];
                    k_data[ySecOff + d]             = kY0 * cosY - kY1 * sinY;
                    k_data[ySecOff + d + mropeHalf] = kY0 * sinY + kY1 * cosY;
                }
            }
        }
        """;

    internal const string VisionContinuous2DRoPE = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer QBuffer { float q_data[]; };
        layout(binding = 1) buffer KBuffer { float k_data[]; };

        layout(push_constant) uniform Params {
            uint patchesX;
            uint patchesY;
            uint heads;
            uint headDim;
            float theta;
        };

        void main() {
            uint tokenIdx = gl_GlobalInvocationID.x;
            uint totalTokens = patchesX * patchesY;
            if (tokenIdx >= totalTokens) return;

            uint py = tokenIdx / patchesX;
            uint px = tokenIdx % patchesX;
            uint halfDim = headDim / 2u;
            uint quarterDim = halfDim / 2u;

            for (uint h = 0u; h < heads; h++) {
                uint headOff = (tokenIdx * heads + h) * headDim;

                for (uint d = 0u; d < quarterDim; d++) {
                    float freq = pow(theta, -(2.0 * float(d)) / float(halfDim));
                    float cosX = cos(float(px) * freq);
                    float sinX = sin(float(px) * freq);

                    uint i0 = headOff + d * 2u;
                    uint i1 = headOff + d * 2u + 1u;

                    float q0 = q_data[i0], q1 = q_data[i1];
                    q_data[i0] = q0 * cosX - q1 * sinX;
                    q_data[i1] = q0 * sinX + q1 * cosX;

                    float k0 = k_data[i0], k1 = k_data[i1];
                    k_data[i0] = k0 * cosX - k1 * sinX;
                    k_data[i1] = k0 * sinX + k1 * cosX;
                }

                for (uint d = 0u; d < quarterDim; d++) {
                    float freq = pow(theta, -(2.0 * float(d)) / float(halfDim));
                    float cosY = cos(float(py) * freq);
                    float sinY = sin(float(py) * freq);

                    uint i0 = headOff + halfDim + d * 2u;
                    uint i1 = headOff + halfDim + d * 2u + 1u;

                    float q0 = q_data[i0], q1 = q_data[i1];
                    q_data[i0] = q0 * cosY - q1 * sinY;
                    q_data[i1] = q0 * sinY + q1 * cosY;

                    float k0 = k_data[i0], k1 = k_data[i1];
                    k_data[i0] = k0 * cosY - k1 * sinY;
                    k_data[i1] = k0 * sinY + k1 * cosY;
                }
            }
        }
        """;

    internal const string VisionLayerNorm = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) readonly  buffer Weight { float weight_data[]; };
        layout(binding = 2) readonly  buffer Bias   { float bias_data[];   };
        layout(binding = 3) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint nTokens;
            uint embd;
            float eps;
            uint hasBias;
        };

        void main() {
            uint t = gl_GlobalInvocationID.x;
            if (t >= nTokens) return;

            uint off = t * embd;
            float sum = 0.0;
            for (uint i = 0u; i < embd; i++) sum += input_data[off + i];
            float mean = sum / float(embd);

            float sumSq = 0.0;
            for (uint i = 0u; i < embd; i++) {
                float diff = input_data[off + i] - mean;
                sumSq += diff * diff;
            }
            float invStd = inversesqrt(sumSq / float(embd) + eps);

            for (uint i = 0u; i < embd; i++) {
                float normalized = (input_data[off + i] - mean) * invStd;
                float w = weight_data[i];
                float b = (hasBias != 0u) ? bias_data[i] : 0.0;
                output_data[off + i] = normalized * w + b;
            }
        }
        """;

    internal const string VisionGelu = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint n;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= n) return;
            float v = data[idx];
            data[idx] = 0.5 * v * (1.0 + tanh(0.79788456 * (v + 0.044715 * v * v * v)));
        }
        """;

    internal const string VisionQuickGelu = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint n;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= n) return;
            float v = data[idx];
            data[idx] = v * (1.0 / (1.0 + exp(-1.702 * v)));
        }
        """;

    internal const string VisionSquaredRelu = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint n;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= n) return;
            float v = data[idx];
            float r = max(0.0, v);
            data[idx] = r * r;
        }
        """;
}
