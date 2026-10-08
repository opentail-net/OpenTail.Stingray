using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class MuseGlimmerSyntheticTests : IDisposable
{
    private const int EmbDim = 64;
    private const int FfnDim = 128;
    private const int NumLayers = 4;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 16;
    private const int QDim = NumHeads * HeadDim; // 64
    private const int KvDim = NumKvHeads * HeadDim; // 32
    private const int Vocab = 128;
    private const int Context = 64;
    private const int SwaWindow = 3;
    private const int SwaPeriod = 2; // P=2: layers 0, 2 are SWA; layers 1, 3 are full attention
    private const float LogitScale = 0.19611613513818404f;
    private const float Softcap = 20.0f;
    private const float QkScaleFactor = 3.87f;
    private const float NormEps = 1e-6f;
    private const float PostNormEps = 1e-8f;

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { }
        }
    }

    [Fact]
    public void MuseGlimmer_SyntheticSpecificationTest_MatchesReferenceWithinTolerance()
    {
        Environment.SetEnvironmentVariable("STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH", "1");
        try
        {
            int seed = 42;
            var (path, tensors) = WriteSyntheticMuseGlimmerGguf(seed);

            using var model = GgufModel.Open(path);
            var hp = ArchitectureModelResolver.ResolveHyperparams(model);

            // Assert ModelGraph properly detected all Muse-Glimmer architecture flags
            Assert.True(hp.InputEmbeddingRmsNorm);
            Assert.True(hp.AttentionOutputGate);
            Assert.True(hp.HasPostAttnNorm);
            Assert.True(hp.HasPostFfwNorm);
            Assert.True(hp.RopeOnlySwaLayers);
            Assert.Equal(PostNormEps, hp.PostNormEps);
            Assert.Equal(LogitScale, hp.LogitScale);
            Assert.Equal(Softcap, hp.FinalLogitSoftcap);
            Assert.NotNull(hp.IsSwaLayer);
            Assert.Equal(SwaWindow, hp.SlidingWindowSize);
            Assert.True(hp.IsSwaLayer[0]);
            Assert.False(hp.IsSwaLayer[1]);
            Assert.True(hp.IsSwaLayer[2]);
            Assert.False(hp.IsSwaLayer[3]);

            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);

            // 6-token prompt
            int[] prompt = [5, 12, 23, 7, 44, 18];

            var refState = new MuseGlimmerReference(tensors, hp);

            for (int pos = 0; pos < prompt.Length; pos++)
            {
                int token = prompt[pos];
                var actualLogits = fwd.Forward(token, pos).ToArray();
                var expectedLogits = refState.Step(token, pos);

                Assert.Equal(Vocab, actualLogits.Length);
                Assert.Equal(Vocab, expectedLogits.Length);

                float maxDelta = 0.0f;
                for (int v = 0; v < Vocab; v++)
                {
                    float delta = MathF.Abs(actualLogits[v] - expectedLogits[v]);
                    if (delta > maxDelta) maxDelta = delta;
                }

                Assert.True(maxDelta < 1e-3f, $"Position {pos} max logit delta {maxDelta} exceeded 1e-3 tolerance.");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH", null);
        }
    }

    private (string Path, Dictionary<string, float[]> Tensors) WriteSyntheticMuseGlimmerGguf(int seed)
    {
        var tensorDict = new Dictionary<string, float[]>(StringComparer.Ordinal);
        var tensors = new List<(string name, long[] dims, float[] data)>();

        void AddTensor(string name, long[] dims, float[] data)
        {
            tensors.Add((name, dims, data));
            tensorDict[name] = data;
        }

        AddTensor("token_embd.weight", [EmbDim, Vocab], RandF32(Vocab * EmbDim, seed + 1));
        AddTensor("output_norm.weight", [EmbDim], OnesF32(EmbDim));
        AddTensor("output.weight", [EmbDim, Vocab], RandF32(Vocab * EmbDim, seed + 2));

        for (int l = 0; l < NumLayers; l++)
        {
            int b = seed + 100 * (l + 1);
            // Four per-layer norms carry the (1 + learned_weight) conversion shift
            AddTensor($"blk.{l}.attn_norm.weight", [EmbDim], ShiftedNormF32(EmbDim, b + 11));
            AddTensor($"blk.{l}.attn_q.weight", [EmbDim, QDim], RandF32(QDim * EmbDim, b + 1));
            AddTensor($"blk.{l}.attn_k.weight", [EmbDim, KvDim], RandF32(KvDim * EmbDim, b + 2));
            AddTensor($"blk.{l}.attn_v.weight", [EmbDim, KvDim], RandF32(KvDim * EmbDim, b + 3));
            AddTensor($"blk.{l}.attn_gate.weight", [EmbDim, QDim], RandF32(QDim * EmbDim, b + 4));
            // Synthesized QK-norm in GGUF: attn_q_norm absorbs qk_scale_factor (3.87), attn_k_norm is 1.0
            AddTensor($"blk.{l}.attn_q_norm.weight", [HeadDim], ConstF32(HeadDim, QkScaleFactor));
            AddTensor($"blk.{l}.attn_k_norm.weight", [HeadDim], OnesF32(HeadDim));
            AddTensor($"blk.{l}.attn_output.weight", [QDim, EmbDim], RandF32(EmbDim * QDim, b + 5));
            AddTensor($"blk.{l}.post_attention_norm.weight", [EmbDim], ShiftedNormF32(EmbDim, b + 12));

            AddTensor($"blk.{l}.ffn_norm.weight", [EmbDim], ShiftedNormF32(EmbDim, b + 13));
            AddTensor($"blk.{l}.ffn_gate.weight", [EmbDim, FfnDim], RandF32(FfnDim * EmbDim, b + 6));
            AddTensor($"blk.{l}.ffn_up.weight", [EmbDim, FfnDim], RandF32(FfnDim * EmbDim, b + 7));
            AddTensor($"blk.{l}.ffn_down.weight", [FfnDim, EmbDim], RandF32(EmbDim * FfnDim, b + 8));
            AddTensor($"blk.{l}.post_ffw_norm.weight", [EmbDim], ShiftedNormF32(EmbDim, b + 14));
        }

        var metadata = new (string key, GgufValueType type, object value)[]
        {
            ("general.architecture", GgufValueType.String, "muse-glimmer"),
            ("general.name", GgufValueType.String, "synthetic-muse-glimmer"),
            ("muse-glimmer.block_count", GgufValueType.Int32, NumLayers),
            ("muse-glimmer.context_length", GgufValueType.Int32, Context),
            ("muse-glimmer.embedding_length", GgufValueType.Int32, EmbDim),
            ("muse-glimmer.feed_forward_length", GgufValueType.Int32, FfnDim),
            ("muse-glimmer.vocab_size", GgufValueType.UInt64, (ulong)Vocab),
            ("muse-glimmer.attention.head_count", GgufValueType.Int32, NumHeads),
            ("muse-glimmer.attention.head_count_kv", GgufValueType.Int32, NumKvHeads),
            ("muse-glimmer.attention.key_length", GgufValueType.Int32, HeadDim),
            ("muse-glimmer.attention.layer_norm_rms_epsilon", GgufValueType.Float32, NormEps),
            ("muse-glimmer.rope.freq_base", GgufValueType.Float32, 10_000f),
            ("muse-glimmer.attention.sliding_window", GgufValueType.Int32, SwaWindow),
            ("muse-glimmer.attention.sliding_window_pattern", GgufValueType.Int32, SwaPeriod),
            ("muse-glimmer.logit_scale", GgufValueType.Float32, LogitScale),
            ("muse-glimmer.final_logit_softcapping", GgufValueType.Float32, Softcap),
        };

        var path = Path.Combine(Path.GetTempPath(), $"muse_glimmer_synth_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

        using (var fs = File.Create(path))
        {
            var w = new ScalarGgufWriter(fs);
            const int alignment = 32;
            w.WriteHeader(version: 3, tensorCount: (ulong)tensors.Count, metadataKvCount: (ulong)metadata.Length);
            foreach (var (key, type, val) in metadata)
                w.WriteMetadataKv(key, type, val);

            ulong offset = 0;
            foreach (var (name, dims, data) in tensors)
            {
                w.WriteTensorInfo(name, dims, DType.Float32, offset);
                offset += AlignUp((ulong)data.Length * sizeof(float), alignment);
            }
            w.PadToAlignment(alignment);

            foreach (var (_, _, data) in tensors)
            {
                long before = w.BytesWritten;
                w.WriteFloats(data);
                w.PadTo(before + (long)AlignUp((ulong)data.Length * sizeof(float), alignment));
            }
        }

        return (path, tensorDict);
    }

    private static ulong AlignUp(ulong v, int a) => (v + (ulong)a - 1) / (ulong)a * (ulong)a;

    private static float[] OnesF32(int n)
    {
        var a = new float[n];
        Array.Fill(a, 1f);
        return a;
    }

    private static float[] RandF32(int n, int seed)
    {
        var a = new float[n];
        uint s = (uint)seed | 1u;
        for (int i = 0; i < n; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            a[i] = ((s & 0xFFFF) / 65535f - 0.5f) * 0.05f;
        }
        return a;
    }

    private static float[] ConstF32(int n, float val)
    {
        var a = new float[n];
        Array.Fill(a, val);
        return a;
    }

    private static float[] ShiftedNormF32(int n, int seed)
    {
        // Models per-layer norm weights converted to GGUF format: data_torch = data_torch + 1
        var a = RandF32(n, seed);
        for (int i = 0; i < n; i++) a[i] += 1.0f;
        return a;
    }

    /*
      Independent reference implementation of the Muse-Glimmer text tower specification:
        h  = rmsnorm_unweighted(embed)                       (eps = norm eps)
        per layer:
          a  = rmsnorm(h, attn_norm)
          q,k,v,g = W·a;  q,k per-head RMSNorm;  RoPE (interleaved) on SWA layers only (full = NoPE)
          attention: causal; SWA layers see keys with p1 - p0 < n_swa (window); SWA iff l % P < P-1
          attn *= sigmoid(g);  o = Wo·attn
          h1 = h  + rmsnorm(o, post_attention_norm, eps = 1e-8)
          h2 = h1 + rmsnorm(SwiGLU(rmsnorm(h1, ffn_norm)), post_ffw_norm, eps = 1e-8)
        logits = tanh(lm_head(rmsnorm(h2)) * logit_scale / cap) * cap
    */
    private sealed unsafe class MuseGlimmerReference
    {
        private readonly Dictionary<string, float[]> _weights;
        private readonly ModelHyperparams _hp;
        private readonly List<float[]>[] _kCache = new List<float[]>[NumLayers];
        private readonly List<float[]>[] _vCache = new List<float[]>[NumLayers];

        public MuseGlimmerReference(Dictionary<string, float[]> weights, ModelHyperparams hp)
        {
            _weights = weights;
            _hp = hp;
            for (int l = 0; l < NumLayers; l++)
            {
                _kCache[l] = [];
                _vCache[l] = [];
            }
        }

        public float[] Step(int token, int pos)
        {
            // 1. Token embedding lookup: token_embd.weight is [EmbDim, Vocab] row-major in memory
            // Column token: index c * EmbDim
            var embd = new float[EmbDim];
            var embdWeight = _weights["token_embd.weight"];
            for (int d = 0; d < EmbDim; d++)
            {
                embd[d] = embdWeight[token * EmbDim + d];
            }

            // 2. Unweighted RMSNorm on embedding before layer 0
            var h = new float[EmbDim];
            PureRmsNorm(embd, h, NormEps);

            // 3. Transformer layers
            for (int l = 0; l < NumLayers; l++)
            {
                // Attn Norm
                var a = new float[EmbDim];
                RmsNorm(h, _weights[$"blk.{l}.attn_norm.weight"], a, NormEps);

                // Projections: Q, K, V, Gate
                var q = new float[QDim];
                var k = new float[KvDim];
                var v = new float[KvDim];
                var gate = new float[QDim];

                MatVec(_weights[$"blk.{l}.attn_q.weight"], a, q, QDim, EmbDim);
                MatVec(_weights[$"blk.{l}.attn_k.weight"], a, k, KvDim, EmbDim);
                MatVec(_weights[$"blk.{l}.attn_v.weight"], a, v, KvDim, EmbDim);
                MatVec(_weights[$"blk.{l}.attn_gate.weight"], a, gate, QDim, EmbDim);

                // Per-head RMSNorm on Q and K
                var qNormW = _weights[$"blk.{l}.attn_q_norm.weight"];
                var kNormW = _weights[$"blk.{l}.attn_k_norm.weight"];
                for (int head = 0; head < NumHeads; head++)
                {
                    RmsNorm(q.AsSpan(head * HeadDim, HeadDim), qNormW, q.AsSpan(head * HeadDim, HeadDim), NormEps);
                }
                for (int head = 0; head < NumKvHeads; head++)
                {
                    RmsNorm(k.AsSpan(head * HeadDim, HeadDim), kNormW, k.AsSpan(head * HeadDim, HeadDim), NormEps);
                }

                // RoPE on SWA layers only (l % P < P - 1)
                bool isSwa = (l % SwaPeriod) < (SwaPeriod - 1);
                if (isSwa)
                {
                    fixed (float* pQ = q)
                        SimdKernels.ApplyRoPE(pQ, pos, NumHeads, HeadDim, 10_000f);
                    fixed (float* pK = k)
                        SimdKernels.ApplyRoPE(pK, pos, NumKvHeads, HeadDim, 10_000f);
                }

                // Store K and V in cache
                _kCache[l].Add((float[])k.Clone());
                _vCache[l].Add((float[])v.Clone());

                // Causal Attention with optional SWA windowing
                var attnOut = new float[QDim];
                float scale = 1.0f / MathF.Sqrt(HeadDim);

                int kvGroup = NumHeads / NumKvHeads;
                int totalPast = _kCache[l].Count;
                Span<float> scores = new float[totalPast];

                for (int head = 0; head < NumHeads; head++)
                {
                    int kvHead = head / kvGroup;
                    var qHead = q.AsSpan(head * HeadDim, HeadDim);

                    for (int j = 0; j < totalPast; j++)
                    {
                        if (isSwa && (pos - j) >= SwaWindow)
                        {
                            scores[j] = float.NegativeInfinity;
                        }
                        else
                        {
                            var kjHead = _kCache[l][j].AsSpan(kvHead * HeadDim, HeadDim);
                            float dot = 0.0f;
                            for (int d = 0; d < HeadDim; d++) dot += qHead[d] * kjHead[d];
                            scores[j] = dot * scale;
                        }
                    }

                    // Softmax
                    Softmax(scores);

                    // Accumulate V
                    var outHead = attnOut.AsSpan(head * HeadDim, HeadDim);
                    for (int j = 0; j < totalPast; j++)
                    {
                        float s = scores[j];
                        if (s <= 0.0f) continue;
                        var vjHead = _vCache[l][j].AsSpan(kvHead * HeadDim, HeadDim);
                        for (int d = 0; d < HeadDim; d++) outHead[d] += s * vjHead[d];
                    }
                }

                // Attention Output Gate: attn *= sigmoid(g)
                for (int i = 0; i < QDim; i++)
                {
                    float sig = 1.0f / (1.0f + MathF.Exp(-gate[i]));
                    attnOut[i] *= sig;
                }

                // Output projection Wo
                var o = new float[EmbDim];
                MatVec(_weights[$"blk.{l}.attn_output.weight"], attnOut, o, EmbDim, QDim);

                // Post-attention norm with eps = 1e-8
                var oNorm = new float[EmbDim];
                RmsNorm(o, _weights[$"blk.{l}.post_attention_norm.weight"], oNorm, PostNormEps);

                // Residual 1
                var h1 = new float[EmbDim];
                for (int d = 0; d < EmbDim; d++) h1[d] = h[d] + oNorm[d];

                // FFN Norm
                var ffnIn = new float[EmbDim];
                RmsNorm(h1, _weights[$"blk.{l}.ffn_norm.weight"], ffnIn, NormEps);

                // SwiGLU FFN
                var fGate = new float[FfnDim];
                var fUp = new float[FfnDim];
                MatVec(_weights[$"blk.{l}.ffn_gate.weight"], ffnIn, fGate, FfnDim, EmbDim);
                MatVec(_weights[$"blk.{l}.ffn_up.weight"], ffnIn, fUp, FfnDim, EmbDim);

                var swiglu = new float[FfnDim];
                for (int d = 0; d < FfnDim; d++)
                {
                    float g = fGate[d];
                    float swish = g * (1.0f / (1.0f + MathF.Exp(-g)));
                    swiglu[d] = swish * fUp[d];
                }

                var ffnOut = new float[EmbDim];
                MatVec(_weights[$"blk.{l}.ffn_down.weight"], swiglu, ffnOut, EmbDim, FfnDim);

                // Post-FFW norm with eps = 1e-8
                var ffnOutNorm = new float[EmbDim];
                RmsNorm(ffnOut, _weights[$"blk.{l}.post_ffw_norm.weight"], ffnOutNorm, PostNormEps);

                // Residual 2
                for (int d = 0; d < EmbDim; d++) h[d] = h1[d] + ffnOutNorm[d];
            }

            // 4. Output norm
            var finalH = new float[EmbDim];
            RmsNorm(h, _weights["output_norm.weight"], finalH, NormEps);

            // 5. LM Head logits
            var logits = new float[Vocab];
            MatVec(_weights["output.weight"], finalH, logits, Vocab, EmbDim);

            // 6. Scale and Softcap
            for (int v = 0; v < Vocab; v++)
            {
                float scaled = logits[v] * LogitScale;
                logits[v] = MathF.Tanh(scaled / Softcap) * Softcap;
            }

            return logits;
        }

        private static void PureRmsNorm(ReadOnlySpan<float> src, Span<float> dst, float eps)
        {
            float sumSq = 0.0f;
            for (int i = 0; i < src.Length; i++) sumSq += src[i] * src[i];
            float invRms = 1.0f / MathF.Sqrt(sumSq / src.Length + eps);
            for (int i = 0; i < src.Length; i++) dst[i] = src[i] * invRms;
        }

        private static void RmsNorm(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float eps)
        {
            float sumSq = 0.0f;
            for (int i = 0; i < src.Length; i++) sumSq += src[i] * src[i];
            float invRms = 1.0f / MathF.Sqrt(sumSq / src.Length + eps);
            for (int i = 0; i < src.Length; i++) dst[i] = src[i] * invRms * weight[i];
        }

        private static void MatVec(ReadOnlySpan<float> matrix, ReadOnlySpan<float> vec, Span<float> dst, int rows, int cols)
        {
            for (int r = 0; r < rows; r++)
            {
                int rowStart = r * cols;
                float sum = 0.0f;
                for (int c = 0; c < cols; c++) sum += matrix[rowStart + c] * vec[c];
                dst[r] = sum;
            }
        }

        private static void Softmax(Span<float> x)
        {
            float maxVal = float.NegativeInfinity;
            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] > maxVal) maxVal = x[i];
            }

            if (float.IsNegativeInfinity(maxVal)) return;

            float sumExp = 0.0f;
            for (int i = 0; i < x.Length; i++)
            {
                if (float.IsNegativeInfinity(x[i]))
                {
                    x[i] = 0.0f;
                }
                else
                {
                    float e = MathF.Exp(x[i] - maxVal);
                    x[i] = e;
                    sumExp += e;
                }
            }

            float invSum = 1.0f / (sumExp + 1e-9f);
            for (int i = 0; i < x.Length; i++) x[i] *= invSum;
        }
    }

    private sealed class ScalarGgufWriter(Stream stream)
    {
        public long BytesWritten { get; private set; }

        public void WriteHeader(uint version, ulong tensorCount, ulong metadataKvCount)
        {
            WriteUInt32(0x46554747); // "GGUF"
            WriteUInt32(version);
            WriteUInt64(tensorCount);
            WriteUInt64(metadataKvCount);
        }

        public void WriteMetadataKv(string key, GgufValueType type, object value)
        {
            WriteString(key);
            WriteUInt32((uint)type);
            switch (type)
            {
                case GgufValueType.String: WriteString((string)value); break;
                case GgufValueType.Int32: WriteInt32((int)value); break;
                case GgufValueType.UInt64: WriteUInt64((ulong)value); break;
                case GgufValueType.Float32: WriteFloat32((float)value); break;
                case GgufValueType.Bool: WriteByte((bool)value ? (byte)1 : (byte)0); break;
                default: throw new NotSupportedException($"type {type}");
            }
        }

        public void WriteTensorInfo(string name, long[] dims, DType type, ulong offset)
        {
            WriteString(name);
            WriteUInt32((uint)dims.Length);
            for (int i = 0; i < dims.Length; i++) WriteUInt64((ulong)dims[i]);
            WriteUInt32((uint)type);
            WriteUInt64(offset);
        }

        public void PadToAlignment(int alignment)
        {
            long rem = BytesWritten % alignment;
            if (rem == 0) return;
            int pad = alignment - (int)rem;
            Span<byte> zeros = stackalloc byte[pad];
            stream.Write(zeros);
            BytesWritten += pad;
        }

        public void PadTo(long absoluteOffset)
        {
            long diff = absoluteOffset - BytesWritten;
            if (diff <= 0) return;
            Span<byte> zeros = stackalloc byte[(int)Math.Min(diff, 64)];
            while (diff > 0)
            {
                int chunk = (int)Math.Min(diff, zeros.Length);
                stream.Write(zeros[..chunk]);
                BytesWritten += chunk;
                diff -= chunk;
            }
        }

        public void WriteFloats(ReadOnlySpan<float> data)
        {
            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(data);
            stream.Write(bytes);
            BytesWritten += bytes.Length;
        }

        private void WriteByte(byte v) { stream.WriteByte(v); BytesWritten++; }
        private void WriteInt32(int v) => WriteUInt32((uint)v);
        private void WriteUInt32(uint v)
        {
            Span<byte> b = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(b, v);
            stream.Write(b);
            BytesWritten += 4;
        }
        private void WriteUInt64(ulong v)
        {
            Span<byte> b = stackalloc byte[8];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(b, v);
            stream.Write(b);
            BytesWritten += 8;
        }
        private void WriteFloat32(float v) => WriteUInt32(BitConverter.SingleToUInt32Bits(v));
        private void WriteString(string s)
        {
            var utf8 = System.Text.Encoding.UTF8.GetBytes(s);
            WriteUInt64((ulong)utf8.Length);
            stream.Write(utf8);
            BytesWritten += utf8.Length;
        }
    }
}
