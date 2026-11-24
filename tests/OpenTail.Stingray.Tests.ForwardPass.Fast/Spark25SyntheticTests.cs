using System.Linq;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

// spark2_5 (Spark-X). Structural and specification checks on a tiny synthetic GGUF. Not real-weight evidence (see
// docs/2-coverage/2026-11-24-spark-x2.6-port-plan.md). The reference below is written from llama.cpp spark2-5.cpp
// and the NEOX/SWA/rope conventions, not from ForwardPass.
public sealed class Spark25SyntheticTests : IDisposable
{
    private const int EmbDim = 64;
    private const int FfnDim = 128;
    private const int NumLayers = 4;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 16;
    private const int QDim = NumHeads * HeadDim;   // 64
    private const int KvDim = NumKvHeads * HeadDim; // 32
    private const int Vocab = 128;
    private const int Context = 64;
    private const int SwaWindow = 3;
    private const int SwaPeriod = 2;   // layers 0 and 2 are SWA (i % 2 < 1), layers 1 and 3 are full attention
    private const int RopeFull = 8;    // rope.dimension_count: partial rotary on full-attention layers
    private const int RopeSwa = 16;    // rope.dimension_count_swa: full head on SWA layers
    private const float BaseFull = 10000f;
    private const float BaseSwa = 500f;
    private const float NormEps = 1e-6f;

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { }
        }
    }

    [Fact]
    public void Spark25_Hyperparameters_SetTheSparkSemantics()
    {
        Environment.SetEnvironmentVariable("STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH", "1");
        try
        {
            var (path, _) = WriteSyntheticSparkGguf(seed: 7);
            using var model = GgufModel.Open(path);
            var hp = ArchitectureModelResolver.ResolveHyperparams(model);

            Assert.True(hp.AttentionOutputGate);
            Assert.True(hp.AttentionOutputGatePerHead);
            Assert.False(hp.RopeOnlySwaLayers);
            Assert.Equal(1f, hp.EmbeddingScale);
            Assert.False(hp.InputEmbeddingRmsNorm);
            Assert.Equal(FfnActivation.GeluApprox, hp.FfnActivation);
            Assert.True(hp.IsNeoxRope);
            Assert.True(hp.RopeWidthPerLayer);
            Assert.Equal(SwaWindow, hp.SlidingWindowSize);
            Assert.NotNull(hp.IsSwaLayer);
            Assert.True(hp.IsSwaLayer[0]);
            Assert.False(hp.IsSwaLayer[1]);
            Assert.True(hp.IsSwaLayer[2]);
            Assert.False(hp.IsSwaLayer[3]);
            Assert.NotNull(hp.LayerRopeDim);
            Assert.Equal(RopeSwa, hp.LayerRopeDim[0]);
            Assert.Equal(RopeFull, hp.LayerRopeDim[1]);
            Assert.Equal(RopeSwa, hp.LayerRopeDim[2]);
            Assert.Equal(RopeFull, hp.LayerRopeDim[3]);
            Assert.Equal(0f, hp.FinalLogitSoftcap);
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH", null);
        }
    }

    [Fact]
    public void Spark25_SyntheticForward_MatchesSpecificationReference()
    {
        Environment.SetEnvironmentVariable("STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH", "1");
        try
        {
            var (path, tensors) = WriteSyntheticSparkGguf(seed: 42);
            using var model = GgufModel.Open(path);
            var hp = ArchitectureModelResolver.ResolveHyperparams(model);

            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);
            var reference = new SparkReference(tensors);

            int[] prompt = [5, 12, 23, 7, 44, 18, 99, 3];
            var report = new List<string>();
            bool ok = true;
            for (int pos = 0; pos < prompt.Length; pos++)
            {
                var actual = fwd.Forward(prompt[pos], pos).ToArray();
                var expected = reference.Step(prompt[pos], pos);

                Assert.Equal(Vocab, actual.Length);
                float maxDelta = 0f, maxAbs = 0f;
                for (int v = 0; v < Vocab; v++)
                {
                    maxDelta = MathF.Max(maxDelta, MathF.Abs(actual[v] - expected[v]));
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(expected[v]));
                }
                report.Add($"pos {pos}: maxDelta {maxDelta:G4} (max|logit| {maxAbs:G4})");
                if (!(maxDelta < 5e-4f)) ok = false;
            }
            // Measured 2026-11-24: worst delta 2.1e-4 (pos 4). Position 0 is 1.3e-7. The residual is at the level the
            // engine's approximate exp (SimdKernels.ExpApprox256, used by softmax and GELU) can produce; not verified.
            Assert.True(ok, "Logit deltas exceed 5e-4: " + string.Join("; ", report));
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH", null);
        }
    }

    private (string Path, Dictionary<string, float[]> Tensors) WriteSyntheticSparkGguf(int seed)
    {
        var tensorDict = new Dictionary<string, float[]>(StringComparer.Ordinal);
        var tensors = new List<(string name, long[] dims, float[] data)>();

        void AddTensor(string name, long[] dims, float[] data)
        {
            tensors.Add((name, dims, data));
            tensorDict[name] = data;
        }

        AddTensor("token_embd.weight", [EmbDim, Vocab], RandF32(Vocab * EmbDim, seed + 1));
        AddTensor("output_norm.weight", [EmbDim], NormF32(EmbDim, seed + 2));
        AddTensor("output.weight", [EmbDim, Vocab], RandF32(Vocab * EmbDim, seed + 3));

        for (int l = 0; l < NumLayers; l++)
        {
            int b = seed + 100 * (l + 1);
            AddTensor($"blk.{l}.attn_norm.weight", [EmbDim], NormF32(EmbDim, b + 11));
            AddTensor($"blk.{l}.attn_q.weight", [EmbDim, QDim], RandF32(QDim * EmbDim, b + 1));
            AddTensor($"blk.{l}.attn_k.weight", [EmbDim, KvDim], RandF32(KvDim * EmbDim, b + 2));
            AddTensor($"blk.{l}.attn_v.weight", [EmbDim, KvDim], RandF32(KvDim * EmbDim, b + 3));
            // Spark's output gate: one value per head ([embDim -> numHeads]).
            AddTensor($"blk.{l}.attn_gate.weight", [EmbDim, NumHeads], RandF32(NumHeads * EmbDim, b + 4));
            AddTensor($"blk.{l}.attn_output.weight", [QDim, EmbDim], RandF32(EmbDim * QDim, b + 5));
            AddTensor($"blk.{l}.ffn_norm.weight", [EmbDim], NormF32(EmbDim, b + 13));
            AddTensor($"blk.{l}.ffn_gate.weight", [EmbDim, FfnDim], RandF32(FfnDim * EmbDim, b + 6));
            AddTensor($"blk.{l}.ffn_up.weight", [EmbDim, FfnDim], RandF32(FfnDim * EmbDim, b + 7));
            AddTensor($"blk.{l}.ffn_down.weight", [FfnDim, EmbDim], RandF32(EmbDim * FfnDim, b + 8));
        }

        var metadata = new (string key, GgufValueType type, object value)[]
        {
            ("general.architecture", GgufValueType.String, "spark2_5"),
            ("general.name", GgufValueType.String, "synthetic-spark2_5"),
            ("spark2_5.block_count", GgufValueType.Int32, NumLayers),
            ("spark2_5.context_length", GgufValueType.Int32, Context),
            ("spark2_5.embedding_length", GgufValueType.Int32, EmbDim),
            ("spark2_5.feed_forward_length", GgufValueType.Int32, FfnDim),
            ("spark2_5.vocab_size", GgufValueType.UInt64, (ulong)Vocab),
            ("spark2_5.attention.head_count", GgufValueType.Int32, NumHeads),
            ("spark2_5.attention.head_count_kv", GgufValueType.Int32, NumKvHeads),
            ("spark2_5.attention.key_length", GgufValueType.Int32, HeadDim),
            ("spark2_5.attention.value_length", GgufValueType.Int32, HeadDim),
            ("spark2_5.attention.layer_norm_rms_epsilon", GgufValueType.Float32, NormEps),
            ("spark2_5.rope.freq_base", GgufValueType.Float32, BaseFull),
            ("spark2_5.rope.freq_base_swa", GgufValueType.Float32, BaseSwa),
            ("spark2_5.rope.dimension_count", GgufValueType.Int32, RopeFull),
            ("spark2_5.rope.dimension_count_swa", GgufValueType.Int32, RopeSwa),
            ("spark2_5.attention.sliding_window", GgufValueType.Int32, SwaWindow),
            ("spark2_5.attention.sliding_window_pattern", GgufValueType.Int32, SwaPeriod),
        };

        var path = Path.Combine(Path.GetTempPath(), $"spark25_synth_{Guid.NewGuid():N}.gguf");
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

    /// <summary>
    /// Per-token reference: tensors are stored as [inner, outer] so row o of a [cols, rows] matrix is contiguous.
    /// Spark rules: RMSNorm; NEOX (rotate-half) RoPE over the first n_rot dims of each head, n_rot = 16 on SWA layers and
    /// 8 otherwise, base 500 on SWA layers and 10000 otherwise; SWA keeps keys with pos - j &lt; window; per-head sigmoid
    /// gate on the attention output; GELU (tanh approximation) gated FFN; no embedding scale; no logit softcap.
    /// </summary>
    private sealed class SparkReference(IReadOnlyDictionary<string, float[]> t)
    {
        private readonly List<float[]>[] _keys = Enumerable.Range(0, NumLayers).Select(_ => new List<float[]>()).ToArray();
        private readonly List<float[]>[] _values = Enumerable.Range(0, NumLayers).Select(_ => new List<float[]>()).ToArray();

        public float[] Step(int token, int pos)
        {
            var x = new float[EmbDim];
            Array.Copy(t["token_embd.weight"], token * EmbDim, x, 0, EmbDim);

            for (int l = 0; l < NumLayers; l++)
            {
                bool swa = l % SwaPeriod < SwaPeriod - 1;
                int nRot = swa ? RopeSwa : RopeFull;
                float theta = swa ? BaseSwa : BaseFull;

                var h = RmsNorm(x, t[$"blk.{l}.attn_norm.weight"]);
                var q = MatVec(t[$"blk.{l}.attn_q.weight"], h, QDim, EmbDim);
                var k = MatVec(t[$"blk.{l}.attn_k.weight"], h, KvDim, EmbDim);
                var v = MatVec(t[$"blk.{l}.attn_v.weight"], h, KvDim, EmbDim);
                var gate = MatVec(t[$"blk.{l}.attn_gate.weight"], h, NumHeads, EmbDim);

                NeoxRope(q, NumHeads, nRot, theta, pos);
                NeoxRope(k, NumKvHeads, nRot, theta, pos);
                _keys[l].Add(k);
                _values[l].Add(v);

                var attn = new float[QDim];
                int first = swa ? Math.Max(0, pos - SwaWindow + 1) : 0;
                float scale = 1f / MathF.Sqrt(HeadDim);
                int group = NumHeads / NumKvHeads;
                for (int hh = 0; hh < NumHeads; hh++)
                {
                    int kvh = hh / group;
                    var scores = new float[pos - first + 1];
                    for (int j = first; j <= pos; j++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < HeadDim; d++)
                            dot += q[hh * HeadDim + d] * _keys[l][j][kvh * HeadDim + d];
                        scores[j - first] = dot * scale;
                    }
                    Softmax(scores);
                    float g = Sigmoid(gate[hh]);
                    for (int d = 0; d < HeadDim; d++)
                    {
                        float acc = 0f;
                        for (int j = first; j <= pos; j++)
                            acc += scores[j - first] * _values[l][j][kvh * HeadDim + d];
                        attn[hh * HeadDim + d] = acc * g;
                    }
                }

                var o = MatVec(t[$"blk.{l}.attn_output.weight"], attn, EmbDim, QDim);
                for (int i = 0; i < EmbDim; i++) x[i] += o[i];

                var h2 = RmsNorm(x, t[$"blk.{l}.ffn_norm.weight"]);
                var fg = MatVec(t[$"blk.{l}.ffn_gate.weight"], h2, FfnDim, EmbDim);
                var fu = MatVec(t[$"blk.{l}.ffn_up.weight"], h2, FfnDim, EmbDim);
                var mid = new float[FfnDim];
                for (int i = 0; i < FfnDim; i++) mid[i] = Gelu(fg[i]) * fu[i];
                var fd = MatVec(t[$"blk.{l}.ffn_down.weight"], mid, EmbDim, FfnDim);
                for (int i = 0; i < EmbDim; i++) x[i] += fd[i];
            }

            var final = RmsNorm(x, t["output_norm.weight"]);
            return MatVec(t["output.weight"], final, Vocab, EmbDim);
        }

        private static void NeoxRope(float[] vec, int heads, int nRot, float theta, int pos)
        {
            int half = nRot / 2;
            for (int hh = 0; hh < heads; hh++)
            {
                int off = hh * HeadDim;
                for (int i = 0; i < half; i++)
                {
                    float freq = MathF.Pow(theta, -2f * i / nRot);
                    float angle = pos * freq;
                    float c = MathF.Cos(angle), s = MathF.Sin(angle);
                    float x0 = vec[off + i], x1 = vec[off + i + half];
                    vec[off + i] = x0 * c - x1 * s;
                    vec[off + i + half] = x0 * s + x1 * c;
                }
            }
        }

        private static float[] RmsNorm(ReadOnlySpan<float> src, float[] weight)
        {
            float ss = 0f;
            foreach (var value in src) ss += value * value;
            float scale = 1f / MathF.Sqrt(ss / src.Length + NormEps);
            var dst = new float[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = src[i] * scale * weight[i];
            return dst;
        }

        private static float[] MatVec(float[] matrix, ReadOnlySpan<float> vec, int rows, int cols)
        {
            var dst = new float[rows];
            for (int o = 0; o < rows; o++)
            {
                float acc = 0f;
                for (int i = 0; i < cols; i++) acc += matrix[o * cols + i] * vec[i];
                dst[o] = acc;
            }
            return dst;
        }

        private static void Softmax(float[] x)
        {
            float max = x.Max();
            float sum = 0f;
            for (int i = 0; i < x.Length; i++) { x[i] = MathF.Exp(x[i] - max); sum += x[i]; }
            for (int i = 0; i < x.Length; i++) x[i] /= sum;
        }

        private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

        private static float Gelu(float x) =>
            0.5f * x * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.044715f * x * x * x)));
    }

    private static ulong AlignUp(ulong v, int a) => (v + (ulong)a - 1) / (ulong)a * (ulong)a;

    private static float[] NormF32(int n, int seed)
    {
        var a = RandF32(n, seed);
        for (int i = 0; i < n; i++) a[i] += 1f;
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
