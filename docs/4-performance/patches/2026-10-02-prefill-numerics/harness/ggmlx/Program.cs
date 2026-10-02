using System.Globalization;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

// ggml vs Stingray int8-activation matvec, on identical rows, against an FP64 operation reference.
// args: model.gguf llamaCppDir
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
unsafe
{
    string dir = args[1];
    IntPtr baseLib = NativeLibrary.Load(Path.Combine(dir, "ggml-base.dll"));
    IntPtr cpuLib = NativeLibrary.Load(Path.Combine(dir, "ggml-cpu-haswell.dll"));
    var getTraits = (delegate* unmanaged[Cdecl]<int, IntPtr>)NativeLibrary.GetExport(baseLib, "ggml_get_type_traits");
    var getCpuTraits = (delegate* unmanaged[Cdecl]<int, IntPtr>)NativeLibrary.GetExport(cpuLib, "ggml_get_type_traits_cpu");
    var cpuInit = (delegate* unmanaged[Cdecl]<void>)NativeLibrary.GetExport(cpuLib, "ggml_cpu_init");
    cpuInit();

    const int GGML_Q4_K = 12, GGML_Q6_K = 14, GGML_Q8_K = 15;
    // ggml_type_traits: name@0 blck@8 interleave@16 type_size@24 is_quantized@32 to_float@40 from_float_ref@48
    // ggml_type_traits_cpu: from_float@0 vec_dot@8 vec_dot_type@16 nrows@24
    IntPtr q8kTraits = getTraits(GGML_Q8_K);
    var q8kToFloat = (delegate* unmanaged[Cdecl]<void*, float*, long, void>)*(IntPtr*)((byte*)q8kTraits + 40);
    var q8kFromRef = (delegate* unmanaged[Cdecl]<float*, void*, long, void>)*(IntPtr*)((byte*)q8kTraits + 48);
    IntPtr q8kCpu = getCpuTraits(GGML_Q8_K);
    var q8kFromSimd = (delegate* unmanaged[Cdecl]<float*, void*, long, void>)*(IntPtr*)((byte*)q8kCpu + 0);
    Console.WriteLine($"ptrs: q8k.to_float {(nint)q8kToFloat:X} q8k.from_ref {(nint)q8kFromRef:X} q8k.cpu.from_float {(nint)q8kFromSimd:X} q4k.vec_dot {*(IntPtr*)((byte*)getCpuTraits(GGML_Q4_K) + 8):X}");

    // ── Q1_0 / Q2_0 dequant vs ggml's own to_float, random blocks (independent check for item 3a) ──
    foreach (var (gt, dt, blk, bytes) in new[] { (41, DType.Q1_0, 128, 18), (42, DType.Q2_0, 64, 18) })
    {
        IntPtr tr = getTraits(gt);
        var toF = (delegate* unmanaged[Cdecl]<void*, float*, long, void>)*(IntPtr*)((byte*)tr + 40);
        Console.WriteLine($"type {gt} ({dt}): ggml blck {*(long*)((byte*)tr + 8)} size {*(long*)((byte*)tr + 24)} to_float {(nint)toF:X}");
        if ((nint)toF == 0) continue;
        var r2 = new Random(gt);
        int nBlocks = 4096, n = nBlocks * blk;
        var src = new byte[nBlocks * bytes];
        r2.NextBytes(src);
        for (int bI = 0; bI < nBlocks; bI++) // finite fp16 scales
        {
            ushort h = BitConverter.HalfToUInt16Bits((Half)((r2.NextDouble() - 0.5) * 0.2));
            src[bI * bytes] = (byte)h; src[bI * bytes + 1] = (byte)(h >> 8);
        }
        var a = new float[n]; var b = new float[n];
        fixed (byte* sp = src) fixed (float* ap = a) toF(sp, ap, n);
        Dequantize.ToFloat32(src, b, dt, n);
        int diff = 0; for (int i = 0; i < n; i++) if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) diff++;
        Console.WriteLine($"  {dt}: {n} values, {diff} differ from ggml (bitwise)");
    }

    var model = GgufModel.Open(args[0]);
    var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
    int e = hp.EmbeddingDim;
    Console.WriteLine($"model {Path.GetFileName(args[0])} n_embd {e}; ggml Q8_K type_size {*(long*)((byte*)q8kTraits + 24)}");

    // ── inputs: real layer-0 inputs (RMSNorm(embed row) * attn_norm) + synthetic heavy-tailed rows ──
    var rnd = new Random(1);
    var tok = GgufTokenizer.FromGgufModel(model);
    var embT = model.FindTensor("token_embd.weight")!.Value;
    var normT = model.FindTensor("blk.0.attn_norm.weight")!.Value;
    float[] Deq(GgufTensorInfo t, long offElems, int count)
    {
        var data = model.GetTensorData(t);
        long bpb = DTypeInfo.ByteSize(DTypeInfo.BlockSize(t.DType), t.DType);
        long byteOff = offElems / DTypeInfo.BlockSize(t.DType) * bpb;
        var dst = new float[count];
        Dequantize.ToFloat32(data.Slice((int)byteOff, (int)DTypeInfo.ByteSize(count, t.DType)), dst, t.DType, count);
        return dst;
    }
    var normW = Deq(normT, 0, e);
    var inputs = new List<(string Name, float[] X)>();
    foreach (int id in tok.Encode("The quick brown fox jumps over the lazy dog, 1234 {code} \n").Take(6))
    {
        var row = Deq(embT, (long)id * e, e);
        double ss = 0; foreach (var v in row) ss += v * v;
        float inv = (float)(1.0 / Math.Sqrt(ss / e + hp.RmsNormEps));
        var x = new float[e]; for (int i = 0; i < e; i++) x[i] = row[i] * inv * normW[i];
        inputs.Add(($"real L0 tok{id}", x));
    }
    foreach (double target in new[] { 25.0, 50.0, 90.0 })
        for (int k = 0; k < 2; k++)
        {
            var x = new float[e];
            for (int i = 0; i < e; i++) x[i] = (float)Gauss(rnd);
            for (int o = 0; o < 3; o++) x[rnd.Next(e)] = (float)(target * (rnd.Next(2) == 0 ? 1 : -1)); // amax/rms ~ target
            inputs.Add(($"synth amax/rms~{target}", x));
        }

    // ── weights: real rows ──
    foreach (var (tname, ggmlType) in new[] { ("blk.0.ffn_gate.weight", GGML_Q4_K), ("blk.0.attn_q.weight", GGML_Q4_K), ("blk.0.attn_v.weight", GGML_Q6_K) })
    {
        var tn = model.FindTensor(tname);
        if (tn is null) { Console.WriteLine($"{tname}: missing"); continue; }
        var t = tn.Value;
        if ((int)t.DType != ggmlType) { Console.WriteLine($"{tname}: dtype {t.DType}, skipping (expected ggml type {ggmlType})"); continue; }
        int rows = Math.Min(512, (int)t.Dimensions[1]);
        int cols = (int)t.Dimensions[0];
        if (cols != e) { Console.WriteLine($"{tname}: cols {cols} != n_embd"); continue; }
        var data = model.GetTensorData(t);
        long rowBytes = DTypeInfo.ByteSize(cols, t.DType);
        var wq = data.Slice(0, (int)(rowBytes * rows)).ToArray();
        var wf = new float[(long)rows * cols];
        Dequantize.ToFloat32(wq, wf, t.DType, (long)rows * cols);
        IntPtr cpuT = getCpuTraits(ggmlType);
        var vecDot = (delegate* unmanaged[Cdecl]<int, float*, nuint, void*, nuint, void*, nuint, int, void>)*(IntPtr*)((byte*)cpuT + 8);
        int vdt = *(int*)((byte*)cpuT + 16);

        Console.WriteLine($"\n=== {tname} ({t.DType}, {rows} rows x {cols}); ggml vec_dot_type {vdt}");
        Console.WriteLine($"{"input",-22} | act-quant relRMS: ggml Q8_K  ours Q8_KS | dot relRMS vs FP64: q-only ggml  q-only ours | KERNEL ggml   KERNEL ours | absMax ggml  absMax ours | |y| rms");
        foreach (var (name, x) in inputs)
        {
            int nb = cols / 256;
            // ggml Q8_K (the SIMD quantizer the runtime uses) and its dequant
            var qg = new byte[nb * 292]; var dg = new float[cols];
            fixed (float* xp = x) fixed (byte* qp = qg) fixed (float* dp = dg) {
                q8kFromSimd(xp, qp, cols);
                for (int bb = 0; bb < nb; bb++)
                {
                    float dd = *(float*)(qp + bb * 292); sbyte* qq = (sbyte*)(qp + bb * 292 + 4);
                    for (int i = 0; i < 256; i++) dp[bb * 256 + i] = dd * qq[i];
                }
            }
            // ours Q8_KS and its dequant (x = d[sub] * q)
            var qo = new byte[SimdKernels.Q8KSScratchBytes(cols)]; var dO = new float[cols];
            fixed (float* xp = x) fixed (byte* qp = qo)
            {
                SimdKernels.QuantizeRowToQ8KS(xp, cols, qp);
                float* d = (float*)qp; sbyte* qs = (sbyte*)(qp + nb * 32);
                for (int i = 0; i < cols; i++) dO[i] = d[i / 32] * qs[i];
            }
            double Rel(float[] a, float[] r) { double e2 = 0, r2 = 0; for (int i = 0; i < a.Length; i++) { double dd = a[i] - r[i]; e2 += dd * dd; r2 += (double)r[i] * r[i]; } return Math.Sqrt(e2 / r2); }

            var yRef = new double[rows]; var yQg = new double[rows]; var yQo = new double[rows];
            var yKg = new float[rows]; var yKo = new float[rows];
            for (int r = 0; r < rows; r++)
            {
                double s0 = 0, s1 = 0, s2 = 0; long o = (long)r * cols;
                for (int i = 0; i < cols; i++) { double w = wf[o + i]; s0 += w * x[i]; s1 += w * dg[i]; s2 += w * dO[i]; }
                yRef[r] = s0; yQg[r] = s1; yQo[r] = s2;
            }
            fixed (byte* wp = wq) fixed (byte* qp = qg) fixed (float* kg = yKg)
                for (int r = 0; r < rows; r++) vecDot(cols, kg + r, 0, wp + r * rowBytes, 0, qp, 0, 1);
            fixed (byte* wp = wq) fixed (float* xp = x) fixed (float* ko = yKo)
                SimdKernels.MatVec(ko, wp, xp, rows, cols, t.DType);

            double RelD(double[] a, double[] r) { double e2 = 0, r2 = 0; for (int i = 0; i < a.Length; i++) { double dd = a[i] - r[i]; e2 += dd * dd; r2 += r[i] * r[i]; } return Math.Sqrt(e2 / r2); }
            double RelF(float[] a, double[] r) { double e2 = 0, r2 = 0; for (int i = 0; i < a.Length; i++) { double dd = a[i] - r[i]; e2 += dd * dd; r2 += r[i] * r[i]; } return Math.Sqrt(e2 / r2); }
            double AbsMax(float[] a, double[] r) { double m = 0; for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - r[i])); return m; }
            double yRms = Math.Sqrt(yRef.Average(v => v * v));
            Console.WriteLine($"{name,-22} | {Rel(dg, x),18:E2} {Rel(dO, x),11:E2} | {RelD(yQg, yRef),22:E2} {RelD(yQo, yRef),11:E2} | {RelF(yKg, yRef),11:E2} {RelF(yKo, yRef),13:E2} | {AbsMax(yKg, yRef),11:E2} {AbsMax(yKo, yRef),11:E2} | {yRms:E2}");
        }
    }

    static double Gauss(Random r) { double u1 = 1 - r.NextDouble(), u2 = r.NextDouble(); return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2); }
}
