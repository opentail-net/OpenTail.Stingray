using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenTail.Stingray.Engine.Calibration;

/// <summary>
/// Measured numbers for one expert weight format on this machine (see <see cref="ExpertCalibration"/>).
/// Rates are achieved, not peak: the real kernels, run the way the expert loop runs them (many experts in parallel).
/// </summary>
/// <param name="Dtype">Weight format, e.g. <c>Q4_K</c>.</param>
/// <param name="Rows">Output rows of the benchmarked matrix (an expert's gate/up projection shape).</param>
/// <param name="Cols">Input columns; a multiple of the format's block size.</param>
/// <param name="PrefillTokens">Rows per expert used for the prefill measurement (the profile of OLMoE averages 93).</param>
/// <param name="CpuDecodeGBs">CPU expert matmul at one token: weight bytes consumed per second (decode is weight-streaming bound).</param>
/// <param name="CpuPrefillGflops">CPU expert matmul at <paramref name="PrefillTokens"/> rows, aggregate GFLOP/s over all workers.</param>
/// <param name="CpuPrefillMsPerExpert">Wall time one expert matmul takes at <paramref name="PrefillTokens"/> rows, per worker.</param>
/// <param name="GpuCopyGBs">Host-to-device copy bandwidth in GB/s, or null without a GPU.</param>
/// <param name="GpuMatmulGflops">GPU batched matmul with resident weights (16 tokens per dispatch, the Vulkan limit), GFLOP/s.</param>
/// <param name="CpuOverlapGflops">CPU prefill rate while the host-to-device copy runs concurrently (they contend for DRAM).</param>
/// <param name="CopyOverlapGBs">Copy bandwidth while the CPU matmul runs concurrently.</param>
public sealed record DtypeCalibration(
    string Dtype, int Rows, int Cols, int PrefillTokens,
    double CpuDecodeGBs, double CpuPrefillGflops, double CpuPrefillMsPerExpert,
    double? GpuCopyGBs, double? GpuMatmulGflops, double? CpuOverlapGflops, double? CopyOverlapGBs)
{
    /// <summary>Bytes of one expert matrix in this format.</summary>
    public long ExpertBytes => (long)Rows * (Cols / DTypeInfo.BlockSize(Enum.Parse<DType>(Dtype))) * DTypeInfo.BytesPerBlock(Enum.Parse<DType>(Dtype));
}

/// <summary>A calibration run: where it ran and what it measured. Stored as JSON in the model home.</summary>
public sealed record CalibrationProfile(
    int Schema, string CreatedUtc, string DeviceKey, string Cpu, string? GpuName, bool? GpuIntegrated,
    IReadOnlyList<DtypeCalibration> Dtypes);

/// <summary>What the measurements say to do; every line carries the numbers it rests on.</summary>
public sealed record CalibrationAdvice(string DecodeColdExperts, string PrefillResidentExperts, string PrefillStreamedExperts, string Note);

/// <summary>
/// Measured chooser (Phase 5 of <c>docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md</c>, modelled on FreeToken's
/// <c>ft bench bw</c>): instead of guessing whether expert work belongs on the CPU or the GPU, time the real kernels on
/// this machine and decide from the ratios. It measures, per expert format: the CPU expert matmul at one token (decode) and
/// at prefill row counts, the host-to-device copy, the GPU batched matmul on resident weights, and the CPU matmul and the
/// copy running <b>at the same time</b> (they share DRAM, so the standalone numbers cannot predict the overlapped ones).
///
/// <para>Rules (explicit, so a result can be argued with): cold experts at decode go to the CPU when the CPU's weight
/// consumption rate exceeds <see cref="CpuOverModelCopyThreshold"/> times the copy bandwidth (FreeToken uses 2x), otherwise they
/// are fetched; resident-weight prefill goes to the GPU when its matmul rate is at least <see cref="GpuPrefillWinMargin"/> times
/// the CPU's; streamed prefill compares the time to copy and run an expert on the GPU with the time to run it on the CPU.</para>
///
/// <para>Today's engine has no batched GPU MoE prefill (Phase 3) and no streamed prefill (Phase 4), so the prefill advice
/// describes what those phases should do on this device, not something the engine can already act on; the decode advice and the
/// profile are what later phases will read.</para>
/// </summary>
public static class ExpertCalibration
{
    /// <summary>FreeToken's rule: compute on the CPU when its consumption rate exceeds this multiple of the PCIe/copy rate.</summary>
    public const double CpuOverModelCopyThreshold = 2.0;

    /// <summary>The GPU must beat the CPU by this factor before resident-weight prefill is recommended there.</summary>
    public const double GpuPrefillWinMargin = 1.25;

    public const int SchemaVersion = 1;

    // ── measurement ──────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs every measurement for <paramref name="dtypes"/> (Q4_K and Q6_K are supported). <paramref name="secondsEach"/> is the window
    /// per measurement; the GPU parts are skipped when <paramref name="useGpu"/> is false or no Vulkan device opens.
    /// </summary>
    public static CalibrationProfile Run(IReadOnlyList<DType> dtypes, int rows, int cols, int prefillTokens, double secondsEach, bool useGpu,
        Action<string>? progress = null)
    {
        VulkanBackend? gpu = null;
        if (useGpu)
        {
            try { gpu = new VulkanBackend(); }
            catch (Exception ex) { progress?.Invoke($"no Vulkan device ({ex.Message}); GPU measurements skipped"); }
        }

        try
        {
            var results = new List<DtypeCalibration>();
            foreach (var dt in dtypes)
            {
                progress?.Invoke($"{dt}: CPU decode");
                var decode = MeasureCpu(dt, rows, cols, 1, secondsEach, null);
                progress?.Invoke($"{dt}: CPU prefill ({prefillTokens} rows per expert)");
                var prefill = MeasureCpu(dt, rows, cols, prefillTokens, secondsEach, null);

                double? copy = null, gpuMatmul = null, cpuOv = null, copyOv = null;
                if (gpu is not null)
                {
                    progress?.Invoke($"{dt}: host-to-device copy");
                    copy = MeasureCopy(gpu, secondsEach, null);
                    progress?.Invoke($"{dt}: GPU matmul on resident weights");
                    gpuMatmul = MeasureGpuMatmul(gpu, dt, rows, cols, secondsEach);
                    progress?.Invoke($"{dt}: CPU matmul and copy at the same time");
                    var stop = new CancellationTokenSource();
                    double ovCpu = 0;
                    var worker = Task.Run(() => ovCpu = MeasureCpu(dt, rows, cols, prefillTokens, secondsEach, stop.Token).Gflops);
                    copyOv = MeasureCopy(gpu, secondsEach, null);
                    stop.Cancel();
                    worker.Wait();
                    cpuOv = ovCpu;
                }

                results.Add(new DtypeCalibration(dt.ToString(), rows, cols, prefillTokens,
                    decode.GBs, prefill.Gflops, prefill.MsPerRun, copy, gpuMatmul, cpuOv, copyOv));
            }

            return new CalibrationProfile(SchemaVersion, DateTime.UtcNow.ToString("o"), DeviceKey(gpu), CpuDescription(),
                gpu?.Name, gpu?.IsIntegratedGpu, results);
        }
        finally
        {
            gpu?.Dispose();
        }
    }

    /// <summary>Stable file-name-safe key: the GPU (or "cpu") and the CPU thread count.</summary>
    public static string DeviceKey(VulkanBackend? gpu)
    {
        string g = gpu is null ? "cpu" : gpu.Name;
        var sb = new System.Text.StringBuilder();
        foreach (char c in g) sb.Append(char.IsLetterOrDigit(c) ? char.ToLowerInvariant(c) : '-');
        return $"{sb}_{SimdKernels.CpuThreads}t".Replace("--", "-");
    }

    public static string CpuDescription() =>
        $"{RuntimeInformation.ProcessArchitecture}, {Environment.ProcessorCount} logical / {SimdKernels.CpuThreads} worker threads, " +
        $"AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported}, AVX512F={System.Runtime.Intrinsics.X86.Avx512F.IsSupported}";

    private readonly record struct CpuResult(double GBs, double Gflops, double MsPerRun);

    /// <summary>
    /// The CPU expert matmul the way the expert loop runs it: a bank of experts larger than the last-level cache, one expert per
    /// parallel work item, the real <c>SimdKernels.MatMulBatched</c> (exact path). Runs until the window ends or
    /// <paramref name="stop"/> fires.
    /// </summary>
    private static unsafe CpuResult MeasureCpu(DType dt, int rows, int cols, int tokens, double seconds, CancellationToken? stop)
    {
        const int experts = 64; // 64 x ~1.2 MB: well past any L3, so the weights stream from DRAM like they do in a real run
        long bytesPerRow = (long)(cols / DTypeInfo.BlockSize(dt)) * DTypeInfo.BytesPerBlock(dt);
        long expertBytes = rows * bytesPerRow;
        byte* bank = (byte*)NativeMemory.Alloc((nuint)(experts * expertBytes));
        try
        {
            FillQuantBank(bank, experts * expertBytes, dt);
            int workers = SimdKernels.CpuThreads;
            var inputs = new float[workers][];
            var outputs = new float[workers][];
            var rng = new Random(1);
            for (int w = 0; w < workers; w++)
            {
                inputs[w] = new float[(long)tokens * cols];
                for (int i = 0; i < inputs[w].Length; i++) inputs[w][i] = (float)(rng.NextDouble() * 2 - 1);
                outputs[w] = new float[(long)tokens * rows];
            }

            var opts = new ParallelOptions { MaxDegreeOfParallelism = workers };
            long runs = 0;
            void Pass()
            {
                int next = -1;
                Parallel.For(0, workers, opts, w =>
                {
                    fixed (float* inp = inputs[w]) fixed (float* outp = outputs[w])
                    {
                        // each worker walks its own slice of the bank so no two touch the same expert at once
                        for (int k = 0; k < experts / workers + 1; k++)
                        {
                            int x = (w + k * workers) % experts;
                            SimdKernels.MatMulBatched(outp, bank + x * expertBytes, inp, tokens, rows, cols, dt, allowQ8: false);
                            Interlocked.Increment(ref runs);
                        }
                    }
                });
                Volatile.Read(ref next);
            }

            Pass(); // warm: page-in, JIT
            runs = 0;
            long t0 = Stopwatch.GetTimestamp();
            do { Pass(); }
            while (Stopwatch.GetElapsedTime(t0).TotalSeconds < seconds && stop?.IsCancellationRequested != true);
            double elapsed = Stopwatch.GetElapsedTime(t0).TotalSeconds;

            double gbs = runs * (double)expertBytes / elapsed / 1e9;
            double gflops = runs * 2.0 * rows * cols * tokens / elapsed / 1e9;
            double msPerRun = elapsed * 1000.0 * workers / Math.Max(1, runs); // one worker's wall time per expert
            return new CpuResult(gbs, gflops, msPerRun);
        }
        finally
        {
            NativeMemory.Free(bank);
        }
    }

    /// <summary>Valid-looking quantised blocks: random payload with sane fp16 scales so no value is NaN or denormal-slow.</summary>
    private static unsafe void FillQuantBank(byte* bank, long bytes, DType dt)
    {
        var rng = new Random(7);
        var span = new Span<byte>(bank, checked((int)bytes));
        rng.NextBytes(span);
        int block = DTypeInfo.BytesPerBlock(dt);
        ushort d = BitConverter.HalfToUInt16Bits((Half)0.01f);
        for (long b = 0; b + block <= bytes; b += block)
        {
            if (dt == DType.Q4_K)
            {
                span[(int)b] = (byte)d; span[(int)b + 1] = (byte)(d >> 8);          // d
                span[(int)b + 2] = (byte)d; span[(int)b + 3] = (byte)(d >> 8);      // dmin
            }
            else if (dt == DType.Q6_K)
            {
                span[(int)b + 208] = (byte)d; span[(int)b + 209] = (byte)(d >> 8);  // d at the end of the block
            }
            else throw new NotSupportedException($"calibration supports Q4_K and Q6_K, not {dt}");
        }
    }

    /// <summary>Host-to-device copy bandwidth through the backend's own upload path (a staging buffer and a transfer, waited on).</summary>
    private static double MeasureCopy(VulkanBackend gpu, double seconds, CancellationToken? stop)
    {
        const int floats = 2 * 1024 * 1024; // 8 MB, a few experts' worth
        var host = new float[floats];
        new Random(3).NextSingle();
        var dst = gpu.Allocate(TensorShape.D1(floats));
        try
        {
            gpu.UploadInto(dst, host); // warm
            long copies = 0;
            long t0 = Stopwatch.GetTimestamp();
            do { gpu.UploadInto(dst, host); copies++; }
            while (Stopwatch.GetElapsedTime(t0).TotalSeconds < seconds && stop?.IsCancellationRequested != true);
            return copies * (double)floats * sizeof(float) / Stopwatch.GetElapsedTime(t0).TotalSeconds / 1e9;
        }
        finally { gpu.Free(dst); }
    }

    /// <summary>GPU batched matmul (16 tokens per dispatch, the Vulkan limit) on resident weights, many dispatches per submit.</summary>
    private static double MeasureGpuMatmul(VulkanBackend gpu, DType dt, int rows, int cols, double seconds)
    {
        const int tokens = 16, perSubmit = 32;
        long bytesPerRow = (long)(cols / DTypeInfo.BlockSize(dt)) * DTypeInfo.BytesPerBlock(dt);
        var raw = new byte[rows * bytesPerRow];
        unsafe { fixed (byte* p = raw) FillQuantBank(p, raw.Length, dt); }
        var asFloats = MemoryMarshal.Cast<byte, float>(raw.AsSpan()).ToArray();
        var weights = gpu.Upload(asFloats, TensorShape.D1(asFloats.Length));
        var input = new float[tokens * cols];
        var rng = new Random(5);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var inT = gpu.Upload(input, TensorShape.D1(input.Length));
        var outT = gpu.Allocate(TensorShape.D1((long)tokens * rows));
        try
        {
            void Submit()
            {
                gpu.BeginRecord();
                for (int i = 0; i < perSubmit; i++)
                {
                    gpu.MatMulBatched(outT, weights, inT, tokens, dt, allowInt8: false);
                    gpu.RecordBarrier();
                }
                gpu.EndRecordAndSubmit();
            }
            Submit(); // warm: pipeline creation
            long dispatches = 0;
            long t0 = Stopwatch.GetTimestamp();
            do { Submit(); dispatches += perSubmit; }
            while (Stopwatch.GetElapsedTime(t0).TotalSeconds < seconds);
            return dispatches * 2.0 * rows * cols * tokens / Stopwatch.GetElapsedTime(t0).TotalSeconds / 1e9;
        }
        finally { gpu.Free(weights); gpu.Free(inT); gpu.Free(outT); }
    }

    // ── decision ─────────────────────────────────────────────────────────────────────────────

    /// <summary>Applies the rules to one format's measurements. Pure; the tests drive it with synthetic numbers.</summary>
    public static CalibrationAdvice Advise(DtypeCalibration c)
    {
        string decode, resident, streamed;
        if (c.GpuCopyGBs is not { } copy || c.GpuMatmulGflops is not { } gpuG)
        {
            decode = "CPU (no GPU measured)";
            resident = "CPU (no GPU measured)";
            streamed = "n/a (no GPU measured)";
            return new CalibrationAdvice(decode, resident, streamed, "No GPU numbers: nothing to choose between.");
        }

        // Decode: FreeToken's rule, on the overlapped copy rate when we have it (the regime decode really runs in).
        double copyForRule = c.CopyOverlapGBs ?? copy;
        double ratio = c.CpuDecodeGBs / Math.Max(copyForRule, 1e-9);
        decode = ratio > CpuOverModelCopyThreshold
            ? $"compute on the CPU: weight consumption {c.CpuDecodeGBs:F1} GB/s is {ratio:F1}x the copy rate {copyForRule:F1} GB/s (rule: above {CpuOverModelCopyThreshold:F0}x)"
            : $"fetch to the GPU: the CPU's {c.CpuDecodeGBs:F1} GB/s is only {ratio:F1}x the copy rate {copyForRule:F1} GB/s (rule: CPU above {CpuOverModelCopyThreshold:F0}x)";

        // Prefill with resident weights: GPU matmul rate against the CPU's.
        double cpuG = c.CpuPrefillGflops;
        double gpuVsCpu = gpuG / Math.Max(cpuG, 1e-9);
        resident = gpuVsCpu >= GpuPrefillWinMargin
            ? $"GPU: its matmul {gpuG:F0} GFLOP/s is {gpuVsCpu:F2}x the CPU's {cpuG:F0}"
            : $"CPU: the GPU's {gpuG:F0} GFLOP/s is {gpuVsCpu:F2}x the CPU's {cpuG:F0} (needs {GpuPrefillWinMargin:F2}x)";

        // Prefill with experts that must be copied first: time per expert, copy then GPU compute against CPU compute.
        double copyMs = c.ExpertBytes / (Math.Max(copy, 1e-9) * 1e9) * 1000.0;
        double gpuMs = 2.0 * c.Rows * c.Cols * c.PrefillTokens / (Math.Max(gpuG, 1e-9) * 1e9) * 1000.0;
        double cpuMs = 2.0 * c.Rows * c.Cols * c.PrefillTokens / (Math.Max(cpuG, 1e-9) * 1e9) * 1000.0; // aggregate throughput, like the GPU figure
        streamed = copyMs + gpuMs < cpuMs
            ? $"GPU after copying: {copyMs:F2} ms copy + {gpuMs:F2} ms compute < {cpuMs:F2} ms on the CPU per expert matrix at the CPU's aggregate rate"
            : $"CPU: {copyMs:F2} ms copy + {gpuMs:F2} ms compute is not below {cpuMs:F2} ms on the CPU per expert matrix at the CPU's aggregate rate";

        string note = c.CpuOverlapGflops is { } ov && c.CopyOverlapGBs is { } cov
            ? $"Running together the CPU kept {100 * ov / Math.Max(cpuG, 1e-9):F0}% of its prefill rate and the copy {100 * cov / Math.Max(copy, 1e-9):F0}% of its bandwidth."
            : "";
        return new CalibrationAdvice(decode, resident, streamed, note);
    }

    // ── persistence ──────────────────────────────────────────────────────────────────────────

    public static string ProfilePath(string root, string deviceKey) => Path.Combine(root, "calibration", deviceKey + ".json");

    public static void Save(CalibrationProfile profile, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var stream = File.Create(path);
        using var w = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
        w.WriteStartObject();
        w.WriteNumber("schema", profile.Schema);
        w.WriteString("created_utc", profile.CreatedUtc);
        w.WriteString("device_key", profile.DeviceKey);
        w.WriteString("cpu", profile.Cpu);
        if (profile.GpuName is not null) w.WriteString("gpu", profile.GpuName);
        if (profile.GpuIntegrated is { } integ) w.WriteBoolean("gpu_integrated", integ);
        w.WriteStartArray("dtypes");
        foreach (var d in profile.Dtypes)
        {
            w.WriteStartObject();
            w.WriteString("dtype", d.Dtype);
            w.WriteNumber("rows", d.Rows); w.WriteNumber("cols", d.Cols); w.WriteNumber("prefill_tokens", d.PrefillTokens);
            w.WriteNumber("cpu_decode_gbs", d.CpuDecodeGBs);
            w.WriteNumber("cpu_prefill_gflops", d.CpuPrefillGflops);
            w.WriteNumber("cpu_prefill_ms_per_expert", d.CpuPrefillMsPerExpert);
            if (d.GpuCopyGBs is { } a) w.WriteNumber("gpu_copy_gbs", a);
            if (d.GpuMatmulGflops is { } b) w.WriteNumber("gpu_matmul_gflops", b);
            if (d.CpuOverlapGflops is { } c) w.WriteNumber("cpu_overlap_gflops", c);
            if (d.CopyOverlapGBs is { } e) w.WriteNumber("copy_overlap_gbs", e);
            w.WriteEndObject();
        }
        w.WriteEndArray();
        w.WriteEndObject();
    }

    /// <summary>Reads a profile; null when the file is missing, unreadable or from another schema (a mismatch is ignored, not trusted).</summary>
    public static CalibrationProfile? Load(string path)
    {
        try
        {
            if (!File.Exists(path)) return null;
            using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
            var r = doc.RootElement;
            if (r.GetProperty("schema").GetInt32() != SchemaVersion) return null;
            double? Opt(JsonElement e, string n) => e.TryGetProperty(n, out var v) ? v.GetDouble() : null;
            var list = new List<DtypeCalibration>();
            foreach (var d in r.GetProperty("dtypes").EnumerateArray())
                list.Add(new DtypeCalibration(d.GetProperty("dtype").GetString()!, d.GetProperty("rows").GetInt32(), d.GetProperty("cols").GetInt32(),
                    d.GetProperty("prefill_tokens").GetInt32(), d.GetProperty("cpu_decode_gbs").GetDouble(),
                    d.GetProperty("cpu_prefill_gflops").GetDouble(), d.GetProperty("cpu_prefill_ms_per_expert").GetDouble(),
                    Opt(d, "gpu_copy_gbs"), Opt(d, "gpu_matmul_gflops"), Opt(d, "cpu_overlap_gflops"), Opt(d, "copy_overlap_gbs")));
            return new CalibrationProfile(r.GetProperty("schema").GetInt32(), r.GetProperty("created_utc").GetString()!,
                r.GetProperty("device_key").GetString()!, r.GetProperty("cpu").GetString()!,
                r.TryGetProperty("gpu", out var g) ? g.GetString() : null,
                r.TryGetProperty("gpu_integrated", out var gi) ? gi.GetBoolean() : null, list);
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException or IOException or FormatException)
        {
            return null;
        }
    }
}
