namespace OpenTail.Stingray.Tests.Vulkan;

public sealed class ZzMatVecDtypeBenchTmp
{
    [Fact]
    public void Bench()
    {
        using var backend = new global::OpenTail.Stingray.Vulkan.VulkanBackend();
        foreach (var (rows, cols) in new[] { (2048, 2048), (8192, 2048), (2048, 8192), (4096, 4096) })
        {
        const int iters = 50;
        var cases = new (DType dt, double bytesPerElem, int blockBytes, int dOff)[]
        {
            (DType.Q4_K, 144/256.0, 144, 0), (DType.Q6_K, 210/256.0, 210, 208),
        };
        var rng = new Random(1);
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var gIn = backend.Upload(input, TensorShape.D1(cols));
        var gOut = backend.Allocate(TensorShape.D1(rows));
        foreach (var (dt, bpe, blockBytes, dOff) in cases)
        {
            long bytes = (long)(rows * (long)cols * bpe);
            var raw = new byte[(bytes + 3) & ~3L];
            rng.NextBytes(raw);
            if (dt == DType.Float32) { var f = new float[rows * cols]; for (int i = 0; i < f.Length; i++) f[i] = (float)rng.NextDouble(); Buffer.BlockCopy(f, 0, raw, 0, raw.Length); }
            else if (dt == DType.Float16) { for (int i = 0; i < raw.Length; i += 2) BitConverter.TryWriteBytes(raw.AsSpan(i), (Half)(float)rng.NextDouble()); }
            else for (int off = 0; off + blockBytes <= raw.Length; off += blockBytes) BitConverter.TryWriteBytes(raw.AsSpan(off + dOff), (Half)0.01f);
            var f32 = new float[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
            var gW = backend.Upload(f32, TensorShape.D1(f32.Length));
            backend.MatMul(gOut, gW, gIn, dt); backend.Synchronize();
            double best = double.MaxValue;
            for (int rep = 0; rep < 5; rep++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                backend.BeginRecord();
                for (int i = 0; i < iters; i++) backend.MatMul(gOut, gW, gIn, dt);
                backend.EndRecordAndSubmit();
                backend.Synchronize();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
            }
            Console.WriteLine($"{rows}x{cols} {dt,-8} {bytes / 1048576.0,7:F1} MiB  {best * 1000,8:F0} us/matvec  {bytes / (best / 1000) / 1e9,6:F1} GB/s");
            backend.Free(gW);
        }
        backend.Free(gIn); backend.Free(gOut);
        }
    }
}
