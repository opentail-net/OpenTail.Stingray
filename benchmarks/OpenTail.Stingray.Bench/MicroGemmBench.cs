
namespace OpenTail.Stingray.Bench;

[MemoryDiagnoser]
[WarmupCount(3)]
[IterationCount(5)]
public unsafe class MicroGemmBench
{
    [Params(4, 16, 39, 64)]
    public int M;

    private const int K = 2048;
    private const int N = 2048;

    private float* _a;
    private float* _b;
    private float* _cMicro;
    private float* _cScalar;

    [GlobalSetup]
    public void Setup()
    {
        _a = (float*)NativeMemory.AllocZeroed((nuint)(M * K * sizeof(float)));
        _b = (float*)NativeMemory.AllocZeroed((nuint)(N * K * sizeof(float)));
        _cMicro  = (float*)NativeMemory.AllocZeroed((nuint)(M * N * sizeof(float)));
        _cScalar = (float*)NativeMemory.AllocZeroed((nuint)(M * N * sizeof(float)));

        for (int i = 0; i < M * K; i++) _a[i] = (i + 1) * 0.001f;
        for (int i = 0; i < N * K; i++) _b[i] = (i + 1) * 0.002f;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.Free(_a);
        NativeMemory.Free(_b);
        NativeMemory.Free(_cMicro);
        NativeMemory.Free(_cScalar);
    }

    [Benchmark(Baseline = true)]
    public void ScalarMatMul()
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++)
                {
                    acc += _a[i * K + k] * _b[j * K + k];
                }
                _cScalar[i * N + j] = acc;
            }
        }
    }

    [Benchmark]
    public void FusedMicroGemm()
    {
        MicroGemmConfig.IsEnabled = true;
        MicroGemmKernel.TryMatMulF32(_a, _b, _cMicro, M, K, N);
        MicroGemmConfig.IsEnabled = false;
    }
}
