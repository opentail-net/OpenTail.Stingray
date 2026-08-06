using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public class TypeConverterSmokeTestFixture
{
    [Fact]
    public void EndToEnd_TensorLoadingAndKvCachePipeline_SmokeTestSucceeds()
    {
        const int numElements = 131_072; // 128K elements (simulating a multi-layer activation / KV-cache tensor)
        const int q8Blocks = numElements / 32;

        Random rand = new Random(1337);

        // 1. Generate realistic Float32 reference activations (Gaussian-ish values around zero)
        float[] origF32 = new float[numElements];
        for (int i = 0; i < numElements; i++)
        {
            origF32[i] = (float)(rand.NextDouble() * 10.0 - 5.0);
        }

        // ── Step A: Convert Float32 -> BF16 weight stream -> Float32 ───────
        byte[] bf16Stream = new byte[numElements * 2];
        FastVectorTypeConverter.ConvertF32ToBf16(origF32, bf16Stream);

        float[] decodedBf16 = new float[numElements];
        FastVectorTypeConverter.ConvertBf16ToF32(bf16Stream, decodedBf16);

        // Assert BF16 fidelity (|f32 - bf16| <= |f32| * 0.01)
        for (int i = 0; i < numElements; i++)
        {
            float diff = MathF.Abs(origF32[i] - decodedBf16[i]);
            float allowed = MathF.Max(0.05f, MathF.Abs(origF32[i]) * 0.01f);
            Assert.True(diff <= allowed, $"BF16 pipeline error at {i}: orig={origF32[i]}, decoded={decodedBf16[i]}");
        }

        // ── Step B: Convert Float32 -> FP16 weight stream -> Float32 ───────
        byte[] f16Stream = new byte[numElements * 2];
        FastVectorTypeConverter.ConvertF32ToF16(origF32, f16Stream);

        float[] decodedF16 = new float[numElements];
        FastVectorTypeConverter.ConvertF16ToF32(f16Stream, decodedF16);

        for (int i = 0; i < numElements; i++)
        {
            float diff = MathF.Abs(origF32[i] - decodedF16[i]);
            float allowed = MathF.Max(0.005f, MathF.Abs(origF32[i]) * 0.001f);
            Assert.True(diff <= allowed, $"FP16 pipeline error at {i}: orig={origF32[i]}, decoded={decodedF16[i]}");
        }

        // ── Step C: Convert Float32 -> Q8_0 KV-cache -> Float32 ────────────
        byte[] q8CacheBuffer = new byte[q8Blocks * 34];
        FastVectorTypeConverter.ConvertF32ToQ8_0(origF32, q8CacheBuffer);

        float[] dequantizedKvCache = new float[numElements];
        FastVectorTypeConverter.ConvertQ8_0ToF32(q8CacheBuffer, dequantizedKvCache);

        // Assert Q8_0 KV cache quantization error bound
        for (int b = 0; b < q8Blocks; b++)
        {
            float maxAbs = 0f;
            for (int j = 0; j < 32; j++)
            {
                float a = MathF.Abs(origF32[b * 32 + j]);
                if (a > maxAbs) maxAbs = a;
            }
            float maxAllowedDiff = (maxAbs / 127.0f) * 1.5f + 0.001f;

            for (int j = 0; j < 32; j++)
            {
                int idx = b * 32 + j;
                float diff = MathF.Abs(origF32[idx] - dequantizedKvCache[idx]);
                Assert.True(diff <= maxAllowedDiff, $"Q8_0 KV cache error at {idx}: orig={origF32[idx]}, dequant={dequantizedKvCache[idx]}");
            }
        }

        // ── Step D: Convert FP8 E4M3 & E5M2 Stream -> Float32 ──────────────
        byte[] fp8E4Stream = new byte[numElements];
        byte[] fp8E5Stream = new byte[numElements];
        for (int i = 0; i < numElements; i++)
        {
            fp8E4Stream[i] = (byte)(i % 256);
            fp8E5Stream[i] = (byte)((i * 3) % 256);
        }

        float[] fp8E4Out = new float[numElements];
        float[] fp8E5Out = new float[numElements];

        FastVectorTypeConverter.ConvertFp8E4M3ToF32(fp8E4Stream, fp8E4Out);
        FastVectorTypeConverter.ConvertFp8E5M2ToF32(fp8E5Stream, fp8E5Out);

        // Assert non-trivial outputs
        Assert.NotEqual(0.0f, fp8E4Out[1]);
        Assert.NotEqual(0.0f, fp8E5Out[1]);
    }
}
