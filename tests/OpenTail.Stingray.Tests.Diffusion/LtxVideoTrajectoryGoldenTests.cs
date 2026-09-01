using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// End-to-end multi-step denoising trajectory check: runs the SAME scheduler-shift + CFG + Euler
/// step loop <see cref="LtxVideoPipeline.GenerateVideo"/> uses (reimplemented here directly against
/// <see cref="LtxVideoModel"/>, since the pipeline method itself builds its own random latents/
/// tokenizes its own prompt rather than accepting fixed ones) against a real 4-step trajectory
/// dumped from the official `ltx_video` package + real `RectifiedFlowScheduler`, real weights.
///
/// Every individual piece (transformer, VAE at both F=1 and F=2, T5 encoder+tokenizer, and the
/// scheduler shift formula in isolation) already golden-tests at or near machine precision -- this
/// test exists because NONE of those prior tests actually exercised the multi-step LOOP itself
/// (accumulating dt*velocity across real steps with real CFG), which is the one piece of the real
/// pipeline that was only ever smoke-tested before. Written 2026-09-01 while investigating a real
/// generation (256x256, 30 steps, real T5 prompt) producing pure noise despite every other stage
/// checking out.
/// </summary>
public sealed class LtxVideoTrajectoryGoldenTests
{
    private const string ModelFileName = "ltx-video-2b-v0.9.1.safetensors";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindGoldenDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "LtxTrajectoryGolden");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadBin(string dir, string name)
    {
        var bytes = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12));
    }

    private static float GetNormalShift(int nTokens, int minTokens = 1024, int maxTokens = 4096,
        float minShift = 0.95f, float maxShift = 2.05f)
    {
        float m = (maxShift - minShift) / (maxTokens - minTokens);
        float b = minShift - m * minTokens;
        return m * nTokens + b;
    }

    private static float TimeShift(float mu, float sigma, float t)
        => MathF.Exp(mu) / (MathF.Exp(mu) + MathF.Pow(1.0f / t - 1.0f, sigma));

    [Fact]
    public void FourStepTrajectory_MatchesRealSchedulerAndTransformer()
    {
        string? modelPath = FindModelPath(ModelFileName);
        string? goldenDir = FindGoldenDir();
        if (modelPath is null || goldenDir is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var model = new LtxVideoModel(loader);

        var latents = ReadBin(goldenDir, "latents0"); // [4,128] token-major
        var condCtx = ReadBin(goldenDir, "cond_ctx"); // [8,4096]
        var uncondCtx = ReadBin(goldenDir, "uncond_ctx");
        var goldenTimesteps = ReadBin(goldenDir, "timesteps");
        var goldenLat1 = ReadBin(goldenDir, "lat1");
        var goldenLat2 = ReadBin(goldenDir, "lat2");
        var goldenLat3 = ReadBin(goldenDir, "lat3");
        var goldenLat4 = ReadBin(goldenDir, "lat4");

        int numFrames = 1, patchH = 2, patchW = 2;
        int numTokens = numFrames * patchH * patchW;
        int steps = 4;
        float guidance = 4.5f;

        int numLatentTokens = numTokens;
        float shift = GetNormalShift(numLatentTokens);
        var shiftedTimesteps = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            float tRaw = 1.0f - (float)i / steps;
            shiftedTimesteps[i] = TimeShift(shift, 1.0f, tRaw);
        }
        shiftedTimesteps[steps] = 0f;

        // Real scheduler timesteps for this exact scenario -- verify our own formula reproduces
        // them before trusting the rest of the loop below.
        for (int i = 0; i < steps; i++)
        {
            Assert.True(MathF.Abs(shiftedTimesteps[i] - goldenTimesteps[i]) < 1e-4f,
                $"shifted timestep[{i}] mismatch: ours={shiftedTimesteps[i]} real={goldenTimesteps[i]}");
        }

        var expectedTrajectory = new[] { goldenLat1, goldenLat2, goldenLat3, goldenLat4 };

        for (int step = 0; step < steps; step++)
        {
            float tShifted = shiftedTimesteps[step];
            float dt = tShifted - shiftedTimesteps[step + 1];
            float timestep = tShifted * model.TimestepScale;

            var vPredCond = model.Forward(latents, timestep, condCtx, numFrames, patchH, patchW);
            var vPredUncond = model.Forward(latents, timestep, uncondCtx, numFrames, patchH, patchW);
            var vPred = new float[vPredCond.Length];
            for (int i = 0; i < vPred.Length; i++)
                vPred[i] = vPredUncond[i] + guidance * (vPredCond[i] - vPredUncond[i]);

            for (int i = 0; i < latents.Length; i++)
                latents[i] -= dt * vPred[i];

            float cos = CosineSimilarity(latents, expectedTrajectory[step]);
            Assert.True(cos > 0.999f, $"step {step} latents cosine-sim too low: {cos}");
        }
    }
}
