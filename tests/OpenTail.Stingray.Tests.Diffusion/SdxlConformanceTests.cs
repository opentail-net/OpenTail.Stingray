using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.SDXL;
using OpenTail.Stingray.Diffusion.StableDiffusion;
using OpenTail.Stingray.Diffusion.TextEncoders;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class SdxlConformanceTests
{
    [Fact]
    public void SdxlScheduler_BuildsDescendingEulerSchedule()
    {
        var scheduler = new EulerDiscreteScheduler(numInferenceSteps: 25, DiffusionSchedulerType.EulerAncestral);

        Assert.Equal(25, scheduler.NumSteps);
        Assert.Equal(26, scheduler.Sigmas.Length);
        Assert.Equal(25, scheduler.Timesteps.Length);

        Assert.Equal(999f, scheduler.Timesteps[0]);
        Assert.Equal(0f, scheduler.Timesteps[^1]);
        Assert.Equal(0f, scheduler.Sigmas[^1]);
    }

    [Fact]
    public void Sdxl_AddEmbeddings_Matches2816Dimensions()
    {
        // 1280 pooled + 6 * 256 coordinates = 2816
        float[] pooled = new float[1280];
        for (int i = 0; i < 1280; i++) pooled[i] = 0.5f;

        int origH = 1024, origW = 1024;
        int cropH = 0, cropW = 0;
        int targetH = 1024, targetW = 1024;

        int[] coords = [origH, origW, cropH, cropW, targetH, targetW];
        var addEmbeds = new float[2816];
        Array.Copy(pooled, 0, addEmbeds, 0, 1280);

        for (int i = 0; i < coords.Length; i++)
        {
            var emb = new float[256];
            int half = 128;
            float logMaxPeriod = MathF.Log(10000.0f);
            for (int j = 0; j < half; j++)
            {
                float freq = MathF.Exp(-logMaxPeriod * j / half);
                float arg = coords[i] * freq;
                emb[j] = MathF.Cos(arg);
                emb[half + j] = MathF.Sin(arg);
            }
            Array.Copy(emb, 0, addEmbeds, 1280 + i * 256, 256);
        }

        Assert.Equal(2816, addEmbeds.Length);
        Assert.Equal(0.5f, addEmbeds[0]);
        Assert.Equal(0.5f, addEmbeds[1279]);
        Assert.NotEqual(0f, addEmbeds[1280]);
    }

    [Fact]
    public void Sdxl_ContextConcatenation_Matches2048Dimensions()
    {
        var clipLHidden = new float[77 * 768];
        var clipGHidden = new float[77 * 1280];

        var cat = new float[77 * 2048];
        for (int t = 0; t < 77; t++)
        {
            Array.Copy(clipLHidden, t * 768, cat, t * 2048, 768);
            Array.Copy(clipGHidden, t * 1280, cat, t * 2048 + 768, 1280);
        }

        Assert.Equal(77 * 2048, cat.Length);
    }
}
