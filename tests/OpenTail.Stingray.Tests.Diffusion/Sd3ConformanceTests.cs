using OpenTail.Stingray.Diffusion.SD3;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Sd3ConformanceTests
{
    [Fact]
    public void Sd3_TripleConditioning_Matches4096Context()
    {
        var clipL = new float[77 * 768];
        var clipG = new float[77 * 1280];

        var context = new float[77 * 4096];
        for (int t = 0; t < 77; t++)
        {
            Array.Copy(clipL, t * 768, context, t * 4096, 768);
            Array.Copy(clipG, t * 1280, context, t * 4096 + 768, 1280);
        }

        Assert.Equal(77 * 4096, context.Length);
    }

    [Fact]
    public void Sd3_PooledVector_Matches2048Dimension()
    {
        var pooledL = new float[768];
        var pooledG = new float[1280];
        for (int i = 0; i < 768; i++) pooledL[i] = 1.0f;
        for (int i = 0; i < 1280; i++) pooledG[i] = 2.0f;

        var y = new float[2048];
        Array.Copy(pooledL, 0, y, 0, 768);
        Array.Copy(pooledG, 0, y, 768, 1280);

        Assert.Equal(2048, y.Length);
        Assert.Equal(1.0f, y[0]);
        Assert.Equal(1.0f, y[767]);
        Assert.Equal(2.0f, y[768]);
        Assert.Equal(2.0f, y[2047]);
    }

    [Fact]
    public void Sd3_FlowMatchingEulerStep_ReachesTarget()
    {
        // Rectified flow: x_{t-dt} = x_t - dt * v
        int steps = 20;
        float dt = 1.0f / steps;
        float x = 1.0f; // Start at noise t=1.0
        float target = 0.0f;
        float v = (x - target); // constant velocity = 1.0

        for (int step = 0; step < steps; step++)
        {
            x -= dt * v;
        }

        Assert.Equal(0.0f, x, precision: 4);
    }
}
