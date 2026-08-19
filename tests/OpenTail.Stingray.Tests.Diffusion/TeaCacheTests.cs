using System;
using OpenTail.Stingray.Diffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class TeaCacheTests
{
    [Fact]
    public void WarmupSteps_AlwaysTriggerFullCompute()
    {
        var config = new TeaCacheConfig
        {
            Mode = TeaCacheMode.FixedInterval,
            WarmupSteps = 3,
            CacheInterval = 5
        };
        var hook = new TeaCacheHook(config);

        Assert.True(hook.ShouldCompute(0));
        hook.Update(0, [1.0f, 2.0f]);

        Assert.True(hook.ShouldCompute(1));
        hook.Update(1, [1.5f, 2.5f]);

        Assert.True(hook.ShouldCompute(2));
        hook.Update(2, [2.0f, 3.0f]);

        // Step 3 is past warmup (3) and interval is 5, so should NOT compute
        Assert.False(hook.ShouldCompute(3));
    }

    [Fact]
    public void FixedIntervalMode_TriggersComputeAtConfiguredInterval()
    {
        var config = new TeaCacheConfig
        {
            Mode = TeaCacheMode.FixedInterval,
            WarmupSteps = 2,
            CacheInterval = 3
        };
        var hook = new TeaCacheHook(config);

        // Step 0 (warmup 0): Compute
        Assert.True(hook.ShouldCompute(0));
        hook.Update(0, [10.0f, 20.0f]);

        // Step 1 (warmup 1): Compute
        Assert.True(hook.ShouldCompute(1));
        hook.Update(1, [12.0f, 22.0f]);

        // Step 2 (offset 1): Skip
        Assert.False(hook.ShouldCompute(2));

        // Step 3 (offset 2): Skip
        Assert.False(hook.ShouldCompute(3));

        // Step 4 (offset 3 == CacheInterval): Compute
        Assert.True(hook.ShouldCompute(4));
        hook.Update(4, [18.0f, 28.0f]);

        // Step 5 (offset 1): Skip
        Assert.False(hook.ShouldCompute(5));
    }

    [Fact]
    public void TaylorExpansion_Order1_AccuratelyExtrapolatesLinearTrajectory()
    {
        var state = new TeaCacheState(maxOrder: 1);

        // Linear velocity: y = 2*t + 10
        // t = 0: y = 10
        state.Update(0, [10.0f, 100.0f]);

        // t = 1: y = 12 (delta = +2, +10)
        state.Update(1, [12.0f, 110.0f]);

        // Predict at t = 2: 12 + 2*1 = 14
        var pred2 = state.Predict(2);
        Assert.Equal(14.0f, pred2[0], tolerance: 1e-4f);
        Assert.Equal(120.0f, pred2[1], tolerance: 1e-4f);

        // Predict at t = 3: 12 + 2*2 = 16
        var pred3 = state.Predict(3);
        Assert.Equal(16.0f, pred3[0], tolerance: 1e-4f);
        Assert.Equal(130.0f, pred3[1], tolerance: 1e-4f);
    }

    [Fact]
    public void TaylorExpansion_Order2_MatchesTaylorSeerPolynomialApproximation()
    {
        var state = new TeaCacheState(maxOrder: 2);

        // Quadratic: y(t) = t^2 + 2t + 5
        // t = 0: 5
        state.Update(0, [5.0f]);

        // t = 1: 1 + 2 + 5 = 8
        state.Update(1, [8.0f]);

        // t = 2: 4 + 4 + 5 = 13
        state.Update(2, [13.0f]);

        // TaylorSeer divided difference prediction at t = 3 from t = 2 (stepOffset = 1):
        // factor0 = 13, factor1 = 5, factor2 = 2
        // y_pred = 13 + 5*1 + 2*(1/2) = 19
        var pred3 = state.Predict(3);
        Assert.Equal(19.0f, pred3[0], tolerance: 1e-3f);
    }

    [Fact]
    public void RelativeVelocityThreshold_SkipsWhenDeltaIsBelowThreshold()
    {
        var config = new TeaCacheConfig
        {
            Mode = TeaCacheMode.RelativeVelocityThreshold,
            WarmupSteps = 1,
            Threshold = 0.20f // 20%
        };
        var hook = new TeaCacheHook(config);

        // Step 0: Warmup
        Assert.True(hook.ShouldCompute(0, indicator: [1.0f, 1.0f]));
        hook.Update(0, [100.0f], indicator: [1.0f, 1.0f]);

        // Step 1: Small indicator change (5%)
        Assert.False(hook.ShouldCompute(1, indicator: [1.05f, 1.05f]));

        // Step 2: Another small indicator change (cumulative ~10%)
        Assert.False(hook.ShouldCompute(2, indicator: [1.10f, 1.10f]));

        // Step 3: Larger change exceeding 20% threshold
        Assert.True(hook.ShouldCompute(3, indicator: [1.35f, 1.35f]));
        hook.Update(3, [135.0f], indicator: [1.35f, 1.35f]);

        // Reset threshold accumulation after update
        Assert.False(hook.ShouldCompute(4, indicator: [1.36f, 1.36f]));
    }

    [Fact]
    public void Reset_ClearsAllCachedStates()
    {
        var hook = new TeaCacheHook(new TeaCacheConfig { WarmupSteps = 1 });
        hook.Update(0, [1.0f, 2.0f]);

        Assert.True(hook.GetState().HasFactors);

        hook.Reset();

        Assert.False(hook.GetState().HasFactors);
        Assert.Null(hook.GetState().LastUpdateStep);
    }

    [Fact]
    public void TeaCacheHook_MaintainsIsolatedModuleStates()
    {
        var hook = new TeaCacheHook();

        hook.Update(0, [1.0f, 2.0f], moduleName: "block.0");
        hook.Update(0, [10.0f, 20.0f], moduleName: "block.1");

        var pred0 = hook.Predict(0, moduleName: "block.0");
        var pred1 = hook.Predict(0, moduleName: "block.1");

        Assert.Equal(1.0f, pred0[0]);
        Assert.Equal(10.0f, pred1[0]);
    }
}
