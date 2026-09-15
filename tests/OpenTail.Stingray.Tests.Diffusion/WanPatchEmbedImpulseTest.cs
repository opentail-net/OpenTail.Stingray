using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Stage 1 of the Wan2.1 tensor-layout diagnostic plan (docs/081):
/// Spatial impulse tests on PackLatents / patch embedding.
///
/// Builds synthetic latent tensors with a single non-zero impulse at known coordinates
/// (c, t, y, x) and tracks exactly which token index and slot become non-zero.
/// Confirms that moving the impulse across x, y, t, and channel moves the affected
/// token index and slot with the exact expected strides and axis orderings.
/// </summary>
public sealed class WanPatchEmbedImpulseTest
{
    private static int LatentIndex(int c, int t, int y, int x, int numFrames, int latH, int latW)
    {
        return ((c * numFrames + t) * latH + y) * latW + x;
    }

    [Fact]
    public void SpatialImpulse_MovesTokensAlongAxesWithExactStrides()
    {
        const int numFrames = 2;
        const int latH = 8;
        const int latW = 8;
        const int latC = 16;
        const int inChannels = 64;

        int patchH = latH / 2; // 4
        int patchW = latW / 2; // 4
        int numTokens = numFrames * patchH * patchW; // 32

        int totalLatent = latC * numFrames * latH * latW;

        // Helper to run PackLatents on a single impulse and locate the non-zero token and slot
        (int tokenIdx, int slot) FindImpulse(int c, int t, int y, int x)
        {
            var latent = new float[totalLatent];
            latent[LatentIndex(c, t, y, x, numFrames, latH, latW)] = 1.0f;

            var packed = WanModel.PackLatents(latent, numFrames, latH, latW);

            int nonZeroCount = 0;
            int foundToken = -1;
            int foundSlot = -1;

            for (int tok = 0; tok < numTokens; tok++)
            {
                for (int s = 0; s < inChannels; s++)
                {
                    if (packed[tok * inChannels + s] != 0f)
                    {
                        nonZeroCount++;
                        foundToken = tok;
                        foundSlot = s;
                    }
                }
            }

            Assert.Equal(1, nonZeroCount);
            return (foundToken, foundSlot);
        }

        // Test 1: Origin (c=0, t=0, y=0, x=0) -> token 0, slot 0
        {
            var (token, slot) = FindImpulse(0, 0, 0, 0);
            Console.WriteLine($"[Impulse (0,0,0,0)] -> token={token}, slot={slot}");
            Assert.Equal(0, token);
            Assert.Equal(0, slot);
        }

        // Test 2: Step 1 in x (within same patch) -> (c=0, t=0, y=0, x=1) -> same token 0, dx=1
        {
            var (token, slot) = FindImpulse(0, 0, 0, 1);
            Console.WriteLine($"[Impulse (0,0,0,1)] -> token={token}, slot={slot}");
            Assert.Equal(0, token);
            // In WanModel.PackLatents: slot = c*4 + dy*2 + dx = 0*4 + 0*2 + 1 = 1
            Assert.Equal(1, slot);
        }

        // Test 3: Step 2 in x (cross into next patch) -> (c=0, t=0, y=0, x=2) -> token 1, dx=0
        {
            var (token, slot) = FindImpulse(0, 0, 0, 2);
            Console.WriteLine($"[Impulse (0,0,0,2)] -> token={token}, slot={slot}");
            Assert.Equal(1, token);
            Assert.Equal(0, slot);
        }

        // Test 4: Step 1 in y (within same patch) -> (c=0, t=0, y=1, x=0) -> same token 0, dy=1
        {
            var (token, slot) = FindImpulse(0, 0, 1, 0);
            Console.WriteLine($"[Impulse (0,0,1,0)] -> token={token}, slot={slot}");
            Assert.Equal(0, token);
            // slot = c*4 + dy*2 + dx = 0*4 + 1*2 + 0 = 2
            Assert.Equal(2, slot);
        }

        // Test 5: Step 2 in y (cross into next patch row) -> (c=0, t=0, y=2, x=0) -> token patchW=4
        {
            var (token, slot) = FindImpulse(0, 0, 2, 0);
            Console.WriteLine($"[Impulse (0,0,2,0)] -> token={token}, slot={slot}");
            Assert.Equal(patchW, token); // token 4
            Assert.Equal(0, slot);
        }

        // Test 6: Step 1 in t (cross into next frame) -> (c=0, t=1, y=0, x=0) -> token patchH*patchW=16
        {
            var (token, slot) = FindImpulse(0, 1, 0, 0);
            Console.WriteLine($"[Impulse (0,1,0,0)] -> token={token}, slot={slot}");
            Assert.Equal(patchH * patchW, token); // token 16
            Assert.Equal(0, slot);
        }

        // Test 7: Step 1 in channel -> (c=1, t=0, y=0, x=0) -> same token 0, slot = 1*4 = 4
        {
            var (token, slot) = FindImpulse(1, 0, 0, 0);
            Console.WriteLine($"[Impulse (1,0,0,0)] -> token={token}, slot={slot}");
            Assert.Equal(0, token);
            Assert.Equal(4, slot);
        }

        // Test 8: Arbitrary position (c=7, t=1, y=5, x=6)
        // ph = 5 / 2 = 2, dy = 1
        // pw = 6 / 2 = 3, dx = 0
        // token = (t * patchH + ph) * patchW + pw = (1 * 4 + 2) * 4 + 3 = 27
        // slot = c * 4 + dy * 2 + dx = 7 * 4 + 1 * 2 + 0 = 30
        {
            var (token, slot) = FindImpulse(7, 1, 5, 6);
            Console.WriteLine($"[Impulse (7,1,5,6)] -> token={token}, slot={slot}");
            Assert.Equal(27, token);
            Assert.Equal(30, slot);
        }
    }

    [Fact]
    public void ExhaustiveImpulseSweep_EveryPositionMapsToUniqueTokenAndSlot()
    {
        const int numFrames = 2;
        const int latH = 6;
        const int latW = 8;
        const int latC = 16;
        const int inChannels = 64;

        int patchH = latH / 2; // 3
        int patchW = latW / 2; // 4
        int numTokens = numFrames * patchH * patchW; // 24

        int totalLatent = latC * numFrames * latH * latW;
        var visited = new bool[numTokens, inChannels];

        for (int c = 0; c < latC; c++)
        {
            for (int t = 0; t < numFrames; t++)
            {
                for (int y = 0; y < latH; y++)
                {
                    for (int x = 0; x < latW; x++)
                    {
                        var latent = new float[totalLatent];
                        latent[LatentIndex(c, t, y, x, numFrames, latH, latW)] = 1.0f;

                        var packed = WanModel.PackLatents(latent, numFrames, latH, latW);

                        int expectedPh = y / 2;
                        int expectedDy = y % 2;
                        int expectedPw = x / 2;
                        int expectedDx = x % 2;
                        int expectedToken = (t * patchH + expectedPh) * patchW + expectedPw;
                        int expectedSlot = c * 4 + expectedDy * 2 + expectedDx;

                        // Verify exactly this element is set
                        Assert.Equal(1.0f, packed[expectedToken * inChannels + expectedSlot]);
                        Assert.False(visited[expectedToken, expectedSlot],
                            $"Collision at token {expectedToken}, slot {expectedSlot} from (c={c},t={t},y={y},x={x})");
                        visited[expectedToken, expectedSlot] = true;
                    }
                }
            }
        }

        // Verify all elements covered (bijection)
        for (int tok = 0; tok < numTokens; tok++)
        {
            for (int s = 0; s < inChannels; s++)
            {
                Assert.True(visited[tok, s], $"Token {tok}, slot {s} was never reached!");
            }
        }
        Console.WriteLine($"[ExhaustiveImpulse] Verified bijection for all {totalLatent} elements.");
    }
}
