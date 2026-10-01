namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// docs/1-correctness/15 Phases 1 and 3: the AnyRes geometry of <see cref="LlavaImagePreprocessor"/> against llama.cpp's
/// <c>mtmd_image_preprocessor_llava_uhd</c> (hand-computed from its formulas), and the view ORDER mtmd feeds an MLP-projector
/// image: slices row-major, overview last. The order was never checked before 2026-10-01; against llama-server on a real
/// LLaVA-1.6 checkpoint, overview-first gave a mean first-token log-prob gap of 0.22 versus 0.07 for overview-last.
/// </summary>
public sealed class LlavaImagePreprocessorGeometryTests
{
    // LLaVA-1.6-mistral-7b's mmproj: image_size 336, image_grid_pinpoints (336,672) (672,336) (672,672) (1008,336) (336,1008).
    private static readonly (int W, int H)[] Pinpoints = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)];
    private static readonly float[] Zero = [0f, 0f, 0f];
    private static readonly float[] One = [1f, 1f, 1f];
    private const int Side = 336;

    // ── Phase 1.1: best-resolution selection (most effective pixels, then least waste) ───────────────────────────────

    [Theory]
    [InlineData(800, 600, 672, 672)]    // scale .84 -> 672x504: effective 338,688 beats every other candidate
    [InlineData(672, 336, 672, 336)]    // exact fit, zero waste
    [InlineData(336, 672, 336, 672)]
    [InlineData(300, 900, 336, 1008)]   // tall: 336x1008 wastes 68,688 px, the 672x672 option only reaches 150,528 effective
    [InlineData(1000, 300, 1008, 336)]  // wide: 1008x302 -> effective 300,000
    [InlineData(500, 500, 672, 672)]    // roughly square -> the big square grid
    public void SelectBestResolution_MatchesLlamaUhd(int w, int h, int expectedW, int expectedH)
    {
        var (bw, bh) = LlavaImagePreprocessor.SelectBestResolution(w, h, Pinpoints);
        Assert.Equal((expectedW, expectedH), (bw, bh));
    }

    // ── Phase 1.2: view counts and sizes ─────────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(336, 336, 1)]     // no slicing: the overview alone (the already-proven LLaVA-1.5 behaviour)
    [InlineData(200, 100, 1)]     // smaller than a tile: overview only
    [InlineData(672, 336, 3)]     // overview + 2x1
    [InlineData(336, 672, 3)]     // overview + 1x2
    [InlineData(800, 600, 5)]     // overview + 2x2
    [InlineData(300, 900, 4)]     // overview + 1x3
    [InlineData(1000, 300, 4)]    // overview + 3x1
    public void ViewCount_IsOverviewPlusTheGridTiles(int w, int h, int expectedViews)
    {
        var views = LlavaImagePreprocessor.PreprocessViews(new byte[w * h * 3], w, h, Side, Pinpoints, Zero, One);
        Assert.Equal(expectedViews, views.Count);
        Assert.All(views, v => Assert.Equal(3 * Side * Side, v.Length));   // every view is a full 336x336 CHW image
    }

    // ── Phase 1.3 / Phase 3: tile ordering with an unmistakable image ────────────────────────────────────────────────

    /// <summary>672x672: red top-left, green top-right, blue bottom-left, gold bottom-right, each exactly one 336x336 tile.</summary>
    private static byte[] FourQuadrants(int size = 672)
    {
        var img = new byte[size * size * 3];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                (byte r, byte g, byte b) = (x < size / 2, y < size / 2) switch
                {
                    (true, true) => ((byte)255, (byte)0, (byte)0),
                    (false, true) => ((byte)0, (byte)255, (byte)0),
                    (true, false) => ((byte)0, (byte)0, (byte)255),
                    _ => ((byte)255, (byte)215, (byte)0),
                };
                int i = (y * size + x) * 3;
                img[i] = r; img[i + 1] = g; img[i + 2] = b;
            }
        return img;
    }

    /// <summary>Mean of each colour plane (R, G, B) of a CHW view, with mean=0/std=1 so values are x/255.</summary>
    private static (double R, double G, double B) PlaneMeans(float[] chw)
    {
        int plane = Side * Side;
        double[] m = new double[3];
        for (int c = 0; c < 3; c++)
        {
            double s = 0; for (int i = 0; i < plane; i++) s += chw[c * plane + i];
            m[c] = s / plane;
        }
        return (m[0], m[1], m[2]);
    }

    [Fact]
    public void PreprocessViews_ReturnsOverviewFirstThenTilesRowMajor()
    {
        var views = LlavaImagePreprocessor.PreprocessViews(FourQuadrants(), 672, 672, Side, Pinpoints, Zero, One);
        Assert.Equal(5, views.Count);

        // overview = the whole image shrunk: every quadrant colour averaged (R: (255+0+0+255)/4/255 = 0.5, G: (0+255+0+215)/4/255 = 0.46)
        var ov = PlaneMeans(views[0]);
        Assert.InRange(ov.R, 0.45, 0.55); Assert.InRange(ov.G, 0.40, 0.52); Assert.InRange(ov.B, 0.20, 0.30);

        var (r, g, b, y) = (PlaneMeans(views[1]), PlaneMeans(views[2]), PlaneMeans(views[3]), PlaneMeans(views[4]));
        Assert.True(r.R > 0.95 && r.G < 0.05 && r.B < 0.05, "tile (row 0, col 0) must be the red quadrant");
        Assert.True(g.G > 0.95 && g.R < 0.05 && g.B < 0.05, "tile (row 0, col 1) must be the green quadrant");
        Assert.True(b.B > 0.95 && b.R < 0.05 && b.G < 0.05, "tile (row 1, col 0) must be the blue quadrant");
        Assert.True(y.R > 0.95 && y.G > 0.80 && y.B < 0.05, "tile (row 1, col 1) must be the gold quadrant");
    }

    [Fact]
    public void OverviewLast_PutsSlicesFirstInRowMajorOrderAndTheOverviewLast()
    {
        var overviewFirst = LlavaImagePreprocessor.PreprocessViews(FourQuadrants(), 672, 672, Side, Pinpoints, Zero, One);
        var ordered = LlavaImagePreprocessor.OverviewLast(overviewFirst);

        Assert.Equal(5, ordered.Count);
        Assert.Same(overviewFirst[1], ordered[0]);   // red   (row 0, col 0)
        Assert.Same(overviewFirst[2], ordered[1]);   // green (row 0, col 1)
        Assert.Same(overviewFirst[3], ordered[2]);   // blue  (row 1, col 0)
        Assert.Same(overviewFirst[4], ordered[3]);   // gold  (row 1, col 1)
        Assert.Same(overviewFirst[0], ordered[4]);   // overview last, as mtmd (ov_img_first = false) feeds it
    }

    [Fact]
    public void OverviewLast_LeavesASingleViewAlone()
    {
        var one = new List<float[]> { new float[3 * Side * Side] };
        Assert.Same(one, LlavaImagePreprocessor.OverviewLast(one));
    }
}
