namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic-only test isolating the Wan VAE decoder from the DiT/RoPE side of the pipeline, to
/// determine which half of the pipeline is responsible for the periodic tiling/checkerboard
/// artifact seen in real end-to-end runs after five other real bugs were fixed (RoPE pairing,
/// QK-norm scope, unpatchify channel ordering, VAE latent-scale formula, VAE upsample coordinate
/// math -- see docs/diffusion-samples/README.md "Round 3"). Feeds an all-zero latent (which,
/// after the decoder's own per-channel un-normalization, becomes a spatially-constant per-channel
/// DC field -- perfectly smooth, no patch/grid structure at all) directly into
/// <see cref="WanVaeDecoder3D.Decode"/>, bypassing <see cref="WanModel"/>/<see cref="WanRoPE"/>
/// entirely. If the decoded PNG still shows the periodic tiling artifact, the bug is confined to
/// the VAE decoder (ResampleSpatial phase/CausalConv slice selection). If it decodes smoothly, the
/// bug is upstream, in the DiT's patchify/RoPE/unpatchify path.
/// Not a correctness assertion (no golden reference for this synthetic case) -- writes an image
/// to docs/diffusion-samples/ for visual inspection, same pattern as other real-weight probes in
/// this project. Requires the real Wan VAE checkpoint locally; no-ops (returns) if absent.
/// </summary>
public sealed class WanVaeSyntheticLatentDiagnosticTests
{
    private static string? FindVaePath()
    {
        string[] candidates =
        {
            @"C:\Git-Public\OpenTail.Stingray\models\wan2.1\Wan2.1_VAE.safetensors",
        };
        foreach (var p in candidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "wan2.1", "Wan2.1_VAE.safetensors");
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Decode_AllZeroSyntheticLatent_IsolatesVaeFromDiT()
    {
        string? vaePath = FindVaePath();
        if (vaePath is null) return;

        const int latH = 32, latW = 32, t = 1, c = WanVaeDecoder3D.LatentChannels;

        // All-zero latent -> after the decoder's own `z = latent * std + mean` un-normalization,
        // every spatial position in every channel becomes exactly `mean` -- a perfectly flat, DC
        // field with zero spatial variation. Any periodic structure in the decoded output can only
        // have been introduced by the decoder itself (ResampleSpatial / CausalConv3D), since the
        // input carries no spatial signal for it to derive from.
        var zeroLatent = new float[c * t * latH * latW];

        using var loader = SafetensorsLoader.Open(vaePath);
        using var vae = new WanVaeDecoder3D(loader);

        List<float[]> frames = vae.Decode(zeroLatent, t, latH, latW);
        Assert.Single(frames);

        int width = latW * WanVaeDecoder3D.SpatialScale;
        int height = latH * WanVaeDecoder3D.SpatialScale;
        Assert.Equal(3 * width * height, frames[0].Length);

        string outDir = FindDocsDiffusionSamplesDir();
        string outPath = Path.Combine(outDir, "wan-vae-synthetic-dc-latent-diagnostic.png");
        PngWriter.Write(outPath, frames[0], width, height);

        // Measure whether the output is smooth (low high-frequency variance) or exhibits periodic
        // banding (large frame-to-frame alternating difference at a fixed stride) as an objective
        // companion to visual inspection -- computes the mean absolute difference between
        // horizontally-adjacent pixels at even vs. odd column parity, per channel. A strong parity
        // split (even-column mean far from odd-column mean) is the numeric signature of the
        // checkerboard artifact; a smooth decode should show near-zero split.
        double evenSum = 0, oddSum = 0;
        int evenCount = 0, oddCount = 0;
        var rgb = frames[0];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float v = rgb[(0 * height + y) * width + x]; // R channel
                if ((x & 1) == 0) { evenSum += v; evenCount++; }
                else { oddSum += v; oddCount++; }
            }
        }
        double evenMean = evenSum / evenCount;
        double oddMean = oddSum / oddCount;
        Console.WriteLine($"[WanVae synthetic-DC diagnostic] R-channel even-column mean={evenMean:F4}, odd-column mean={oddMean:F4}, |split|={Math.Abs(evenMean - oddMean):F4}");

        // Row-parity split (the axis the visible artifact is actually on) plus raw per-row
        // R-channel values at x=0 and x=width/2, to distinguish a genuine periodic indexing bug
        // (constant-amplitude alternation across the whole frame) from a boundary/padding decay
        // effect (large near row 0/height-1, decaying to ~0 toward the center).
        double evenRowSum = 0, oddRowSum = 0;
        int evenRowCount = 0, oddRowCount = 0;
        for (int y = 0; y < height; y++)
        {
            float v = rgb[y * width + 0];
            if ((y & 1) == 0) { evenRowSum += v; evenRowCount++; }
            else { oddRowSum += v; oddRowCount++; }
        }
        Console.WriteLine($"[WanVae synthetic-DC diagnostic] R-channel even-row mean={evenRowSum / evenRowCount:F4}, odd-row mean={oddRowSum / oddRowCount:F4}, |split|={Math.Abs(evenRowSum / evenRowCount - oddRowSum / oddRowCount):F4}");

        int xMid = width / 2;
        for (int y = 0; y < Math.Min(height, 24); y++)
        {
            float atX0 = rgb[y * width + 0];
            float atXMid = rgb[y * width + xMid];
            Console.WriteLine($"[WanVae row dump] row {y,3}: R@x0={atX0:F5}  R@xmid={atXMid:F5}");
        }
        for (int y = height / 2 - 4; y < height / 2 + 4; y++)
        {
            float atX0 = rgb[y * width + 0];
            float atXMid = rgb[y * width + xMid];
            Console.WriteLine($"[WanVae row dump, center] row {y,3}: R@x0={atX0:F5}  R@xmid={atXMid:F5}");
        }
        Console.WriteLine($"[WanVae synthetic-DC diagnostic] wrote {outPath}");
    }

    private static string FindDocsDiffusionSamplesDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "docs", "diffusion-samples");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return Directory.GetCurrentDirectory();
    }
}
