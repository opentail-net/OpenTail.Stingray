using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// M-RoPE positions for Qwen2-VL / Qwen2.5-VL / Qwen3-VL (mtmd <c>MTMD_POS_TYPE_MROPE</c>), shared by the CPU
/// <see cref="ForwardPass"/> and the Vulkan <see cref="GpuForwardPass"/>. An nx x ny image occupies nx*ny contiguous
/// KV slots but advances the position by only max(nx, ny); its tokens get (t, h, w) = (p0, p0 + row, p0 + col).
/// Text tokens get (p, p, p), shifted back by every earlier image's saving. KV slots stay contiguous; only the RoPE
/// angles change.
/// </summary>
internal sealed class MRopeImageLayout
{
    private readonly List<(int Start, int Nx, int Ny)> _images = [];

    public int Count => _images.Count;

    /// <summary>Registers an image whose soft tokens occupy slots starting at <paramref name="startSlot"/>,
    /// replacing any registered at or after it.</summary>
    public void Add(int startSlot, int nx, int ny)
    {
        _images.RemoveAll(r => r.Start >= startSlot);
        _images.Add((startSlot, nx, ny));
    }

    /// <summary>Drops images that do not fit entirely inside the first <paramref name="length"/> slots.</summary>
    public void TruncateTo(int length) => _images.RemoveAll(r => r.Start + r.Nx * r.Ny > length);

    public void Clear() => _images.Clear();

    /// <summary>(t, h, w) M-RoPE position of a KV slot; all three are equal for text tokens.</summary>
    public (int T, int H, int W) Position(int slot)
    {
        int shift = 0;
        foreach (var (start, nx, ny) in _images)
        {
            int n = nx * ny;
            if (slot >= start + n) { shift += n - Math.Max(nx, ny); continue; }
            if (slot < start) break;
            int p0 = start - shift, i = slot - start;
            return (p0, p0 + i / nx, p0 + i % nx);
        }
        int p = slot - shift;
        return (p, p, p);
    }

    /// <summary>Position each of the <paramref name="pairPositions"/>.Length rotation pairs uses at
    /// <paramref name="slot"/>: its section's t, h or w, and 0 for the fourth (never-rotating) section.</summary>
    public void FillPairPositions(ModelHyperparams hp, int slot, Span<float> pairPositions)
    {
        var (t, h, w) = Position(slot);
        for (int i = 0; i < pairPositions.Length; i++)
            pairPositions[i] = hp.MRopeComponent(i) switch { 0 => t, 1 => h, 2 => w, _ => 0 };
    }
}
