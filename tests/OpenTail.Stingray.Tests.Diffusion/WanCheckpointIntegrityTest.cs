using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// The one category of check never done this entire Priority-0 Wan accuracy investigation
/// (docs/081, update #14): real checkpoint integrity across ALL 30 blocks, not just spot-checks
/// of block 0/29. A corrupted or partially-garbage download would produce exactly the observed
/// symptom (plausible global magnitude, zero coherent structure) regardless of how correct the
/// code is.
/// </summary>
public sealed class WanCheckpointIntegrityTest
{
    private static string? FindModelPath(string relativePath)
    {
        var candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", relativePath),
            Path.Combine("..", "..", "..", relativePath),
            relativePath,
            Path.Combine(AppContext.BaseDirectory, relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath)
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return relativePath;
    }

    [Fact]
    public void DitCheckpoint_AllBlocks_NoNanInfOrSuspiciousAllZero()
    {
        string? path = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));
        if (path is null || !File.Exists(path)) { Console.WriteLine("Checkpoint missing, skipping."); return; }

        using var st = SafetensorsLoader.Open(path);
        Console.WriteLine($"[Integrity] Total tensors: {st.TensorCount}");

        int checkedCount = 0;
        int nanInfCount = 0;
        int allZeroCount = 0;
        double totalMinAbs = double.MaxValue, totalMaxAbs = 0;

        foreach (var name in st.TensorNames)
        {
            // Only check real weight matrices (skip tiny per-block scalars/1D norms which can
            // legitimately be all-zero or small -- focus on the big Linear weight matrices where
            // real corruption would show up as either NaN/Inf or an implausible all-zero block).
            if (!name.EndsWith(".weight") || !name.Contains("blocks.")) continue;

            var shape = st.GetShape(name);
            long count = 1;
            foreach (var d in shape) count *= d;
            if (count < 1000) continue; // skip tiny norm vectors

            var data = st.ReadF32(name);
            checkedCount++;

            bool hasNanInf = false;
            bool allZero = true;
            float maxAbs = 0, minAbsNonzero = float.MaxValue;
            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) { hasNanInf = true; }
                if (v != 0f) { allZero = false; float a = MathF.Abs(v); if (a > maxAbs) maxAbs = a; if (a < minAbsNonzero) minAbsNonzero = a; }
            }

            if (hasNanInf) { nanInfCount++; Log($"[Integrity] NaN/Inf FOUND in {name}"); }
            if (allZero) { allZeroCount++; Log($"[Integrity] ALL-ZERO tensor: {name} (shape=[{string.Join(",", shape)}])"); }
            if (maxAbs > (float)totalMaxAbs) totalMaxAbs = maxAbs;
            if (minAbsNonzero < totalMinAbs) totalMinAbs = minAbsNonzero;
        }

        Log($"[Integrity] Checked {checkedCount} weight tensors across all blocks: nanInf={nanInfCount}, allZero={allZeroCount}, maxAbsSeen={totalMaxAbs:E4}, minAbsNonzeroSeen={totalMinAbs:E4}");

        Assert.Equal(0, nanInfCount);
        Assert.Equal(0, allZeroCount);
    }

    [Fact]
    public void VaeCheckpoint_AllTensors_NoNanInfOrSuspiciousAllZero()
    {
        string? path = FindModelPath(Path.Combine("models", "wan2.1", "Wan2.1_VAE.safetensors"));
        if (path is null || !File.Exists(path)) { Console.WriteLine("VAE checkpoint missing, skipping."); return; }

        using var st = SafetensorsLoader.Open(path);
        Console.WriteLine($"[Integrity/VAE] Total tensors: {st.TensorCount}");

        int checkedCount = 0, nanInfCount = 0, allZeroCount = 0;
        foreach (var name in st.TensorNames)
        {
            if (!name.EndsWith(".weight")) continue;
            var shape = st.GetShape(name);
            long count = 1;
            foreach (var d in shape) count *= d;
            if (count < 1000) continue;

            var data = st.ReadF32(name);
            checkedCount++;
            bool hasNanInf = false, allZero = true;
            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) hasNanInf = true;
                if (v != 0f) allZero = false;
            }
            if (hasNanInf) { nanInfCount++; Log($"[Integrity/VAE] NaN/Inf FOUND in {name}"); }
            if (allZero) { allZeroCount++; Log($"[Integrity/VAE] ALL-ZERO tensor: {name}"); }
        }

        Log($"[Integrity/VAE] Checked {checkedCount} weight tensors: nanInf={nanInfCount}, allZero={allZeroCount}");
        Assert.Equal(0, nanInfCount);
        Assert.Equal(0, allZeroCount);
    }

    private void Log(string msg) => Console.WriteLine(msg);
}
