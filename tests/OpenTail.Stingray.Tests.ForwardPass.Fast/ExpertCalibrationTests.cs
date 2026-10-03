using OpenTail.Stingray.Engine.Calibration;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>The decision rules and the profile file of <see cref="ExpertCalibration"/>, driven with synthetic numbers (no GPU needed).</summary>
public sealed class ExpertCalibrationTests
{
    private static DtypeCalibration Make(double decodeGBs, double cpuG, double copyGBs, double gpuG, double? copyOv = null) =>
        new("Q4_K", 1024, 2048, 93, decodeGBs, cpuG, 9.0, copyGBs, gpuG, cpuG * 0.9, copyOv);

    [Fact]
    public void Decode_UsesFreeTokenRule_OnTheOverlappedCopyRate()
    {
        // 32 GB/s vs 2.2 GB/s overlapped: far above 2x -> CPU. Standalone copy 20 GB/s would say 1.6x -> fetch, so the overlapped rate must win.
        Assert.StartsWith("compute on the CPU", ExpertCalibration.Advise(Make(32, 300, 20, 300, copyOv: 2.2)).DecodeColdExperts);
        Assert.StartsWith("fetch to the GPU", ExpertCalibration.Advise(Make(32, 300, 20, 300, copyOv: 20)).DecodeColdExperts);
    }

    [Fact]
    public void Decode_BoundaryIsStrictlyAboveTwoTimes()
    {
        Assert.StartsWith("fetch", ExpertCalibration.Advise(Make(40, 300, 20, 300)).DecodeColdExperts);   // exactly 2.0x
        Assert.StartsWith("compute", ExpertCalibration.Advise(Make(40.1, 300, 20, 300)).DecodeColdExperts);
    }

    [Fact]
    public void ResidentPrefill_NeedsTheMarginToPickTheGpu()
    {
        Assert.StartsWith("CPU", ExpertCalibration.Advise(Make(30, 300, 5, 360)).PrefillResidentExperts);   // 1.2x < 1.25x
        Assert.StartsWith("GPU", ExpertCalibration.Advise(Make(30, 300, 5, 400)).PrefillResidentExperts);
    }

    [Fact]
    public void StreamedPrefill_ComparesCopyPlusGpuAgainstCpuAtAggregateRate()
    {
        // Slow copy (0.05 GB/s) makes streaming lose even with a fast GPU; fast copy wins.
        Assert.StartsWith("CPU", ExpertCalibration.Advise(Make(30, 300, 0.05, 3000)).PrefillStreamedExperts);
        Assert.StartsWith("GPU", ExpertCalibration.Advise(Make(30, 300, 20, 3000)).PrefillStreamedExperts);
    }

    [Fact]
    public void NoGpu_AdvisesCpu()
    {
        var c = new DtypeCalibration("Q4_K", 1024, 2048, 93, 30, 300, 9, null, null, null, null);
        Assert.StartsWith("CPU", ExpertCalibration.Advise(c).DecodeColdExperts);
    }

    [Fact]
    public void Profile_RoundTripsAndRejectsOtherSchemas()
    {
        string dir = Path.Combine(Path.GetTempPath(), "stingray-calib-" + Guid.NewGuid().ToString("N"));
        try
        {
            string path = ExpertCalibration.ProfilePath(dir, "dev_8t");
            var p = new CalibrationProfile(ExpertCalibration.SchemaVersion, "2026-10-04T00:00:00Z", "dev_8t", "cpu", "gpu", true,
                [Make(32, 300, 5, 300, copyOv: 2.2), new DtypeCalibration("Q6_K", 1024, 2048, 93, 1, 2, 3, null, null, null, null)]);
            ExpertCalibration.Save(p, path);
            var back = ExpertCalibration.Load(path)!;
            Assert.Equal("dev_8t", back.DeviceKey);
            Assert.Equal(true, back.GpuIntegrated);
            Assert.Equal(p.Dtypes[0], back.Dtypes[0]);
            Assert.Null(back.Dtypes[1].GpuCopyGBs);

            File.WriteAllText(path, File.ReadAllText(path).Replace("\"schema\": 1", "\"schema\": 99"));
            Assert.Null(ExpertCalibration.Load(path));
            File.WriteAllText(path, "{ not json");
            Assert.Null(ExpertCalibration.Load(path));
            Assert.Null(ExpertCalibration.Load(Path.Combine(dir, "missing.json")));
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
    }
}
