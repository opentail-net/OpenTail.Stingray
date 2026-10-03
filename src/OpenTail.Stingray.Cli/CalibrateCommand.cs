using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Engine.Calibration;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Times the real expert kernels on this machine (CPU matmul, host-to-device copy, GPU matmul, and the CPU and copy together),
/// stores the result in the model home and prints which side of the CPU/GPU split each kind of expert work belongs on.
/// </summary>
public sealed class CalibrateCommand : Command<CalibrateCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("--seconds <N>")]
        [Description("Window per measurement in seconds (default 2)")]
        public int Seconds { get; init; } = 2;

        [CommandOption("--no-gpu")]
        [Description("Skip the GPU measurements")]
        public bool NoGpu { get; init; }

        [CommandOption("--tokens <N>")]
        [Description("Rows per expert for the prefill measurement (default 93, the OLMoE average)")]
        public int Tokens { get; init; } = 93;

        [CommandOption("--no-save")]
        [Description("Print only; do not write the profile")]
        public bool NoSave { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        var profile = ExpertCalibration.Run([DType.Q4_K, DType.Q6_K], 1024, 2048, settings.Tokens, settings.Seconds, !settings.NoGpu,
            msg => Console.Error.WriteLine("  " + msg));

        Console.WriteLine($"device: {profile.DeviceKey}");
        Console.WriteLine($"cpu:    {profile.Cpu}");
        if (profile.GpuName is not null) Console.WriteLine($"gpu:    {profile.GpuName} (integrated: {profile.GpuIntegrated})");
        foreach (var d in profile.Dtypes)
        {
            Console.WriteLine();
            Console.WriteLine($"{d.Dtype}  ({d.Rows}x{d.Cols} per expert matrix, {d.PrefillTokens} rows for prefill)");
            Console.WriteLine($"  CPU decode            {d.CpuDecodeGBs,8:F1} GB/s of weights");
            Console.WriteLine($"  CPU prefill           {d.CpuPrefillGflops,8:F1} GFLOP/s   ({d.CpuPrefillMsPerExpert:F2} ms per expert matrix)");
            if (d.GpuCopyGBs is { } c) Console.WriteLine($"  host-to-device copy   {c,8:F1} GB/s");
            if (d.GpuMatmulGflops is { } g) Console.WriteLine($"  GPU matmul (resident) {g,8:F1} GFLOP/s");
            if (d.CpuOverlapGflops is { } o) Console.WriteLine($"  CPU while copying     {o,8:F1} GFLOP/s");
            if (d.CopyOverlapGBs is { } co) Console.WriteLine($"  copy while CPU runs   {co,8:F1} GB/s");
            var a = ExpertCalibration.Advise(d);
            Console.WriteLine($"  -> decode, cold experts:        {a.DecodeColdExperts}");
            Console.WriteLine($"  -> prefill, resident experts:   {a.PrefillResidentExperts}");
            Console.WriteLine($"  -> prefill, experts to copy:    {a.PrefillStreamedExperts}");
            if (a.Note.Length > 0) Console.WriteLine($"     {a.Note}");
        }

        if (!settings.NoSave)
        {
            string path = ExpertCalibration.ProfilePath(ModelHome.Default().Root, profile.DeviceKey);
            ExpertCalibration.Save(profile, path);
            Console.WriteLine();
            Console.WriteLine($"profile written to {path}");
        }
        return 0;
    }
}
