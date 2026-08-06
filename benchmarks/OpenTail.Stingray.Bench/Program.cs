using BenchmarkDotNet.Running;

// Manual (non-BenchmarkDotNet) harnesses dispatch before the switcher.
if (args.Contains("--cb"))
{
    await OpenTail.Stingray.Bench.ContinuousBatchingHarness.Run(args);
    return;
}

if (args.Contains("--gdn"))
{
    OpenTail.Stingray.Bench.GdnPrefillHarness.Run(args);
    return;
}

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
