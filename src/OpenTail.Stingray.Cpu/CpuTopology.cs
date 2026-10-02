using System.Runtime.InteropServices;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Physical core count, used as the default kernel thread cap (as llama.cpp's
/// <c>cpu_get_num_physical_cores</c> does). Measured 2026-10-02 on a Ryzen 7 5700G (8 cores /
/// 16 threads): 8 threads matched or beat 16 on decode and prefill for SmolLM2-135M/1.7B,
/// Qwen2.5-0.5B, Granite-Vision-3.2-2B and Mistral-7B (docs/4-performance/perf-sweep-plan.md
/// Phase 3). Falls back to <see cref="Environment.ProcessorCount"/> whenever the topology cannot
/// be read, so the default is never lower-confidence than before.
/// </summary>
public static partial class CpuTopology
{
    private static readonly Lazy<int> s_physicalCores = new(Detect);

    /// <summary>Number of physical cores (SMT siblings counted once), or the logical count if unknown.</summary>
    public static int PhysicalCores => s_physicalCores.Value;

    private static int Detect()
    {
        int logical = Environment.ProcessorCount;
        int cores = 0;
        try
        {
            if (OperatingSystem.IsWindows()) cores = WindowsCoreCount();
            else if (OperatingSystem.IsLinux()) cores = LinuxCoreCount();
        }
        catch
        {
            cores = 0;
        }
        return cores > 0 && cores <= logical ? cores : logical;
    }

    private const int RelationProcessorCore = 0;

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static unsafe partial bool GetLogicalProcessorInformationEx(int relationshipType, byte* buffer, ref uint returnedLength);

    private static unsafe int WindowsCoreCount()
    {
        uint len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, null, ref len);
        if (len == 0) return 0;
        var buf = new byte[len];
        fixed (byte* p = buf)
        {
            if (!GetLogicalProcessorInformationEx(RelationProcessorCore, p, ref len)) return 0;
            // Each SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX record starts with
            // { int Relationship; uint Size; ... }; one record per physical core.
            int count = 0;
            for (uint off = 0; off + 8 <= len;)
            {
                uint size = *(uint*)(p + off + 4);
                if (size == 0) break;
                if (*(int*)(p + off) == RelationProcessorCore) count++;
                off += size;
            }
            return count;
        }
    }

    private static int LinuxCoreCount()
    {
        var seen = new HashSet<string>();
        foreach (var dir in Directory.EnumerateDirectories("/sys/devices/system/cpu", "cpu*"))
        {
            string topo = Path.Combine(dir, "topology");
            string coreFile = Path.Combine(topo, "core_id"), pkgFile = Path.Combine(topo, "physical_package_id");
            if (!File.Exists(coreFile) || !File.Exists(pkgFile)) continue;
            seen.Add(File.ReadAllText(pkgFile).Trim() + ":" + File.ReadAllText(coreFile).Trim());
        }
        return seen.Count;
    }
}
