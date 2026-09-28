namespace OpenTail.Stingray.Tests.Vulkan;
public sealed class ZzHybRopeTmp
{
    [Fact]
    public void Compare()
    {
        if (Environment.GetEnvironmentVariable("ZZ_HYB") != "1") return;
        string sp = Environment.GetEnvironmentVariable("ZZ_SP")!;
        string m = Environment.GetEnvironmentVariable("ZZ_MODEL")!;
        int gpuLayers = int.Parse(Environment.GetEnvironmentVariable("ZZ_GL") ?? "14");
        using var model = GgufModel.Open($@"C:/Git-Public/OpenTail.Stingray/models/_models/{m}.gguf");
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int[] ids = GgufTokenizer.FromGgufModel(model).Encode(File.ReadAllText(Path.Combine(sp, "wiki20k.txt"))).Take(1501).ToArray();
        using var cpuB = new CpuBackend();
        using var cpu = new Engine.ForwardPass(model, cpuB, hp, maxContextLength: 2048);
        using var vk = new VulkanBackend();
        var plan = TierPlanner.Plan(model, hp, HardwareProfile.Detect(vk), false, requestedCtxSize: 2048, pinGpuLayers: gpuLayers);
        using var hyb = new HybridForwardPass(model, vk, hp, plan, false);
        for (int i = 0; i < ids.Length; i++)
        {
            float[] a = cpu.Forward(ids[i], i).ToArray();
            float[] b = hyb.Forward(ids[i], i).ToArray();
            if (i == 100 || i == ids.Length - 1)
            {
                double d = 0, na = 0, nb = 0;
                for (int j = 0; j < a.Length; j++) { d += (double)a[j] * b[j]; na += (double)a[j] * a[j]; nb += (double)b[j] * b[j]; }
                Console.WriteLine($"[hyb-rope] pos {i}: cos {d / Math.Sqrt(na * nb):F6}");
            }
        }
    }
}
