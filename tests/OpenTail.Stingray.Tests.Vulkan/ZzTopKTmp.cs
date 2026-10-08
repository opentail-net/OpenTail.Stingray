namespace OpenTail.Stingray.Tests.Vulkan;
public sealed class ZzTopKTmp
{
    [Fact]
    public void TopK()
    {
        if (Environment.GetEnvironmentVariable("ZZ_TOPK") != "1") return;
        string m = Environment.GetEnvironmentVariable("ZZ_MODEL")!;
        int[] prompt = Environment.GetEnvironmentVariable("ZZ_PROMPT")!.Split(',').Select(int.Parse).ToArray();
        int[] forced = Environment.GetEnvironmentVariable("ZZ_FORCED")!.Split(',').Select(int.Parse).ToArray();
        using var model = GgufModel.Open($@"C:/Git-Public/OpenTail.Stingray/models/_models/{m}.gguf");
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        using var backend = new CpuBackend();
        using var vk = Environment.GetEnvironmentVariable("ZZ_GPU") == "1" ? new VulkanBackend() : null;
        int split = int.Parse(Environment.GetEnvironmentVariable("ZZ_SPLIT") ?? "0");
        using var vk2 = split > 0 ? new VulkanBackend() : null;
        using IForwardPass fwd = vk is not null
            ? new DeepSeek2GpuForwardPass(model, vk, hp, maxContextLength: 2048)
            : vk2 is not null ? new VulkanLayerSplitForwardPass(model, vk2, hp, split, 2048)
            : new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        if (Environment.GetEnvironmentVariable("ZZ_GREEDY") is { } gs)
        {
            float[] g = fwd.Prefill(prompt).ToArray();
            for (int st = 0; st < int.Parse(gs); st++)
            {
                var ord = g.Select((v, i) => (v, i)).OrderByDescending(t => t.v).Take(2).ToArray();
                Console.WriteLine($"greedy {st}: {ord[0].i} margin {ord[0].v - ord[1].v:F3} (2nd {ord[1].i})");
                g = fwd.Forward(ord[0].i, prompt.Length + st).ToArray();
            }
            return;
        }
        float[] l;
        if (Environment.GetEnvironmentVariable("ZZ_SEQ") == "1")
        {
            l = [];
            for (int i = 0; i < prompt.Length; i++) l = fwd.Forward(prompt[i], i).ToArray();
        }
        else l = fwd.Prefill(prompt).ToArray();
        for (int s = 0; s <= forced.Length; s++)
        {
            double max = l.Max(), sum = 0;
            foreach (var v in l) sum += Math.Exp(v - max);
            double lse = max + Math.Log(sum);
            var top = l.Select((v, i) => (v, i)).OrderByDescending(t => t.v).Take(5)
                .Select(t => $"{t.i} {t.v - lse:F3}");
            if (Environment.GetEnvironmentVariable("ZZ_NLL") == "1") { if (s < forced.Length) Console.WriteLine($"nll {s} {lse - l[forced[s]]:F4}"); }
            else Console.WriteLine($"step {s}: {string.Join(" | ", top)}");
            if (s == forced.Length) break;
            l = fwd.Forward(forced[s], prompt.Length + s).ToArray();
        }
    }
}
