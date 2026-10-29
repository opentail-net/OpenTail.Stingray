using OpenTail.Stingray.Diffusion;

namespace OpenTail.Stingray.Cli;

public sealed partial class ImageCommand
{
    private static int RunQwenImage(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string output = s.OutputPath ?? "output.png";
        int steps = s.Steps > 0 ? s.Steps : 20;
        float guidance = s.CfgScale >= 0 ? s.CfgScale : 2.5f;

        IComputeBackend? gpu = null;
        if (!deviceNone && deviceIndex >= 0)
        {
            try { gpu = new VulkanBackend(deviceIndex); }
            catch (Exception ex) { AnsiConsole.MarkupLine($"[yellow]Note:[/] Vulkan GPU init failed ({Markup.Escape(ex.Message)}); falling back to CPU."); }
        }

        AnsiConsole.MarkupLine("[bold]Qwen Image (60-layer MM-DiT + Qwen2.5-VL)[/]");
        AnsiConsole.MarkupLine($"[dim]Model:[/]    {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {s.Width}×{s.Height}  steps={steps}  guidance={guidance}  seed={s.Seed}");
        if (s.UpscalerPath is not null)
            AnsiConsole.MarkupLine($"[dim]Upscaler:[/] {Markup.Escape(s.UpscalerPath)}");
        AnsiConsole.MarkupLine($"[dim]Output:[/]   {Markup.Escape(output)}");
        AnsiConsole.WriteLine();

        try
        {
            var sw = Stopwatch.StartNew();
            RRDBNet? upscaler = null;
            if (s.UpscalerPath is not null)
                upscaler = RRDBNet.Load(s.UpscalerPath, gpu);
            using var pipeline = s.QwenEncoderPath is { Length: > 0 } qwenEncoder
                ? OpenTail.Stingray.Diffusion.QwenImage.QwenImagePipeline.Load(modelPath, qwenEncoder, s.VaePath, gpu)
                : OpenTail.Stingray.Diffusion.QwenImage.QwenImagePipeline.Load(modelPath, s.VaePath, gpu);

            string target = gpu is not null ? gpu.Name : "CPU";
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .Start($"Denoising ({steps} steps on {target})…", ctx =>
                {
                    pipeline.Generate(
                        prompt: s.Prompt!,
                        negativePrompt: s.NegativePrompt,
                        width: s.Width,
                        height: s.Height,
                        steps: steps,
                        guidance: guidance,
                        flowShift: 3.0f,
                        seed: s.Seed,
                        outputPath: output,
                        progress: (step, total) => ctx.Status($"Denoising step {step}/{total} on {target}…"),
                        upscaler: upscaler,
                        upscaleBlend: s.UpscaleBlend);
                });

            sw.Stop();
            AnsiConsole.MarkupLine($"[green]✓[/] Image saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/] in [yellow]{sw.Elapsed.TotalSeconds:F1}s[/]");
            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
        finally
        {
            gpu?.Dispose();
        }
    }
}
