using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.StableDiffusion;
using OpenTail.Stingray.Diffusion.SDXL;
using OpenTail.Stingray.Diffusion.SD3;

namespace OpenTail.Stingray.Cli;

public sealed partial class ImageCommand
{
    private static int RunStableDiffusion(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string output = s.OutputPath ?? "output.png";
        int steps = s.Steps > 0 ? s.Steps : 20;
        float guidance = s.CfgScale >= 0f ? s.CfgScale : 7.5f;

        string backendChoice = (s.Backend ?? "auto").ToLowerInvariant();
        if (deviceNone && (backendChoice is "cuda" or "vulkan"))
            AnsiConsole.MarkupLine("[yellow]Note:[/] --device none overrides --backend; running on CPU.");
        bool forceCpu = s.NGpuLayers == 0 || backendChoice == "cpu" || deviceNone;
        bool forceCuda = backendChoice == "cuda";
        bool forceVulkan = backendChoice == "vulkan";

        IComputeBackend? gpu = null;
        if (!forceCpu)
        {
            if (forceCuda || (!forceVulkan && CudaBackend.IsAvailable()))
            {
                gpu = CudaBackend.Create();
                AnsiConsole.MarkupLine("[dim]Compute:[/]  [green]CUDA GPU[/]");
            }
            else
            {
                try
                {
                    var vulkan = new VulkanBackend(deviceIndex);
                    gpu = vulkan;
                    vulkan.PrintDeviceInfo();
                    AnsiConsole.MarkupLine($"[dim]Compute:[/]  [green]Vulkan GPU[/] ({vulkan.Name})");
                }
                catch when (deviceIndex < 0)
                {
                    AnsiConsole.MarkupLine("[dim]Compute:[/]  [yellow]CPU (no Vulkan GPU detected)[/]");
                }
            }
        }
        else
        {
            AnsiConsole.MarkupLine("[dim]Compute:[/]  [yellow]CPU SIMD[/]");
        }

        var schedulerType = ParseScheduler(s.Sampler);
        float[]? initImage = LoadInitImage(s.InitImagePath);
        // SD-Turbo's own scheduler_config.json sets timestep_spacing="trailing" too (same real
        // diffusers config as SDXL-Turbo) -- reuse the existing turbo/schnell/lcm heuristic.
        var timestepSpacing = IsDistilled(modelPath) ? TimestepSpacing.Trailing : TimestepSpacing.Linspace;

        AnsiConsole.MarkupLine("[bold]Stable Diffusion 1.5[/]");
        AnsiConsole.MarkupLine($"[dim]Model:[/]    {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {s.Width}×{s.Height}  steps={steps}  guidance={guidance}  seed={s.Seed}  sampler={schedulerType}");
        if (initImage is not null)
            AnsiConsole.MarkupLine($"[dim]Init Image:[/] {Markup.Escape(s.InitImagePath!)} (strength={s.Strength:F2})");
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

            using var pipeline = StableDiffusionPipeline.Load(modelPath, s.ClipTokenizerPath, gpu);

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
                        seed: s.Seed,
                        schedulerType: schedulerType,
                        outputPath: output,
                        progress: (step, total) => ctx.Status($"Denoising step {step}/{total} on {target}…"),
                        upscaler: upscaler,
                        upscaleBlend: s.UpscaleBlend,
                        initImageRgb: initImage,
                        strength: s.Strength,
                        timestepSpacing: timestepSpacing);
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

    private static int RunSd3(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string output = s.OutputPath ?? "output.png";
        int steps = s.Steps > 0 ? s.Steps : 20;
        float guidance = s.CfgScale >= 0f ? s.CfgScale : 4.5f;

        string backendChoice = (s.Backend ?? "auto").ToLowerInvariant();
        if (deviceNone && (backendChoice is "cuda" or "vulkan"))
            AnsiConsole.MarkupLine("[yellow]Note:[/] --device none overrides --backend; running on CPU.");
        bool forceCpu = s.NGpuLayers == 0 || backendChoice == "cpu" || deviceNone;
        bool forceCuda = backendChoice == "cuda";
        bool forceVulkan = backendChoice == "vulkan";

        IComputeBackend? gpu = null;
        if (!forceCpu)
        {
            if (forceCuda || (!forceVulkan && CudaBackend.IsAvailable()))
            {
                gpu = CudaBackend.Create();
            }
            else
            {
                try
                {
                    var vulkan = new VulkanBackend(deviceIndex);
                    gpu = vulkan;
                    vulkan.PrintDeviceInfo();
                }
                catch when (deviceIndex < 0) { }
            }
        }

        int width = s.Width > 0 ? s.Width : 1024;
        int height = s.Height > 0 ? s.Height : 1024;

        AnsiConsole.MarkupLine("[bold]Stable Diffusion 3 / 3.5 (MMDiT)[/]");
        AnsiConsole.MarkupLine($"[dim]Model:[/]    {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {width}×{height}  steps={steps}  guidance={guidance}  seed={s.Seed}");
        if (gpu is not null)
            AnsiConsole.MarkupLine($"[dim]Compute:[/]  [green]{Markup.Escape(gpu.Name)}[/]");
        else
            AnsiConsole.MarkupLine("[dim]Compute:[/]  [yellow]CPU SIMD[/]");
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

            // Real bug found and fixed 2026-09-21 (SD3.5 composition-bug investigation): this call
            // never passed --t5xxl/--t5-tokenizer even though both flags exist (wired for FLUX
            // only) -- every SD3/3.5 `stingray image` invocation silently ran with T5 conditioning
            // entirely zero-filled, a real and severe conditioning gap distinct from any of this
            // investigation's other findings. Threaded through to match Sd3BaselineTests' own
            // already-correct xunit-test wiring.
            using var pipeline = s.ClipGPath is not null && s.ClipLPath is not null && s.VaePath is not null
                ? Sd3Pipeline.LoadSeparate(s.ClipLPath, s.ClipGPath, modelPath, s.VaePath, s.ClipTokenizerPath, gpu,
                    t5EncoderPath: s.T5XXLPath, t5TokenizerPath: s.T5TokenizerPath)
                : Sd3Pipeline.Load(modelPath, s.ClipTokenizerPath, gpu);

            string target = gpu is not null ? gpu.Name : "CPU";
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .Start($"Denoising ({steps} steps on {target})…", ctx =>
                {
                    pipeline.Generate(
                        prompt: s.Prompt!,
                        negativePrompt: s.NegativePrompt,
                        width: width,
                        height: height,
                        steps: steps,
                        guidance: guidance,
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

    private static int RunSdxl(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string output = s.OutputPath ?? "output.png";
        // Distilled checkpoints (SDXL-Turbo) are trained for guidance_scale=0 (CFG off) and 1-4 steps.
        // Defaulting them to CFG 7.5 degraded the image AND ran the UNet twice per step (cond + uncond).
        bool distilled = IsDistilled(modelPath);
        int steps = s.Steps > 0 ? s.Steps : distilled ? 4 : 20;
        float guidance = s.CfgScale >= 0f ? s.CfgScale : distilled ? 0f : 7.5f;

        string backendChoice = (s.Backend ?? "auto").ToLowerInvariant();
        if (deviceNone && (backendChoice is "cuda" or "vulkan"))
            AnsiConsole.MarkupLine("[yellow]Note:[/] --device none overrides --backend; running on CPU.");
        bool forceCpu = s.NGpuLayers == 0 || backendChoice == "cpu" || deviceNone;
        bool forceCuda = backendChoice == "cuda";
        bool forceVulkan = backendChoice == "vulkan";

        IComputeBackend? gpu = null;
        if (!forceCpu)
        {
            if (forceCuda || (!forceVulkan && CudaBackend.IsAvailable()))
            {
                gpu = CudaBackend.Create();
            }
            else
            {
                try
                {
                    var vulkan = new VulkanBackend(deviceIndex);
                    gpu = vulkan;
                    vulkan.PrintDeviceInfo();
                }
                catch when (deviceIndex < 0) { }
            }
        }

        int width = s.Width > 0 ? s.Width : 1024;
        int height = s.Height > 0 ? s.Height : 1024;

        AnsiConsole.MarkupLine("[bold]Stable Diffusion XL (SDXL)[/]");
        AnsiConsole.MarkupLine($"[dim]Model:[/]    {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {width}×{height}  steps={steps}  guidance={guidance}  seed={s.Seed}");
        if (gpu is not null)
            AnsiConsole.MarkupLine($"[dim]Compute:[/]  [green]{Markup.Escape(gpu.Name)}[/]");
        else
            AnsiConsole.MarkupLine("[dim]Compute:[/]  [yellow]CPU SIMD[/]");
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

            var schedulerType = ParseScheduler(s.Sampler);
            float[]? initImage = LoadInitImage(s.InitImagePath);
            // SDXL-Turbo's own scheduler_config.json sets timestep_spacing="trailing" (confirmed
            // 2026-09-11 against real diffusers source) -- base SDXL/SD1.5 use "linspace". Reuse
            // the existing turbo/schnell/lcm filename heuristic (IsDistilled) rather than adding a
            // second, redundant detector.
            var timestepSpacing = IsDistilled(modelPath) ? TimestepSpacing.Trailing : TimestepSpacing.Linspace;

            using var pipeline = SdxlPipeline.Load(modelPath, s.ClipTokenizerPath, gpu);

            string target = gpu is not null ? gpu.Name : "CPU";
            if (gpu is VulkanBackend) VulkanBackend.ResetGpuProfile();
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .Start($"Denoising ({steps} steps on {target})…", ctx =>
                {
                    pipeline.Generate(
                        prompt: s.Prompt!,
                        negativePrompt: s.NegativePrompt,
                        width: width,
                        height: height,
                        steps: steps,
                        guidance: guidance,
                        seed: s.Seed,
                        schedulerType: schedulerType,
                        outputPath: output,
                        progress: (step, total) => ctx.Status($"Denoising step {step}/{total} on {target}…"),
                        upscaler: upscaler,
                        upscaleBlend: s.UpscaleBlend,
                        initImageRgb: initImage,
                        strength: s.Strength,
                        timestepSpacing: timestepSpacing,
                        log: s.Verbose ? msg => AnsiConsole.MarkupLine($"[dim]{Markup.Escape(msg)}[/]") : null);
                });
            if (gpu is VulkanBackend)
            {
                VulkanBackend.PrintGpuProfile("SDXL full run");
                VulkanBackend.PrintGpuMemoryProfile("SDXL full run");
            }
            SafetensorsLoader.PrintWeightReadProfile("SDXL full run");

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
