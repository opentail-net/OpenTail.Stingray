using OpenTail.Stingray.Diffusion;

namespace OpenTail.Stingray.Cli;

public sealed partial class ImageCommand
{
    private static int RunFlux(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string? vaePath           = s.VaePath           ?? ResolveFluxVae();
        string? clipLPath         = s.ClipLPath         ?? ResolveFluxClipL();
        string? clipTokenizerPath = s.ClipTokenizerPath ?? ResolveFluxClipTokenizer();
        string? t5XXLPath         = s.T5XXLPath         ?? ResolveFluxT5XXL();
        string? t5TokenizerPath   = s.T5TokenizerPath   ?? ResolveFluxT5Tokenizer();

        if (!RequireFile(vaePath,           "--vae",            "models/flux1-schnell/ae.safetensors"))        return 1;
        if (!RequireFile(clipLPath,         "--clip-l",         "models/flux1-schnell/clip_l.safetensors"))     return 1;
        if (!RequireFile(clipTokenizerPath, "--clip-tokenizer", "models/flux1-schnell/tokenizer_clip/tokenizer.json")) return 1;
        if (!RequireFile(t5XXLPath,         "--t5xxl",          "models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors")) return 1;
        if (!RequireFile(t5TokenizerPath,   "--t5-tokenizer",   "models/flux1-schnell/tokenizer_t5/tokenizer.json")) return 1;

        string output = s.OutputPath ?? "output.png";
        int steps     = s.Steps > 0 ? s.Steps : IsDistilled(modelPath) ? 4 : 20;
        float cfg     = s.CfgScale >= 0f ? s.CfgScale : 1.0f;

        string backendChoice = (s.Backend ?? "auto").ToLowerInvariant();
        if (deviceNone && (backendChoice is "cuda" or "vulkan"))
            AnsiConsole.MarkupLine("[yellow]Note:[/] --device none overrides --backend; running on CPU.");
        bool forceCpu    = s.NGpuLayers == 0 || backendChoice == "cpu" || deviceNone;
        bool forceCuda   = backendChoice == "cuda";
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
                catch when (deviceIndex < 0)
                {
                    gpu = null;
                }
            }
        }

        AnsiConsole.MarkupLine("[bold]FLUX.1[/] (MM-DiT + CLIP-L + T5-XXL)");
        AnsiConsole.MarkupLine($"[dim]DiT:[/]     {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]VAE:[/]     {Markup.Escape(vaePath!)}");
        AnsiConsole.MarkupLine($"[dim]CLIP-L:[/]  {Markup.Escape(clipLPath!)}");
        AnsiConsole.MarkupLine($"[dim]T5-XXL:[/]  {Markup.Escape(t5XXLPath!)}");
        AnsiConsole.MarkupLine($"[dim]Backend:[/] {(gpu is not null ? $"GPU ({gpu.GetType().Name})" : "CPU")}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]    {s.Width}×{s.Height}  steps={steps}  cfg={cfg:F1}  seed={s.Seed}");
        if (s.UpscalerPath is not null)
            AnsiConsole.MarkupLine($"[dim]Upscaler:[/] {Markup.Escape(s.UpscalerPath)}");
        AnsiConsole.MarkupLine($"[dim]Output:[/]  {Markup.Escape(output)}");
        AnsiConsole.WriteLine();

        try
        {
            var sw = Stopwatch.StartNew();
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .Start("Loading FLUX models…", ctx =>
                {
                    RRDBNet? upscaler = null;
                    IComputeBackend? upscalerGpu = null;
                    try
                    {
                        ctx.Status("Loading DiT + VAE + CLIP-L + T5-XXL…");
                        var pipeline = ImagePipeline.Load(
                            modelPath,
                            vaePath!,
                            clipLPath!,   clipTokenizerPath!,
                            t5XXLPath!,   t5TokenizerPath!,
                            gpu);

                        if (s.UpscalerPath is not null)
                        {
                            // Prefer CudaBackend directly when NVRTC image kernels are available.
                            // Fall back to VulkanBackend, then CudaBackend SGEMM path.
                            // --device none/cpu (deviceNone) skips the GPU upscaler entirely.
                            try
                            {
                                if (deviceNone)
                                {
                                    AnsiConsole.MarkupLine("[dim]Upscaler backend:[/] CPU (--device none)");
                                }
                                else if (CudaBackend.IsAvailable())
                                {
                                    var cudaForUpscaler = CudaBackend.Create();
                                    if (cudaForUpscaler.ImageKernelsAvailable)
                                    {
                                        upscalerGpu = cudaForUpscaler;
                                        AnsiConsole.MarkupLine("[dim]Upscaler backend:[/] GPU (CUDA)");
                                    }
                                    else
                                    {
                                        ((IDisposable)cudaForUpscaler).Dispose();
                                        throw new InvalidOperationException("NVRTC not available");
                                    }
                                }
                                else
                                {
                                    throw new InvalidOperationException("CUDA not available");
                                }
                            }
                            catch
                            {
                                try
                                {
                                    upscalerGpu = new VulkanBackend(deviceIndex);
                                    AnsiConsole.MarkupLine("[dim]Upscaler backend:[/] GPU (Vulkan, native kernels)");
                                }
                                // An explicit --device error propagates to the outer handler (surfaced,
                                // not swallowed into a wrong-device or silent-CPU fallback).
                                catch when (deviceIndex < 0)
                                {
                                    try
                                    {
                                        if (CudaBackend.IsAvailable())
                                        {
                                            upscalerGpu = CudaBackend.Create();
                                            AnsiConsole.MarkupLine("[dim]Upscaler backend:[/] GPU (CUDA)");
                                        }
                                    }
                                    catch
                                    {
                                        upscalerGpu = null;
                                        AnsiConsole.MarkupLine("[dim]Upscaler backend:[/] CPU (no GPU detected)");
                                    }
                                }
                            }
                            ctx.Status("Loading RRDBNet upscaler…");
                            upscaler = RRDBNet.Load(s.UpscalerPath, upscalerGpu ?? gpu);
                            AnsiConsole.MarkupLine($"[dim]Upscaler:[/] RRDBNet ×{upscaler.Scale} loaded");
                        }

                        if (gpu is VulkanBackend) VulkanBackend.ResetGpuProfile();
                        using (pipeline)
                        {
                            ctx.Status($"Generating {s.Width}×{s.Height} image…");
                            var stepSw = Stopwatch.StartNew();
                            pipeline.Generate(
                                s.Prompt!, s.Width, s.Height, steps, cfg, s.Seed, output,
                                progress: (step, total) =>
                                {
                                    ctx.Status($"Step {step}/{total} — {stepSw.Elapsed.TotalSeconds:F1}s elapsed…");
                                    stepSw.Restart();
                                },
                                upscaler: upscaler,
                                upscaleBlend: s.UpscaleBlend);
                        }
                        if (gpu is VulkanBackend) VulkanBackend.PrintGpuProfile("FLUX full run");

                        AnsiConsole.MarkupLine($"[green]✓[/] Done in [cyan]{sw.Elapsed.TotalSeconds:F1}s[/]");
                        AnsiConsole.MarkupLine($"[green]✓[/] Image saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/]");
                    }
                    finally
                    {
                        upscaler?.Dispose();
                        if (upscalerGpu is IDisposable dg && !ReferenceEquals(dg, gpu)) dg.Dispose();
                        if (gpu is IDisposable dGpu) dGpu.Dispose();
                    }
                });
        }
        catch (Exception ex)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.ToString())}");
            return 1;
        }

        return 0;
    }

    private static string? ResolveFluxVae()
    {
        foreach (var c in new[] {
            "models/flux1-schnell/ae.safetensors",
            "models/ae.safetensors" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveFluxClipL()
    {
        foreach (var c in new[] {
            "models/flux1-schnell/clip_l.safetensors",
            "models/clip_l.safetensors" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveFluxClipTokenizer()
    {
        foreach (var c in new[] {
            "models/flux1-schnell/tokenizer_clip/tokenizer.json",
            "models/flux1-schnell/tokenizer_2/tokenizer.json",
            "models/tokenizer_clip.json" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveFluxT5XXL()
    {
        foreach (var c in new[] {
            "models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors",
            "models/flux1-schnell/t5xxl_fp16.safetensors",
            "models/t5xxl_fp16.safetensors" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveFluxT5Tokenizer()
    {
        foreach (var c in new[] {
            "models/flux1-schnell/tokenizer_t5/tokenizer.json",
            "models/tokenizer_t5.json" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static int RunFlux2(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string output = s.OutputPath ?? "output.png";
        int steps = s.Steps > 0 ? s.Steps : 20;
        float guidance = s.CfgScale >= 0 ? s.CfgScale : 3.5f;

        if (string.IsNullOrWhiteSpace(s.TextEncoderPath) || string.IsNullOrWhiteSpace(s.VaePath))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] FLUX.2 needs [yellow]--text-encoder <Mistral-Small GGUF>[/] and [yellow]--vae <flux2-vae.safetensors>[/].");
            return 1;
        }

        IComputeBackend? gpu = null;
        if (!deviceNone && deviceIndex >= 0)
        {
            try { gpu = new VulkanBackend(deviceIndex); }
            catch (Exception ex) { AnsiConsole.MarkupLine($"[yellow]Note:[/] Vulkan GPU init failed ({Markup.Escape(ex.Message)}); falling back to CPU."); }
        }

        AnsiConsole.MarkupLine("[bold]FLUX.2 (MM-DiT + Mistral-Small conditioning)[/]");
        AnsiConsole.MarkupLine($"[dim]Model:[/]    {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {s.Width}×{s.Height}  steps={steps}  guidance={guidance}  seed={s.Seed}");
        AnsiConsole.MarkupLine($"[dim]Output:[/]   {Markup.Escape(output)}");
        AnsiConsole.WriteLine();

        try
        {
            var sw = Stopwatch.StartNew();
            using var pipeline = OpenTail.Stingray.Diffusion.Flux2.Flux2Pipeline.Load(modelPath, s.TextEncoderPath, s.VaePath, ditBackend: gpu);
            string target = gpu is not null ? gpu.Name : "CPU";
            pipeline.Generate(new OpenTail.Stingray.Diffusion.Flux2.Flux2GenerationRequest
            {
                Prompt = s.Prompt!,
                Width = s.Width,
                Height = s.Height,
                Steps = steps,
                Guidance = guidance,
                Seed = s.Seed,
                OutputPath = output,
                Progress = (step, total) => Console.Error.WriteLine($"[FLUX.2] step {step}/{total} on {target}"),
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
            (gpu as IDisposable)?.Dispose();
        }
    }
}
