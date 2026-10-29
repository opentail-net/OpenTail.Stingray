using OpenTail.Stingray.Diffusion;

namespace OpenTail.Stingray.Cli;

public sealed partial class ImageCommand
{
    private static int RunZImage(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        // Auto-discover component paths from models/ directory if not explicitly provided
        string? vaePath       = s.VaePath       ?? ResolveZImageVae();
        string? encoderPath   = s.QwenEncoderPath   ?? ResolveZImageEncoder();
        string? tokenizerPath = s.QwenTokenizerPath ?? ResolveZImageTokenizer();

        if (!RequirePathExists(vaePath, "--vae", "models/z-image-turbo/vae/")) return 1;
        if (!RequireFile(encoderPath,   "--qwen-encoder",   "models/Z-Image-AbliteratedV1.Q5_K_M.gguf")) return 1;
        if (!RequireFile(tokenizerPath, "--qwen-tokenizer", "models/z-image-turbo/tokenizer/tokenizer.json")) return 1;

        string output = s.OutputPath ?? "output.png";
        // Pass -1 when no explicit --steps given so ZImagePipeline uses ZImageParams.DefaultSteps (4)
        int steps = s.Steps > 0 ? s.Steps : -1;

        AnsiConsole.MarkupLine("[bold]Z-Image-Turbo[/] (S3-DiT + Qwen3-4B)");
        AnsiConsole.MarkupLine($"[dim]DiT:[/]      {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]VAE:[/]      {Markup.Escape(vaePath!)}");
        AnsiConsole.MarkupLine($"[dim]Encoder:[/]  {Markup.Escape(encoderPath!)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {s.Width}×{s.Height}  steps={steps}  seed={s.Seed}");
        if (s.UpscalerPath is not null)
            AnsiConsole.MarkupLine($"[dim]Upscaler:[/] {Markup.Escape(s.UpscalerPath)}");
        AnsiConsole.MarkupLine($"[dim]Output:[/]   {Markup.Escape(output)}");
        AnsiConsole.WriteLine();

        try
        {
            var sw = Stopwatch.StartNew();
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .Start("Loading Z-Image models…", ctx =>
                {
                    ctx.Status("Loading DiT + VAE + Qwen3-4B…");

                    IComputeBackend? gpu = null;
                    IComputeBackend? upscalerGpu = null; // owned VulkanBackend created for upscaler (CUDA path)
                    RRDBNet? upscaler = null;
                    try
                    {
                        // Resolve which backend to use:
                        //   --backend cuda|vulkan|cpu  → forced
                        //   -g 0                       → CPU only
                        //   default (-1)               → CUDA → Vulkan → CPU fallback
                        string backendChoice = (s.Backend ?? "auto").ToLowerInvariant();
                        if (deviceNone && (backendChoice is "cuda" or "vulkan"))
                            AnsiConsole.MarkupLine("[yellow]Note:[/] --device none overrides --backend; running on CPU.");
                        bool forceCpu    = s.NGpuLayers == 0 || backendChoice == "cpu" || deviceNone;
                        bool forceCuda   = backendChoice == "cuda";
                        bool forceVulkan = backendChoice == "vulkan";

                        if (!forceCpu)
                        {
                            if (forceCuda || (!forceVulkan && CudaBackend.IsAvailable()))
                            {
                                gpu = CudaBackend.Create();
                                AnsiConsole.MarkupLine("[dim]Backend:[/]  GPU (CUDA cuBLAS)");
                            }
                            else
                            {
                                try
                                {
                                    var vulkan = new VulkanBackend(deviceIndex);
                                    gpu = vulkan;
                                    vulkan.PrintDeviceInfo();
                                    AnsiConsole.MarkupLine("[dim]Backend:[/]  GPU (Vulkan SGEMM)");
                                }
                                // Only auto-select (deviceIndex < 0) silently falls back to CPU.
                                // An explicit --device must surface its error, not vanish into a CPU run.
                                catch when (deviceIndex < 0)
                                {
                                    AnsiConsole.MarkupLine("[dim]Backend:[/]  CPU (no GPU detected)");
                                }
                            }
                        }
                        else
                        {
                            AnsiConsole.MarkupLine("[dim]Backend:[/]  CPU");
                        }

                        if (s.UpscalerPath is not null)
                        {
                            ctx.Status("Loading RRDBNet upscaler…");
                            // Prefer CudaBackend directly when it supports IImageOpsBackend (NVRTC available).
                            // Fall back to a separate VulkanBackend if NVRTC is unavailable.
                            IComputeBackend? upscalerBackend = gpu;
                            if (gpu is CudaBackend cudaGpu && cudaGpu.ImageKernelsAvailable)
                            {
                                upscalerBackend = gpu;
                                AnsiConsole.MarkupLine("[dim]Upscaler backend:[/] GPU (CUDA)");
                            }
                            else if (gpu is not null && gpu is not IImageOpsBackend)
                            {
                                try
                                {
                                    upscalerGpu = new VulkanBackend(deviceIndex);
                                    upscalerBackend = upscalerGpu;
                                    AnsiConsole.MarkupLine("[dim]Upscaler backend:[/] GPU (Vulkan, native kernels)");
                                }
                                catch when (deviceIndex < 0)
                                {
                                    upscalerBackend = gpu; // auto only: fall back to SGEMM on main backend
                                }
                            }
                            if (upscalerBackend is null)
                                AnsiConsole.MarkupLine(deviceNone
                                    ? "[dim]Upscaler backend:[/] CPU (--device none)"
                                    : "[dim]Upscaler backend:[/] CPU");
                            upscaler = RRDBNet.Load(s.UpscalerPath, upscalerBackend);
                            AnsiConsole.MarkupLine($"[dim]Upscaler:[/] RRDBNet ×{upscaler.Scale} loaded");
                        }

                        var pipeline = ZImagePipeline.Load(
                            modelPath,
                            vaePath!,
                            encoderPath!,
                            tokenizerPath!,
                            gpu);

                        using (pipeline)
                        {
                            var stepSw = Stopwatch.StartNew();
                            pipeline.Generate(
                                s.Prompt!, s.Width, s.Height, steps, s.Seed, output,
                                upscaler: upscaler,
                                upscaleBlend: s.UpscaleBlend,
                                progress: (step, total) =>
                                {
                                    ctx.Status($"Step {step}/{total} — {stepSw.Elapsed.TotalSeconds:F1}s elapsed…");
                                    stepSw.Restart();
                                },
                                statusCallback: st => ctx.Status(st));
                        }

                        AnsiConsole.MarkupLine($"[green]✓[/] Done in [cyan]{sw.Elapsed.TotalSeconds:F1}s[/]");
                        AnsiConsole.MarkupLine($"[green]✓[/] Image saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/]");
                    }
                    finally
                    {
                        upscaler?.Dispose();
                        if (upscalerGpu is IDisposable dug) dug.Dispose();
                        if (gpu is IDisposable d) d.Dispose();
                    }
                });
        }
        catch (Exception ex)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }

        return 0;
    }

    private static string? ResolveZImageVae()
    {
        foreach (var c in new[] {
            "models/z-image-turbo/vae",
            "models/z-image-turbo/vae/diffusion_pytorch_model.safetensors",
            "models/ae.safetensors" })
            if (File.Exists(c) || Directory.Exists(c)) return c;
        return null;
    }

    private static string? ResolveZImageEncoder()
    {
        foreach (var c in new[] {
            "models/Z-Image-AbliteratedV1.Q5_K_M.gguf",
            "models/Z-Image-AbliteratedV1.Q4_K_M.gguf",
            "models/Z-Image-AbliteratedV1.Q8_0.gguf",
            "models/Z-Image-AbliteratedV1.F16.gguf" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveZImageTokenizer()
    {
        foreach (var c in new[] {
            "models/z-image-turbo/tokenizer/tokenizer.json",
            "models/tokenizer.json" })
            if (File.Exists(c)) return c;
        return null;
    }
}
