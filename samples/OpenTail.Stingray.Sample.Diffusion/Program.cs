using System.Diagnostics;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.StableDiffusion;

namespace OpenTail.Stingray.Sample.Diffusion;

internal static class Program
{
    public static int Main(string[] args)
    {
        Console.WriteLine("=================================================");
        Console.WriteLine(" OpenTail Stingray — Diffusion Generation Sample ");
        Console.WriteLine("=================================================");
        Console.WriteLine();

        var parsed = ParseArgs(args);
        if (parsed.HelpRequested)
        {
            PrintUsage();
            return 0;
        }

        // 1. Locate or validate the model checkpoint
        string? modelPath = parsed.ModelPath;
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            // Auto-resolve known common diffusion checkpoints
            if (!ModelHome.TryResolveModelPath("v1-5-pruned-emaonly.safetensors", out modelPath) &&
                !ModelHome.TryResolveModelPath("sd-v1-5.safetensors", out modelPath) &&
                !ModelHome.TryResolveModelPath("v1-5-pruned.safetensors", out modelPath) &&
                !ModelHome.TryResolveModelPath("model.safetensors", out modelPath))
            {
                modelPath = null;
            }
        }

        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("No diffusion model checkpoint found or specified.");
            Console.ResetColor();
            Console.WriteLine();
            Console.WriteLine("To run this sample with Stable Diffusion 1.5:");
            Console.WriteLine("  1. Download a SD 1.5 safetensors checkpoint (e.g. from HuggingFace):");
            Console.WriteLine("     curl -L -o models/v1-5-pruned-emaonly.safetensors https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors");
            Console.WriteLine("  2. Download the CLIP tokenizer (if not already present):");
            Console.WriteLine("     curl -L -o models/clip_tokenizer.json https://huggingface.co/openai/clip-vit-base-patch32/raw/main/tokenizer.json");
            Console.WriteLine("  3. Run this sample:");
            Console.WriteLine("     dotnet run --project samples/OpenTail.Stingray.Sample.Diffusion -- --prompt \"A serene mountain lake at sunrise\"");
            Console.WriteLine();
            Console.WriteLine("Or specify a model file directly via --model <path>.");
            return 1;
        }

        Console.WriteLine($"Model checkpoint : {modelPath}");
        Console.WriteLine($"Prompt           : \"{parsed.Prompt}\"");
        Console.WriteLine($"Image Size       : {parsed.Width}x{parsed.Height}");
        Console.WriteLine($"Denoise Steps    : {parsed.Steps}");
        Console.WriteLine($"Guidance Scale   : {parsed.Guidance:F1}");
        Console.WriteLine($"RNG Seed         : {(parsed.Seed >= 0 ? parsed.Seed.ToString() : "random")}");
        Console.WriteLine($"Output File      : {parsed.OutputPath}");
        Console.WriteLine();

        // 2. Hardware backend selection
        var (backend, ownsBackend) = DiffusionBackendResolver.Resolve(explicitBackend: null);
        string backendName = backend != null ? backend.GetType().Name : "CPU (SIMD)";
        Console.WriteLine($"Hardware Backend : {backendName}");
        Console.WriteLine();

        try
        {
            var sw = Stopwatch.StartNew();
            Console.Write("Loading diffusion pipeline... ");

            // In OpenTail Stingray, diffusion pipelines implement IDiffusionPipeline
            using IDiffusionPipeline pipeline = StableDiffusionPipeline.Load(modelPath, parsed.TokenizerPath, backend);
            long loadMs = sw.ElapsedMilliseconds;
            Console.WriteLine($"Done ({loadMs} ms)");

            Console.WriteLine();
            Console.WriteLine("Generating image...");
            sw.Restart();

            var request = new ImageGenerationRequest
            {
                Prompt = parsed.Prompt,
                Width = parsed.Width,
                Height = parsed.Height,
                Steps = parsed.Steps,
                Guidance = parsed.Guidance,
                Seed = parsed.Seed,
                OutputPath = parsed.OutputPath,
                Progress = (step, total) =>
                {
                    int pct = total > 0 ? (step * 100) / total : 0;
                    Console.Write($"\r[Denoising] Step {step}/{total} ({pct}%)... ");
                }
            };

            pipeline.Generate(request);
            sw.Stop();
            Console.WriteLine();
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Success] Image written to: {Path.GetFullPath(parsed.OutputPath)}");
            Console.ResetColor();
            Console.WriteLine($"Denoising Latency: {sw.ElapsedMilliseconds} ms ({(double)sw.ElapsedMilliseconds / parsed.Steps:F1} ms/step)");

            return 0;
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[Error] Generation failed: {ex.Message}");
            Console.ResetColor();
            return 1;
        }
        finally
        {
            if (ownsBackend && backend is IDisposable dispBackend)
            {
                dispBackend.Dispose();
            }
        }
    }

    private static Options ParseArgs(string[] args)
    {
        var opt = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a is "--help" or "-h" or "-?")
            {
                opt.HelpRequested = true;
            }
            else if ((a is "--model" or "-m") && i + 1 < args.Length)
            {
                opt.ModelPath = args[++i];
            }
            else if ((a is "--prompt" or "-p") && i + 1 < args.Length)
            {
                opt.Prompt = args[++i];
            }
            else if ((a is "--output" or "-o") && i + 1 < args.Length)
            {
                opt.OutputPath = args[++i];
            }
            else if (a == "--tokenizer" && i + 1 < args.Length)
            {
                opt.TokenizerPath = args[++i];
            }
            else if (a == "--steps" && i + 1 < args.Length && int.TryParse(args[++i], out int steps))
            {
                opt.Steps = steps;
            }
            else if (a == "--guidance" && i + 1 < args.Length && float.TryParse(args[++i], out float g))
            {
                opt.Guidance = g;
            }
            else if (a == "--seed" && i + 1 < args.Length && int.TryParse(args[++i], out int seed))
            {
                opt.Seed = seed;
            }
            else if (a == "--width" && i + 1 < args.Length && int.TryParse(args[++i], out int w))
            {
                opt.Width = w;
            }
            else if (a == "--height" && i + 1 < args.Length && int.TryParse(args[++i], out int h))
            {
                opt.Height = h;
            }
            else if (!a.StartsWith("-") && opt.Prompt == DefaultPrompt)
            {
                opt.Prompt = a;
            }
        }
        return opt;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Usage: dotnet run --project samples/OpenTail.Stingray.Sample.Diffusion [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -m, --model <path>      Path to model checkpoint (.safetensors or .gguf)");
        Console.WriteLine("  -p, --prompt <text>     Text prompt describing desired image");
        Console.WriteLine("  -o, --output <path>     Output PNG image path (default: output.png)");
        Console.WriteLine("      --tokenizer <path>  Path to clip_tokenizer.json (if not in models/)");
        Console.WriteLine("      --steps <int>       Number of denoising steps (default: 20)");
        Console.WriteLine("      --guidance <float>  Classifier-free guidance scale (default: 7.5)");
        Console.WriteLine("      --seed <int>        RNG seed for reproducibility (-1 for random)");
        Console.WriteLine("      --width <int>       Image width in pixels (divisible by 8, default: 512)");
        Console.WriteLine("      --height <int>      Image height in pixels (divisible by 8, default: 512)");
        Console.WriteLine("  -h, --help              Show this help message");
    }

    private const string DefaultPrompt = "A serene alpine mountain lake at sunrise, highly detailed digital art";

    private sealed class Options
    {
        public bool HelpRequested { get; set; }
        public string? ModelPath { get; set; }
        public string? TokenizerPath { get; set; }
        public string Prompt { get; set; } = DefaultPrompt;
        public string OutputPath { get; set; } = "output.png";
        public int Steps { get; set; } = 20;
        public float Guidance { get; set; } = 7.5f;
        public int Seed { get; set; } = -1;
        public int Width { get; set; } = 512;
        public int Height { get; set; } = 512;
    }
}
