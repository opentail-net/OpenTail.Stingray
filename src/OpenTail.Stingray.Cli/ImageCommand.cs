using OpenTail.Stingray.Diffusion.StableDiffusion;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Image generation command. Supports two native pipelines:
///
///   Z-Image-Turbo (auto-detected when model path contains "z_image" or "zimage"):
///     opentail-llm image -m models/z_image_turbo-Q5_K_M.gguf --vae vae/
///                  --qwen-encoder Z-Image-AbliteratedV1.Q5_K_M.gguf
///                  --qwen-tokenizer tokenizer.json -p "a cat" -W 1024 -H 1024 -o out.png
///
///   Recommended models (jayn7/Z-Image-Turbo-GGUF  +  BennyDaBall abliterated encoder):
///     DiT Q5_K_M: z_image_turbo-Q5_K_M.gguf  (5.52 GB, highest quality)
///     DiT Q4_K_M: z_image_turbo-Q4_K_M.gguf  (4.50 GB, good quality, faster)
///     Encoder:    Z-Image-AbliteratedV1.Q5_K_M.gguf  (2.89 GB, uncensored)
///
///   FLUX.1-schnell / FLUX.1-dev:
///     opentail-llm image -m flux1-schnell-q4_k.gguf --vae ae.safetensors
///                  --clip-l clip_l.safetensors --clip-tokenizer tokenizer_clip.json
///                  --t5xxl t5xxl_fp16.safetensors --t5-tokenizer tokenizer_t5.json
///                  -p "a cinematic photograph of a cat" -W 512 -H 512 -o out.png
///
/// Model downloads:
///   Z-Image-Turbo GGUF (Q5_K_M + Q4_K_M): https://huggingface.co/jayn7/Z-Image-Turbo-GGUF
///   Z-Image-Turbo full:                    https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
///   Text encoder (uncensored Q4/Q5):       https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1
///   Qwen3-4B GGUF:                         https://huggingface.co/Qwen/Qwen3-4B-GGUF
///   FLUX.1-schnell GGUF:                   https://huggingface.co/city96/FLUX.1-schnell-gguf
///   VAE + CLIP + T5:                       https://huggingface.co/comfyanonymous/flux_text_encoders
/// </summary>
public sealed partial class ImageCommand : Command<ImageCommand.Settings>
{

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (settings.UseSdCpp)
            return RunSdCpp(settings);

        return RunNative(settings);
    }

    // ── Native pipeline ───────────────────────────────────────────────────

    private static int RunNative(Settings s)
    {
        var modelPath = ResolveModelPath(s.ModelPath);
        if (modelPath is null)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] No diffusion model found. Use [yellow]-m <path>[/]");
            PrintModelDownloadHint();
            return 1;
        }

        if (string.IsNullOrWhiteSpace(s.Prompt))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] Prompt required. Use [yellow]-p \"your prompt\"[/]");
            return 1;
        }

        int deviceIndex;
        bool deviceNone;
        try { deviceIndex = GpuDevice.Resolve(s.Device, out deviceNone); }
        catch (InvalidOperationException ex)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }

        if (IsFlux2(modelPath))
            return RunFlux2(s, modelPath, deviceIndex, deviceNone);
        if (IsQwenImage(modelPath))
            return RunQwenImage(s, modelPath, deviceIndex, deviceNone);
        if (IsZImage(modelPath))
            return RunZImage(s, modelPath, deviceIndex, deviceNone);
        if (IsSd3(modelPath))
            return RunSd3(s, modelPath, deviceIndex, deviceNone);
        if (IsSdxl(modelPath))
            return RunSdxl(s, modelPath, deviceIndex, deviceNone);
        if (IsWan(modelPath))
            return RunWan(s, modelPath, deviceIndex, deviceNone);
        if (IsHunyuanVideo(modelPath))
            return RunHunyuanVideo(s, modelPath, deviceIndex, deviceNone);
        if (IsLtxVideo(modelPath))
            return RunLtxVideo(s, modelPath, deviceIndex, deviceNone);
        if (IsStableDiffusion(modelPath))
            return RunStableDiffusion(s, modelPath, deviceIndex, deviceNone);
        return RunFlux(s, modelPath, deviceIndex, deviceNone);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static bool RequireFile(string? path, string flag, string example)
    {
        if (!string.IsNullOrWhiteSpace(path) && File.Exists(path)) return true;
        AnsiConsole.ErrorLine($"[red]Error:[/] Missing [yellow]{flag} <path>[/] (e.g. [cyan]{example}[/])");
        return false;
    }

    private static bool RequirePathExists(string? path, string flag, string example)
    {
        if (!string.IsNullOrWhiteSpace(path) && (File.Exists(path) || Directory.Exists(path))) return true;
        AnsiConsole.ErrorLine($"[red]Error:[/] Missing [yellow]{flag} <path>[/] (e.g. [cyan]{example}[/])");
        return false;
    }

    private static string? ResolveModelPath(string? given)
    {
        if (given is not null) return (File.Exists(given) || Directory.Exists(given)) ? given : null;
        foreach (var c in new[] {
            "models/z_image_turbo-Q5_K_M.gguf", "models/z_image_turbo-Q4_K_M.gguf", "models/z_image_turbo-Q8_0.gguf",
            "models/flux1-schnell/flux1-schnell-Q4_K_S.gguf",
            "models/flux1-schnell/flux1-schnell-q4_k.gguf",
            "models/flux1-schnell-q4_k.gguf",   "models/flux1-dev-q4_k.gguf" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static void PrintModelDownloadHint()
    {
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("Z-Image-Turbo GGUF (Q5_K_M + Q4_K_M):");
        AnsiConsole.MarkupLine("  [link]https://huggingface.co/jayn7/Z-Image-Turbo-GGUF[/]");
        AnsiConsole.MarkupLine("  z_image_turbo-Q5_K_M.gguf  (5.52 GB, best quality)");
        AnsiConsole.MarkupLine("  z_image_turbo-Q4_K_M.gguf  (4.50 GB, good quality, faster dequant)");
        AnsiConsole.MarkupLine("Text encoder (uncensored):");
        AnsiConsole.MarkupLine("  [link]https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1[/]");
        AnsiConsole.MarkupLine("  Z-Image-AbliteratedV1.Q5_K_M.gguf or Q4_K_M.gguf");
        AnsiConsole.MarkupLine("FLUX.1-schnell GGUF: [link]https://huggingface.co/city96/FLUX.1-schnell-gguf[/]");
    }



    private static bool IsHunyuanVideo(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("hunyuan") || n.Contains("hyvideo");
    }

    private static bool IsWan(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("wan") || n.Contains("wan2");
    }

    private static bool IsFlux2(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("flux2") || n.Contains("flux.2") || n.Contains("flux-2");
    }

    private static bool IsQwenImage(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return (n.Contains("qwen") && (n.Contains("image") || n.Contains("img"))) ||
               n.Contains("qwen_image") || n.Contains("qwen-image");
    }

    private static bool IsSd3(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("sd3") || n.Contains("sd35") || n.Contains("sd-3") || n.Contains("sd_3") || n.Contains("mmdit");
    }
    private static bool IsSdxl(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        if (n.Contains("sdxl") || n.Contains("sd_xl") || n.Contains("ssd-1b") || n.Contains("ssd1b") || 
            n.Contains("segmind") || n.Contains("sd-xl") || n.Contains("ponyxl") || n.Contains("animagine") || 
            n.Contains("juggernautxl") || n.Contains("realvisxl"))
            return true;

        if (File.Exists(path) && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                using var loader = SafetensorsLoader.Open(path);
                // Real, SDXL-EXCLUSIVE signal only: the second (OpenCLIP-G) text encoder.
                // input_blocks.4.1...attn2.to_k.weight was ALSO checked here before, but that
                // tensor exists in a standard SD1.5 UNet too (block 4 has cross-attention there
                // as well) -- confirmed directly against a real v1-5-pruned-emaonly.safetensors,
                // which caused every SD1.5 checkpoint to be misdetected as SDXL and fail to load
                // (SDXL's text-encoder tensor names don't exist in an SD1.5 file). Removed.
                if (loader.Contains("conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"))
                    return true;
            }
            catch { }
        }
        return false;
    }
    private static bool IsLtxVideo(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("ltx") || n.Contains("ltxv");
    }

    private static bool IsStableDiffusion(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        if (n.Contains("v1-5") || n.Contains("sd-v1") || n.Contains("sd15") || n.Contains("stable-diffusion") || 
            n.Contains("v1_5") || n.Contains("v1.5") || n.Contains("sd1.5") || n.Contains("realistic") || 
            n.Contains("dreamshaper") || n.Contains("cyberrealistic") || n.Contains("epicrealism") || 
            n.Contains("deliberate") || n.Contains("revanimated") || n.Contains("chillout") || 
            n.Contains("counterfeit") || n.Contains("meina") || n.Contains("anything") || n.Contains("abyss"))
            return true;

        if (File.Exists(path) && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                using var loader = SafetensorsLoader.Open(path);
                if (loader.Contains("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight") ||
                    loader.Contains("diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight") ||
                    loader.Contains("cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight"))
                    return true;
            }
            catch { }
        }

        return !IsSdxl(path) && !IsSd3(path) && !IsFlowMatching(path) && !IsZImage(path) && !path.ToLowerInvariant().Contains("wan") && !IsHunyuanVideo(path) && !IsLtxVideo(path);
    }
    private static bool IsZImage(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        if (n.Contains("z_image") || n.Contains("zimage") || n.Contains("z-image")) return true;
        // Also check if it's a directory named "transformer" with z-image sibling vae
        if (Directory.Exists(path))
        {
            string parent = Path.GetDirectoryName(Path.GetFullPath(path)) ?? "";
            string lp     = parent.ToLowerInvariant();
            return lp.Contains("z_image") || lp.Contains("zimage") || lp.Contains("z-image");
        }
        return false;
    }

    private static bool IsDistilled(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("schnell") || n.Contains("turbo") || n.Contains("lcm");
    }

    private static bool IsFlowMatching(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("flux") || n.Contains("z_image") || n.Contains("z-image");
    }

    private static DiffusionSchedulerType ParseScheduler(string? sampler)
    {
        if (string.IsNullOrWhiteSpace(sampler)) return DiffusionSchedulerType.Euler;
        string s = sampler.Trim().ToLowerInvariant().Replace("_", "").Replace("-", "").Replace("+", "plus");
        return s switch
        {
            "eulera" or "eulerancestral" => DiffusionSchedulerType.EulerAncestral,
            "ddim" => DiffusionSchedulerType.Ddim,
            "dpm2m" or "dpmplusplus2m" or "dpmpp2m" => DiffusionSchedulerType.DpmPlusPlus2M,
            "dpm2mkarras" or "dpmplusplus2mkarras" or "dpmpp2mkarras" or "karras" => DiffusionSchedulerType.DpmPlusPlus2MKarras,
            "lcm" => DiffusionSchedulerType.Lcm,
            _ => DiffusionSchedulerType.Euler
        };
    }

    private static float[]? LoadInitImage(string? path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return null;
        byte[] raw = OpenTail.Stingray.Vision.ImageIO.LoadRgb(path, out int w, out int h);
        var planar = new float[3 * h * w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int src = (y * w + x) * 3;
                int dst = y * w + x;
                planar[0 * (h * w) + dst] = raw[src + 0] / 255.0f;
                planar[1 * (h * w) + dst] = raw[src + 1] / 255.0f;
                planar[2 * (h * w) + dst] = raw[src + 2] / 255.0f;
            }
        }
        return planar;
    }
}


















