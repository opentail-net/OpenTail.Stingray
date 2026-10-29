namespace OpenTail.Stingray.Cli;

public sealed partial class ImageCommand
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to diffusion model GGUF or safetensors directory (FLUX.1, Z-Image-Turbo, …)")]
        public string? ModelPath { get; init; }

        [CommandOption("-p|--prompt")]
        [Description("Text prompt describing the image to generate")]
        public string? Prompt { get; init; }

        [CommandOption("--negative-prompt")]
        [Description("Negative prompt — what to avoid in the generated image")]
        public string? NegativePrompt { get; init; }

        [CommandOption("--vae")]
        [Description("Path to VAE safetensors file or directory (ae.safetensors or vae/ dir)")]
        public string? VaePath { get; init; }

        // ── Z-Image options ───────────────────────────────────────────────

        [CommandOption("--qwen-encoder")]
        [Description("(Z-Image) Path to Qwen3-4B GGUF text encoder (from Qwen/Qwen3-4B-GGUF)")]
        public string? QwenEncoderPath { get; init; }

        [CommandOption("--qwen-tokenizer")]
        [Description("(Z-Image) Path to Qwen3 tokenizer.json")]
        public string? QwenTokenizerPath { get; init; }

        // ── Wan options ──────────────────────────────────────────────────

        [CommandOption("--umt5-encoder")]
        [Description("(Wan) Path to the real UMT5-XXL text encoder safetensors (converted from Wan-AI/Wan2.1-T2V-1.3B's models_t5_umt5-xxl-enc-bf16.pth)")]
        public string? Umt5EncoderPath { get; init; }

        [CommandOption("--umt5-tokenizer")]
        [Description("(Wan) Path to the real UMT5 tokenizer.json (Wan-AI/Wan2.1-T2V-1.3B's google/umt5-xxl/tokenizer.json)")]
        public string? Umt5TokenizerPath { get; init; }

        [CommandOption("--ngl|--n-gpu-layers|--gpu-layers|-g")]
        [Description("(Z-Image) GPU acceleration: -1 = auto (CUDA→Vulkan→CPU, default), 0 = CPU only")]
        [DefaultValue(-1)]
        public int NGpuLayers { get; init; }

        [CommandOption("--device")]
        [Description("(Z-Image) GPU to use: auto (default), none/cpu, an index (0, 1), or a named device (CUDA0, Vulkan1)")]
        public string? Device { get; init; }

        [CommandOption("--backend")]
        [Description("(Z-Image) Force compute backend: auto (default), cuda, vulkan, cpu")]
        public string? Backend { get; init; }

        // ── FLUX options ──────────────────────────────────────────────────

        [CommandOption("--clip-l")]
        [Description("(FLUX) Path to CLIP-L encoder safetensors")]
        public string? ClipLPath { get; init; }

        [CommandOption("--clip-g")]
        [Description("(SD3/3.5) Path to OpenCLIP-bigG encoder safetensors -- when given together with --clip-l and --vae, -m is read as the standalone MMDiT transformer safetensors (the standard diffusers multi-file layout) instead of a single combined checkpoint")]
        public string? ClipGPath { get; init; }

        [CommandOption("--clip-tokenizer")]
        [Description("(FLUX) Path to CLIP tokenizer.json")]
        public string? ClipTokenizerPath { get; init; }

        [CommandOption("--t5xxl")]
        [Description("(FLUX) Path to T5-XXL encoder safetensors")]
        public string? T5XXLPath { get; init; }

        [CommandOption("--t5-tokenizer")]
        [Description("(FLUX) Path to T5 tokenizer.json")]
        public string? T5TokenizerPath { get; init; }

        // ── Common options ────────────────────────────────────────────────

        [CommandOption("-W|--width")]
        [Description("Output image width in pixels — must be divisible by 16 (default: 512)")]
        [DefaultValue(512)]
        public int Width { get; init; }

        [CommandOption("-H|--height")]
        [Description("Output image height in pixels — must be divisible by 16 (default: 512)")]
        [DefaultValue(512)]
        public int Height { get; init; }

        [CommandOption("--steps")]
        [Description("Denoising steps (default: 4 for Z-Image-Turbo, 4 for FLUX schnell, 20 for dev)")]
        [DefaultValue(0)]
        public int Steps { get; init; }

        [CommandOption("--cfg-scale")]
        [Description("Guidance scale — not used for Z-Image (distilled), 1.0 for FLUX schnell (default: auto)")]
        [DefaultValue(-1f)]
        public float CfgScale { get; init; }

        [CommandOption("-s|--seed")]
        [Description("RNG seed (-1 = random, default: -1)")]
        [DefaultValue(-1)]
        public int Seed { get; init; }

        [CommandOption("-o|--output")]
        [Description("Output PNG file path (default: output.png)")]
        public string? OutputPath { get; init; }

        [CommandOption("--upscaler")]
        [Description("Path to ESRGAN/Real-ESRGAN upscaler weights (.safetensors). Upscales the generated image by ×2 or ×4 before saving.")]
        public string? UpscalerPath { get; init; }

        [CommandOption("--upscale-blend")]
        [Description("Blend factor for the upscaled result (0.0–1.0). 1.0 = full RRDB (sharpest), <1.0 softens by blending with bicubic. Default 1.0.")]
        [DefaultValue(1.0f)]
        public float UpscaleBlend { get; init; } = 1.0f;

        [CommandOption("-v|--verbose")]
        [Description("Show per-step timing and progress")]
        [DefaultValue(false)]
        public bool Verbose { get; init; }

        [CommandOption("-i|--init-image|--init-img")]
        [Description("Path to initial image for image-to-image (img2img) generation")]
        public string? InitImagePath { get; init; }

        [CommandOption("--mask-image|--mask-img|--mask")]
        [Description("Path to inpainting mask image (white = inpaint region, black = preserve original)")]
        public string? MaskImagePath { get; init; }

        [CommandOption("--strength")]
        [Description("Strength for img2img generation (0.0 to 1.0, default: 0.75). Higher values add more variation from the initial image.")]
        [DefaultValue(0.75f)]
        public float Strength { get; init; } = 0.75f;

        [CommandOption("--sampler")]
        [Description("Diffusion scheduler algorithm: euler (default), euler-a, ddim, dpm2m, dpm2m-karras, lcm")]
        public string? Sampler { get; init; }

        [CommandOption("--control-net|--controlnet")]
        [Description("Path to ControlNet weights (.safetensors or .gguf)")]
        public string? ControlNetPath { get; init; }

        [CommandOption("--control-image|--control-img")]
        [Description("Path to ControlNet condition hint image (Canny edge, depth map, openpose, etc.)")]
        public string? ControlImagePath { get; init; }

        [CommandOption("--control-strength")]
        [Description("ControlNet conditioning scale/strength (default: 1.0)")]
        [DefaultValue(1.0f)]
        public float ControlStrength { get; init; } = 1.0f;

        [CommandOption("--video-frames|--frames")]
        [Description("Number of video frames to generate for Wan video diffusion models (default: 1)")]
        [DefaultValue(1)]
        public int VideoFrames { get; init; } = 1;

        // ── sd-cli fallback ───────────────────────────────────────────────

        [CommandOption("--use-sdcpp")]
        [Description("Delegate to stable-diffusion.cpp sd-cli instead of native pipeline (for comparison)")]
        [DefaultValue(false)]
        public bool UseSdCpp { get; init; }

        [CommandOption("--sd-cli")]
        [Description("Path to sd-cli executable used when --use-sdcpp is set (overrides STINGRAY_SDCPP env var)")]
        public string? SdCliPath { get; init; }

        [CommandOption("--text-encoder")]
        [Description("Path to the LLM-style text encoder GGUF (FLUX.2: Mistral-Small; also used by sd-cli mode)")]
        public string? TextEncoderPath { get; init; }
    }
}
