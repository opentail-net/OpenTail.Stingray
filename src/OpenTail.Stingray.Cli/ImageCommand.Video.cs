using OpenTail.Stingray.Diffusion;

namespace OpenTail.Stingray.Cli;

public sealed partial class ImageCommand
{
    private static int RunHunyuanVideo(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string output = s.OutputPath ?? "output.png";
        int steps = s.Steps > 0 ? s.Steps : 20;
        float guidance = s.CfgScale >= 0 ? s.CfgScale : 6.0f;
        int frames = s.VideoFrames > 0 ? s.VideoFrames : 1;
        float[]? initImage = LoadInitImage(s.InitImagePath);

        IComputeBackend? gpu = null;
        if (!deviceNone && deviceIndex >= 0)
        {
            try { gpu = new VulkanBackend(deviceIndex); }
            catch (Exception ex) { AnsiConsole.MarkupLine($"[yellow]Note:[/] Vulkan GPU init failed ({Markup.Escape(ex.Message)}); falling back to CPU."); }
        }

        AnsiConsole.MarkupLine("[bold]HunyuanVideo Video Diffusion (DiT + Dual Text Conditioning)[/]");
        AnsiConsole.MarkupLine($"[dim]Model:[/]    {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {s.Width}×{s.Height} (frames={frames})  steps={steps}  guidance={guidance}  seed={s.Seed}");
        if (initImage is not null)
            AnsiConsole.MarkupLine($"[dim]Init Image:[/] {Markup.Escape(s.InitImagePath!)} (I2V mode)");
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

            using var pipeline = OpenTail.Stingray.Diffusion.HunyuanVideo.HunyuanVideoPipeline.Load(modelPath, s.VaePath, gpu);

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
                        numFrames: frames,
                        steps: steps,
                        guidance: guidance,
                        flowShift: 7.0f,
                        seed: s.Seed,
                        outputPath: output,
                        progress: (step, total) => ctx.Status($"Denoising step {step}/{total} on {target}…"),
                        upscaler: upscaler,
                        upscaleBlend: s.UpscaleBlend,
                        initImageRgb: initImage);
                });

            sw.Stop();
            AnsiConsole.MarkupLine($"[green]✓[/] Output saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/] in [yellow]{sw.Elapsed.TotalSeconds:F1}s[/]");
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

    private static int RunWan(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string output = s.OutputPath ?? "output.png";
        int steps = s.Steps > 0 ? s.Steps : 20;
        float guidance = s.CfgScale >= 0 ? s.CfgScale : 6.0f;
        int frames = s.VideoFrames > 0 ? s.VideoFrames : 1;

        string? modelDir = Path.GetDirectoryName(Path.GetFullPath(modelPath));
        string? umt5EncoderPath = s.Umt5EncoderPath ?? ResolveWanUmt5Encoder(modelDir);
        string? umt5TokenizerPath = s.Umt5TokenizerPath ?? ResolveWanUmt5Tokenizer(modelDir);
        string? vaePath = s.VaePath ?? ResolveWanVae(modelDir);
        if (vaePath is null)
            AnsiConsole.MarkupLine("[yellow]Note:[/] No real Wan VAE found (--vae) -- decode will fail with a missing-tensor error once denoising completes.");

        IComputeBackend? gpu = null;
        if (!deviceNone && deviceIndex >= 0)
        {
            try { gpu = new VulkanBackend(deviceIndex); }
            catch (Exception ex) { AnsiConsole.MarkupLine($"[yellow]Note:[/] Vulkan GPU init failed ({Markup.Escape(ex.Message)}); falling back to CPU."); }
        }

        AnsiConsole.MarkupLine("[bold]Wan 2.1 / 2.2 Video Diffusion (DiT + UMT5)[/]");
        AnsiConsole.MarkupLine($"[dim]Model:[/]    {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {s.Width}×{s.Height} (frames={frames})  steps={steps}  guidance={guidance}  seed={s.Seed}");
        if (umt5EncoderPath is not null && umt5TokenizerPath is not null)
            AnsiConsole.MarkupLine($"[dim]UMT5:[/]     {Markup.Escape(umt5EncoderPath)}");
        else
            AnsiConsole.MarkupLine("[yellow]Note:[/] No real UMT5 encoder/tokenizer found (--umt5-encoder/--umt5-tokenizer) -- text conditioning will be all-zero (unconditional generation only).");
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

            float[]? condContext = null, uncondContext = null;
            if (umt5EncoderPath is not null && umt5TokenizerPath is not null)
            {
                AnsiConsole.MarkupLine("[dim]Encoding prompt with real UMT5-XXL…[/]");
                var tokenizer = OpenTail.Stingray.Diffusion.TextEncoders.T5Tokenizer.FromFile(umt5TokenizerPath, maxLen: 512);
                using var umt5 = new OpenTail.Stingray.Diffusion.TextEncoders.UMT5Encoder(umt5EncoderPath);

                // Real diffusers WanPipeline._get_t5_prompt_embeds: encode the REAL (unpadded)
                // tokens, then zero-pad the EMBEDDINGS (not the token ids) up to a fixed
                // max_sequence_length=226 for cross-attention -- this is ZERO padding of the
                // embedding, NOT running the encoder on pad_token_id=0 (which would produce real,
                // nonzero "pad" embeddings that don't match the reference). Found 2026-09-14 during
                // the Priority-0 Wan accuracy investigation (docs/081) -- this pipeline previously
                // fed cross-attention only the raw ~N-token context with no fixed-length padding at
                // all, a real, confirmed mismatch vs. the reference (measurably changes denoising
                // convergence: final latent std dropped from ~1.0-2.0 to ~0.4-0.7 with this fix).
                const int fixedTxtLen = 226;
                const int txtDim = 4096;
                static float[] ZeroPadEmbedding(float[] raw)
                {
                    var padded = new float[fixedTxtLen * txtDim];
                    Array.Copy(raw, padded, Math.Min(raw.Length, fixedTxtLen * txtDim));
                    return padded;
                }

                if (gpu is IVisionOpsBackend vBackend)
                {
                    // One pass over the 24 streamed layers for both prompts (EncodePairGpu), not
                    // two full EncodeGpu passes that each re-read and re-upload every layer.
                    var (condRaw, uncondRaw) = umt5.EncodePairGpu(
                        tokenizer.Tokenize(s.Prompt!), tokenizer.Tokenize(s.NegativePrompt ?? ""), vBackend);
                    condContext = ZeroPadEmbedding(condRaw);
                    uncondContext = ZeroPadEmbedding(uncondRaw);
                }
                else
                {
                    condContext = ZeroPadEmbedding(umt5.Encode(tokenizer.Tokenize(s.Prompt!)));
                    uncondContext = ZeroPadEmbedding(umt5.Encode(tokenizer.Tokenize(s.NegativePrompt ?? "")));
                }
                Console.Error.WriteLine($"[WanPipeline Profile] UMT5 encode (cond+uncond) took {sw.ElapsedMilliseconds} ms");
            }

            using var pipeline = OpenTail.Stingray.Diffusion.Wan.WanPipeline.Load(modelPath, vaePath, gpu);

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
                        numFrames: frames,
                        steps: steps,
                        guidance: guidance,
                        flowShift: 3.0f,
                        seed: s.Seed,
                        outputPath: output,
                        progress: (step, total) => ctx.Status($"Denoising step {step}/{total} on {target}…"),
                        upscaler: upscaler,
                        upscaleBlend: s.UpscaleBlend,
                        textContext: condContext,
                        negativeTextContext: uncondContext);
                });

            sw.Stop();
            AnsiConsole.MarkupLine($"[green]✓[/] Output saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/] in [yellow]{sw.Elapsed.TotalSeconds:F1}s[/]");
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

    private static string? ResolveWanUmt5Encoder(string? modelDir)
    {
        foreach (var c in new[] { Path.Combine(modelDir ?? ".", "wan2.1", "models_t5_umt5-xxl-enc-bf16.safetensors"), "models/wan2.1/models_t5_umt5-xxl-enc-bf16.safetensors" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveWanUmt5Tokenizer(string? modelDir)
    {
        foreach (var c in new[] { Path.Combine(modelDir ?? ".", "wan2.1", "umt5-tokenizer.json"), "models/wan2.1/umt5-tokenizer.json" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveWanVae(string? modelDir)
    {
        foreach (var c in new[] { Path.Combine(modelDir ?? ".", "wan2.1", "Wan2.1_VAE.safetensors"), "models/wan2.1/Wan2.1_VAE.safetensors" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static int RunLtxVideo(Settings s, string modelPath, int deviceIndex, bool deviceNone)
    {
        string output = s.OutputPath ?? "ltx_video_output.png";
        int width = s.Width > 0 ? s.Width : 768;
        int height = s.Height > 0 ? s.Height : 512;
        int steps = s.Steps > 0 ? s.Steps : 25;
        float cfgScale = s.CfgScale >= 0 ? s.CfgScale : 3.0f;
        int videoFrames = s.VideoFrames > 0 ? s.VideoFrames : 25;

        AnsiConsole.MarkupLine("[bold]LTX-Video[/] (Lightricks Video DiT)");
        AnsiConsole.MarkupLine($"[dim]Model:[/]   {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]Prompt:[/]  \"{Markup.Escape(s.Prompt!)}\"");
        AnsiConsole.MarkupLine($"[dim]Size:[/]    {width}x{height}, {videoFrames} frames, {steps} steps, CFG {cfgScale:F1}");

        // Auto-detect a `<models>/ltx-t5/text_encoder` + `tokenizer/tokenizer.json` pair sitting
        // alongside the model directory (this project's own local download convention -- see
        // docs/done/055-ltx-video-implementation-plan.md step 6) for real T5-v1.1-XXL text conditioning;
        // falls back to placeholder conditioning if absent.
        string? modelsDir = Path.GetDirectoryName(Path.GetFullPath(modelPath));
        string? textEncoderDir = modelsDir is null ? null : Path.Combine(modelsDir, "ltx-t5", "text_encoder");
        string? tokenizerJsonPath = modelsDir is null ? null : Path.Combine(modelsDir, "ltx-t5", "tokenizer", "tokenizer.json");
        if (textEncoderDir is not null && !Directory.Exists(textEncoderDir)) textEncoderDir = null;
        if (tokenizerJsonPath is not null && !File.Exists(tokenizerJsonPath)) tokenizerJsonPath = null;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var pipeline = OpenTail.Stingray.Diffusion.LTXVideo.LtxVideoPipeline.Load(
            modelPath, textEncoderDir: textEncoderDir, tokenizerJsonPath: tokenizerJsonPath);

        var req = new OpenTail.Stingray.Diffusion.ImageGenerationRequest
        {
            Prompt = s.Prompt!,
            Width = width,
            Height = height,
            Steps = steps,
            Guidance = cfgScale,
            Seed = s.Seed,
            OutputPath = output
        };

        pipeline.Generate(req);
        sw.Stop();

        AnsiConsole.MarkupLine($"[green]✓[/] Primary frame saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/] in [yellow]{sw.Elapsed.TotalSeconds:F1}s[/]");
        return 0;
    }
}
