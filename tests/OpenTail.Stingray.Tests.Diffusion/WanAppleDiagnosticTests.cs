using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.TextEncoders;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanAppleDiagnosticTests
{
    private readonly ITestOutputHelper _output;

    public WanAppleDiagnosticTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static string FindModelPath(string relativePath)
    {
        var candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", relativePath),
            Path.Combine("..", "..", "..", relativePath),
            relativePath,
            Path.Combine(AppContext.BaseDirectory, relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath)
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return relativePath;
    }

    private void Log(string msg)
    {
        _output.WriteLine(msg);
        Console.WriteLine(msg);
    }

    [Fact]
    public void Stage1_Tokenizer_InspectTokens()
    {
        string tokPath = FindModelPath(Path.Combine("models", "wan2.1", "umt5-tokenizer.json"));
        if (!File.Exists(tokPath))
        {
            Log($"Tokenizer not found at {tokPath}");
            return;
        }

        var tokenizer = T5Tokenizer.FromFile(tokPath, maxLen: 512);
        string prompt = "a red apple on a wooden table, photorealistic";
        var tokens = tokenizer.Tokenize(prompt);

        Log($"[Stage 1: Tokenizer] Prompt: \"{prompt}\"");
        Log($"[Stage 1: Tokenizer] Token count: {tokens.Length}");
        Log($"[Stage 1: Tokenizer] Tokens: [{string.Join(", ", tokens)}]");

        int unkCount = tokens.Count(t => t == 2);
        Log($"[Stage 1: Tokenizer] UNK count: {unkCount} / {tokens.Length}");
        Assert.True(tokens.Length > 2, "Token sequence too short");
        Assert.True(unkCount < tokens.Length / 2, "Too many UNK tokens in prompt");
    }

    [Fact]
    public void Stage1To7_FullStageByStagePipelineDiagnostic()
    {
        string tokPath = FindModelPath(Path.Combine("models", "wan2.1", "umt5-tokenizer.json"));
        string encPath = FindModelPath(Path.Combine("models", "wan2.1", "models_t5_umt5-xxl-enc-bf16.safetensors"));
        string ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));
        string vaePath = FindModelPath(Path.Combine("models", "wan2.1", "Wan2.1_VAE.safetensors"));

        if (!File.Exists(tokPath) || !File.Exists(encPath) || !File.Exists(ditPath) || !File.Exists(vaePath))
        {
            Log("One or more Wan models missing, skipping full diagnostic test.");
            return;
        }

        using var vulkan = new VulkanBackend();
        Log($"[Diagnostic] Compute device: {vulkan.Name}");

        // STAGE 1: Tokenizer
        var tokenizer = T5Tokenizer.FromFile(tokPath, maxLen: 512);
        string prompt = "a red apple on a wooden table, photorealistic";
        var rawTokens = tokenizer.Tokenize(prompt);
        Log($"[Stage 1: Tokenizer] Raw tokens ({rawTokens.Length}): [{string.Join(", ", rawTokens)}]");

        int[] tokens = rawTokens;
        Log($"[Stage 1: Tokenizer] Tokens ({tokens.Length}): [{string.Join(", ", tokens.Take(30))}{(tokens.Length > 30 ? ", ..." : "")}]");

        // STAGE 2: Text Encoder (UMT5 GPU)
        using var umt5 = new UMT5Encoder(encPath);
        var rawContext = umt5.EncodeGpu(tokens, vulkan);
        int txtDim = 4096;

        // Real diffusers WanPipeline._get_t5_prompt_embeds: encodes the REAL (unpadded) tokens,
        // then zero-pads the EMBEDDINGS (not the token ids) up to a fixed max_sequence_length=226
        // for cross-attention. Critically this is ZERO padding of the embedding, NOT running the
        // encoder on pad_token_id=0 (which would produce real, nonzero encoded "pad" embeddings
        // that don't match the reference at all) -- our pipeline previously used only the raw
        // ~12-token context with no fixed-length padding whatsoever, a real, unverified mismatch
        // vs. the reference (docs/081, 2026-09-14 update #9).
        bool zeroPadTo226 = Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_ZEROPAD226") == "1";
        float[] condContext;
        int numTxtTokens;
        if (zeroPadTo226)
        {
            const int fixedLen = 226;
            numTxtTokens = fixedLen;
            condContext = new float[fixedLen * txtDim];
            Array.Copy(rawContext, condContext, Math.Min(rawContext.Length, fixedLen * txtDim));
            Log($"[Stage 2: UMT5 Context] Zero-padded embeddings from {tokens.Length} real tokens to fixed length {fixedLen}");
        }
        else
        {
            condContext = rawContext;
            numTxtTokens = tokens.Length;
        }
        float mean = 0, sumSq = 0;
        for (int i = 0; i < condContext.Length; i++) { mean += condContext[i]; sumSq += condContext[i] * condContext[i]; }
        mean /= condContext.Length;
        float std = MathF.Sqrt(sumSq / condContext.Length - mean * mean);
        Log($"[Stage 2: UMT5 Context] Length={condContext.Length} ({numTxtTokens}x{txtDim}), mean={mean:F4}, std={std:F4}");

        // STAGE 3: DiT Load & Cross KV Cache
        using var pipeline = WanPipeline.Load(ditPath, vaePath, vulkan);
        Log($"[Stage 3: Pipeline] Pipeline loaded successfully.");

        // STAGE 4: Run single-step forward pass
        int width = 256, height = 256;
        int latH = height / 8; // 32
        int latW = width / 8;  // 32
        int patchH = latH / 2; // 16
        int patchW = latW / 2; // 16
        int numTokens = 1 * patchH * patchW; // 256 tokens

        Log($"[Stage 4: Latent Config] Width={width}, Height={height}, latH={latH}, latW={latW}, numTokens={numTokens}");

        // Generate 4-step image and inspect latent trajectory
        float dbgGuidance = float.TryParse(Environment.GetEnvironmentVariable("STINGRAY_WAN_DEBUG_GUIDANCE"), out var gVal) ? gVal : 6.0f;
        var frames = pipeline.Generate(
            prompt: prompt,
            width: width,
            height: height,
            numFrames: 1,
            steps: 4,
            guidance: dbgGuidance,
            flowShift: 3.0f,
            seed: 42,
            outputPath: "wan_diagnostic_apple.png",
            progress: (step, total) => Log($"[Denoise Step {step}/{total}]"),
            textContext: condContext);

        Log($"[Stage 7: VAE Decode] Generated frame count: {frames.Count}, frame[0] size: {frames[0].Length}");
        float rMean = 0, gMean = 0, bMean = 0;
        int pixelCount = width * height;
        for (int i = 0; i < pixelCount; i++)
        {
            rMean += frames[0][0 * pixelCount + i];
            gMean += frames[0][1 * pixelCount + i];
            bMean += frames[0][2 * pixelCount + i];
        }
        rMean /= pixelCount; gMean /= pixelCount; bMean /= pixelCount;
        Log($"[Stage 7: VAE Output RGB Means] R={rMean:F4}, G={gMean:F4}, B={bMean:F4}");
    }

    /// <summary>
    /// Single real CPU forward pass (30 blocks, real weights) with per-block activation stat
    /// dumps (set STINGRAY_WAN_DEBUG_PERBLOCK=1) -- a real numeric bisection to localize which
    /// block the "pure noise output" bug actually originates in, instead of further guessing
    /// against the reference. See docs/081, Priority 0, "2026-09-14 update #5".
    /// </summary>
    [Fact]
    public void Stage_CpuForward_PerBlockActivationBisection()
    {
        string tokPath = FindModelPath(Path.Combine("models", "wan2.1", "umt5-tokenizer.json"));
        string encPath = FindModelPath(Path.Combine("models", "wan2.1", "models_t5_umt5-xxl-enc-bf16.safetensors"));
        string ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));

        if (!File.Exists(tokPath) || !File.Exists(encPath) || !File.Exists(ditPath))
        {
            Log("One or more Wan models missing, skipping per-block bisection test.");
            return;
        }

        Environment.SetEnvironmentVariable("STINGRAY_WAN_DEBUG_PERBLOCK", "1");

        using var vulkan = new VulkanBackend();
        var tokenizer = T5Tokenizer.FromFile(tokPath, maxLen: 512);
        var tokens = tokenizer.Tokenize("a red apple on a wooden table, photorealistic");

        using var umt5 = new UMT5Encoder(encPath);
        var rawCond = umt5.EncodeGpu(tokens, vulkan);
        const int fixedTxtLen = 226, txtDim = 4096;
        var condContext = new float[fixedTxtLen * txtDim];
        Array.Copy(rawCond, condContext, Math.Min(rawCond.Length, fixedTxtLen * txtDim));

        var ditLoader = SafetensorsLoader.Open(ditPath);
        using var model = new WanModel(ditLoader, "");

        int latH = 32, latW = 32, latC = 16, numFrames = 1;
        var latent = new float[latC * numFrames * latH * latW];
        var rng = new Random(42);
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // A real, mid-noise timestep (not t=1000 pure-noise edge case) so any real denoising
        // signal has the best chance of being visible.
        Log("[PerBlock Bisection] Running one real CPU Forward pass (30 blocks, real weights)...");
        var velocity = model.Forward(latent, 500f, condContext, numFrames, latH, latW);

        float vMean = 0, vSumSq = 0;
        for (int i = 0; i < velocity.Length; i++) { vMean += velocity[i]; vSumSq += velocity[i] * velocity[i]; }
        vMean /= velocity.Length;
        float vStd = MathF.Sqrt(Math.Max(0, vSumSq / velocity.Length - vMean * vMean));
        Log($"[PerBlock Bisection] Final velocity output: mean={vMean:F6} std={vStd:F6}");

        Environment.SetEnvironmentVariable("STINGRAY_WAN_DEBUG_PERBLOCK", null);
    }

    [Fact]
    public void Stage_InspectRealWeightShapes_CheckLinearOrientation()
    {
        string ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));
        if (!File.Exists(ditPath)) { Log("Wan DiT checkpoint missing, skipping."); return; }

        using var st = SafetensorsLoader.Open(ditPath);
        string[] namesToCheck =
        [
            "text_embedding.0.weight",   // real: Linear(text_dim=4096, dim=1536) -> PyTorch [out,in]=[1536,4096]
            "patch_embedding.weight",    // real: Conv3d -> [dim, in_ch, kt, kh, kw] = [1536,16,1,2,2]
            "head.head.weight",          // real: Linear(dim=1536, out_dim=64) -> [64, 1536]
            "blocks.0.self_attn.q.weight",
            "blocks.0.ffn.0.weight",     // real: Linear(dim=1536, ffn_dim=8960) -> [8960, 1536]
        ];
        foreach (var name in namesToCheck)
        {
            if (!st.TensorNames.Contains(name)) { Log($"[ShapeCheck] {name}: NOT FOUND"); continue; }
            var shape = st.GetShape(name);
            Log($"[ShapeCheck] {name}: [{string.Join(", ", shape)}]");
        }
    }
}
