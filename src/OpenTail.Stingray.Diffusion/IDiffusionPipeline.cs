namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Common entry point shared by every image-generation pipeline (FLUX, Z-Image, and the
/// planned SD1.5/SDXL/SD3.5 families — see docs/to do - stable diffusion/033-native-stable-diffusion-family-port-plan.md).
/// <see cref="ImagePipeline"/> and <see cref="ZImagePipeline"/> implement this explicitly;
/// their existing public <c>Generate(...)</c> overloads are unchanged, so callers that already
/// use those concrete types see no behavior difference.
/// </summary>
public interface IDiffusionPipeline : IDisposable
{
    void Generate(ImageGenerationRequest request);
}

/// <summary>
/// Pipeline-agnostic generation request. Mirrors the union of parameters
/// <see cref="ImagePipeline.Generate"/> and <see cref="ZImagePipeline.Generate"/> already
/// accept; a pipeline that doesn't use a given field (e.g. <see cref="Guidance"/> for
/// Z-Image, which is CFG-distilled) ignores it.
/// </summary>
public sealed record ImageGenerationRequest
{
    public required string Prompt { get; init; }
    public int Width { get; init; } = 512;
    public int Height { get; init; } = 512;

    /// <summary>Denoising steps. -1 means "use the pipeline's own default".</summary>
    public int Steps { get; init; } = -1;

    public float Guidance { get; init; } = 1.0f;
    public int Seed { get; init; } = -1;
    public required string OutputPath { get; init; }
    public RRDBNet? Upscaler { get; init; }
    public float UpscaleBlend { get; init; } = 1.0f;
    public Action<int, int>? Progress { get; init; }
    public Action<string>? StatusCallback { get; init; }
}

/// <summary>
/// Decodes a latent tensor to RGB pixels. Implemented today by <see cref="VaeDecoder"/>
/// (FLUX/Z-Image, 16-channel latents). SD1.5/SDXL's 4-channel VAE is architecturally
/// different internally (per 033 Phase 2.6, extend rather than duplicate where
/// mathematically compatible) but decodes through the same shape of call.
/// </summary>
public interface IVaeDecoder
{
    /// <summary>latent: [C, latentHeight, latentWidth] (batch=1) → RGB [3, latentHeight*8, latentWidth*8] in [0,1].</summary>
    float[] Decode(float[] latent, int latentHeight, int latentWidth);
}

/// <summary>
/// Encodes a prompt into whatever conditioning shape a model family needs (CLIP hidden
/// states, pooled embeddings, T5 states, ...). Deliberately opaque here — SD1.5 needs a
/// single CLIP context tensor, SDXL needs CLIP-L+CLIP-G concatenated plus a pooled vector,
/// SD3.5 needs CLIP-L+CLIP-G+T5-XXL. <see cref="IConditioning"/> is the per-family payload;
/// this interface is not yet implemented by FLUX/Z-Image's concrete encoders
/// (<c>ClipLEncoder</c>, <c>T5Encoder</c>, <c>QwenTextEncoder</c>) — retrofitting those is
/// deferred until SD1.5's CLIP encoder exists as a second real implementation to validate
/// the shape against, per 033 Phase 2.2.
/// </summary>
public interface ITextConditioner
{
    IConditioning Encode(string prompt);
}

/// <summary>Marker for a family-specific conditioning payload produced by <see cref="ITextConditioner"/>.</summary>
public interface IConditioning;

/// <summary>
/// One forward pass of a diffusion model: predict noise/velocity for the current latent
/// at a given timestep. Target contract for SD1.5's UNet (033 Phase 2.3); not yet
/// implemented by <c>FluxDiT</c>/<c>ZImageDiT</c> for the same reason as
/// <see cref="ITextConditioner"/> — those take pipeline-specific extra arguments
/// (image/text RoPE position ids, pooled embeddings) that don't collapse into a single
/// generic signature without being guessed at ahead of a second real implementation.
/// </summary>
public interface IDiffusionModel
{
    float[] Forward(float[] latent, float timestep, IConditioning conditioning, float guidanceScale);
}

/// <summary>
/// Turns noise into a denoised latent over a fixed step schedule. FLUX/Z-Image already
/// have a concrete, working scheduler (<c>EulerFlowScheduler</c>) with its own
/// pack/unpack helpers tied to their specific patch layouts; this is the target contract
/// for SD1.5/SDXL's scheduler (033 Phase 2.4: start with one baseline scheduler, then add
/// Euler ancestral/DDIM/DPM), not a retrofit of <c>EulerFlowScheduler</c> itself.
/// </summary>
public interface IDiffusionScheduler
{
    float[] SampleNoise(int elementCount, int seed);

    float[] Denoise(float[] initialLatent, Func<float[], float, float[]> predictNoise,
                     Action<int, int>? progress = null);
}

/// <summary>
/// Classifier-free-guidance combination and per-step update math, factored out from
/// <see cref="IDiffusionScheduler"/> so a scheduler (e.g. DDIM vs. Euler) and a sampler
/// (e.g. how CFG is combined) can vary independently, per 033 Phase 2.5.
/// </summary>
public interface IDiffusionSampler
{
    float[] CombineGuidance(float[] noisePredConditional, float[] noisePredUnconditional, float guidanceScale);
}
