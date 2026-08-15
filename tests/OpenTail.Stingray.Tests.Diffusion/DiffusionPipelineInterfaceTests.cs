using OpenTail.Stingray.Diffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Phase 1 acceptance guard for docs/to do - stable diffusion/033-native-stable-diffusion-family-port-plan.md:
/// FLUX (<see cref="ImagePipeline"/>) and Z-Image (<see cref="ZImagePipeline"/>) must conform to the
/// shared <see cref="IDiffusionPipeline"/> abstraction. These pipelines require real GGUF/safetensors
/// weights to instantiate, so this only asserts the interface contract (compile-time-checkable
/// regression: if the explicit interface implementation is ever removed, this fails) rather than
/// exercising generation — that's covered by the existing manual/benchmark image-generation paths.
/// </summary>
public sealed class DiffusionPipelineInterfaceTests
{
    [Fact]
    public void ImagePipeline_ImplementsIDiffusionPipeline()
    {
        Assert.True(typeof(IDiffusionPipeline).IsAssignableFrom(typeof(ImagePipeline)));
    }

    [Fact]
    public void ZImagePipeline_ImplementsIDiffusionPipeline()
    {
        Assert.True(typeof(IDiffusionPipeline).IsAssignableFrom(typeof(ZImagePipeline)));
    }

    [Fact]
    public void VaeDecoder_ImplementsIVaeDecoder()
    {
        Assert.True(typeof(IVaeDecoder).IsAssignableFrom(typeof(VaeDecoder)));
    }

    [Fact]
    public void ImageGenerationRequest_DefaultsMatchExistingPipelineDefaults()
    {
        var request = new ImageGenerationRequest { Prompt = "a cat", OutputPath = "out.png" };

        Assert.Equal(512, request.Width);
        Assert.Equal(512, request.Height);
        Assert.Equal(-1, request.Steps); // sentinel: "use the pipeline's own default"
        Assert.Equal(1.0f, request.Guidance);
        Assert.Equal(-1, request.Seed);
        Assert.Equal(1.0f, request.UpscaleBlend);
        Assert.Null(request.Upscaler);
    }
}
