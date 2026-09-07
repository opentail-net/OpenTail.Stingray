using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test chaining EVERY VoxCPM2 piece real-weight verified this
/// session (text tokenizer, `base_lm` prefill, `residual_lm`, step projection, CFM/DiT patch
/// generation, local encoder self-conditioning, AudioVAE decode) into the real reference-audio-
/// free `generate_once` loop from `generator.cpp` -- the first full text-to-waveform run for this
/// model, analogous to `PersonaPlexFullPipelineRealWeightsTests`.
/// </summary>
public sealed class VoxCpm2GeneratorRealWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private const int NumLayers = 28, HiddenDim = 2048, NumHeads = 16, NumKvHeads = 2, HeadDim = 128;
    private const int FfDim = 6144, VocabSize = 73448;
    private const float RopeTheta = 10000f, RmsNormEps = 1e-5f;

    // Real config.audio_vae_config from the checkpoint's embedded config.json (confirmed via
    // VoxCpm2DumpDebugTest, not guessed) -- same as VoxCpm2AudioVaeDecoderRealWeightsTests.
    private static VoxCpm2AudioVaeConfig RealVaeConfig() => new()
    {
        LatentDim = 64,
        DecoderDim = 2048,
        DecoderRates = [8, 6, 5, 2, 2, 2],
        OutputSampleRate = 48000,
        SampleRateBinBoundaries = [20000, 30000, 40000],
    };

    [Fact]
    public void Generate_ThenDecode_OnRealCheckpoint_ProducesFiniteWaveform()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        var tokenizer = VoxCpm2TextTokenizer.LoadFromPackedGguf(model);
        var textIds = tokenizer.Encode("This is a real end to end test.");
        var promptTokenIds = new int[textIds.Count + 1];
        textIds.CopyTo(promptTokenIds);
        promptTokenIds[^1] = tokenizer.AudioStartTokenId;

        var llm = new VoxCpm2LlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, embeddingScale: 1f, residualScale: 1f, logitScale: 1f);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var residualLm = VoxCpm2ResidualLm.Load(source.GetTensor);
        var projWeights = VoxCpm2StepProjectionWeights.Load(source.GetTensor);
        var ditWeights = VoxCpm2DiTEstimatorWeights.Load(numLayers: 12, source.GetTensor);
        var encoderWeights = VoxCpm2LocalEncoderWeights.Load(numLayers: 12, source.GetTensor);

        var result = VoxCpm2Generator.Generate(
            fwd, residualLm, projWeights, ditWeights, encoderWeights,
            promptTokenIds, maxPatches: 3, cfmTimesteps: 4, cfgValue: 2.0f, minTokens: 0,
            new Random(11));

        Assert.True(result.Patches.Length > 0);
        foreach (var patch in result.Patches)
        {
            Assert.Equal(VoxCpm2Generator.PatchSize, patch.Length);
            foreach (var frame in patch)
            {
                Assert.Equal(VoxCpm2Generator.FeatDim, frame.Length);
                Assert.All(frame, v => Assert.True(float.IsFinite(v)));
            }
        }

        int totalFrames = result.Patches.Length * VoxCpm2Generator.PatchSize;
        var vaeConfig = RealVaeConfig();
        Assert.Equal(VoxCpm2Generator.FeatDim, vaeConfig.LatentDim);
        var latentChannelMajor = new float[vaeConfig.LatentDim * totalFrames];
        int f = 0;
        foreach (var patch in result.Patches)
        {
            foreach (var frame in patch)
            {
                for (int c = 0; c < vaeConfig.LatentDim; c++) latentChannelMajor[c * totalFrames + f] = frame[c];
                f++;
            }
        }

        var vaeWeights = VoxCpm2AudioVaeDecoderWeights.Load(vaeConfig, name => source.GetTensor($"audiovae_weights/{name}"));
        var waveform = VoxCpm2AudioVaeDecoder.Decode(vaeConfig, vaeWeights, latentChannelMajor, totalFrames);

        Assert.True(waveform.Length > 0);
        Assert.All(waveform, v => Assert.True(float.IsFinite(v)));
        Assert.All(waveform, v => Assert.InRange(v, -1f, 1f));
    }
}
