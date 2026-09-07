using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test for VoxCPM2's real voice-cloning / reference-audio
/// conditioning path: encodes a synthetic "reference audio" waveform through the real AudioVAE
/// ENCODER (ported this session), chunks it into real patches, embeds each patch through the
/// real local encoder, splices those as real AUDIO prefill rows (matching
/// `build_prefill_sequence`'s real reference-audio branch) alongside real TEXT rows, then runs
/// the real generation loop and decodes the result -- the first real reference-audio-conditioned
/// run for this model.
/// </summary>
public sealed class VoxCpm2GeneratorWithPromptRealWeightsTests : HeavyTestBase
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

    private static VoxCpm2AudioVaeConfig RealVaeConfig() => new()
    {
        LatentDim = 64,
        DecoderDim = 2048,
        DecoderRates = [8, 6, 5, 2, 2, 2],
        OutputSampleRate = 48000,
        SampleRateBinBoundaries = [20000, 30000, 40000],
        EncoderDim = 128,
        EncoderRates = [2, 5, 8, 8],
    };

    [Fact]
    public void GenerateWithPrompt_ThenDecode_OnRealCheckpoint_ProducesFiniteWaveform()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var vaeConfig = RealVaeConfig();
        var vaeWeights = VoxCpm2AudioVaeDecoderWeights.Load(vaeConfig, name => source.GetTensor($"audiovae_weights/{name}"));

        var tokenizer = VoxCpm2TextTokenizer.LoadFromPackedGguf(model);
        var encoderWeights = VoxCpm2LocalEncoderWeights.Load(numLayers: 12, source.GetTensor);

        // Real reference-audio conditioning path: encode a synthetic waveform (stand-in for a
        // real reference clip) through the real AudioVAE encoder, chunk into real
        // [PatchSize][FeatDim] patches, embed each through the real local encoder.
        int encoderStride = vaeConfig.EncoderRates.Aggregate(1, (a, b) => a * b);
        int refPatches = 3;
        int refSamples = encoderStride * VoxCpm2Generator.PatchSize * refPatches;
        var rng = new Random(7);
        var refPcm = new float[refSamples];
        for (int i = 0; i < refSamples; i++) refPcm[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var refLatentChannelMajor = VoxCpm2AudioVaeDecoder.Encode(vaeConfig, vaeWeights, refPcm);
        int refFrames = refLatentChannelMajor.Length / vaeConfig.LatentDim;
        Assert.True(refFrames >= VoxCpm2Generator.PatchSize * refPatches);

        var rows = new List<VoxCpm2Generator.PrefillRow> { VoxCpm2Generator.PrefillRow.Text(tokenizer.ReferenceAudioStartTokenId) };
        for (int p = 0; p < refPatches; p++)
        {
            var patch = new float[VoxCpm2Generator.PatchSize][];
            for (int f = 0; f < VoxCpm2Generator.PatchSize; f++)
            {
                int frame = p * VoxCpm2Generator.PatchSize + f;
                var row = new float[VoxCpm2Generator.FeatDim];
                for (int c = 0; c < VoxCpm2Generator.FeatDim; c++) row[c] = refLatentChannelMajor[c * refFrames + frame];
                patch[f] = row;
            }
            var embedding = VoxCpm2LocalEncoder.EncodePatch(encoderWeights, patch, HiddenDim);
            rows.Add(VoxCpm2Generator.PrefillRow.Audio(patch, embedding));
        }
        rows.Add(VoxCpm2Generator.PrefillRow.Text(tokenizer.ReferenceAudioEndTokenId));

        var textIds = tokenizer.Encode("This is a real reference audio test.");
        foreach (int id in textIds) rows.Add(VoxCpm2Generator.PrefillRow.Text(id));
        rows.Add(VoxCpm2Generator.PrefillRow.Text(tokenizer.AudioStartTokenId));

        var llm = new VoxCpm2LlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, embeddingScale: 1f, residualScale: 1f, logitScale: 1f);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var residualLm = VoxCpm2ResidualLm.Load(source.GetTensor);
        var projWeights = VoxCpm2StepProjectionWeights.Load(source.GetTensor);
        var ditWeights = VoxCpm2DiTEstimatorWeights.Load(numLayers: 12, source.GetTensor);

        var result = VoxCpm2Generator.GenerateWithPrompt(
            fwd, residualLm, projWeights, ditWeights, encoderWeights,
            rows, maxPatches: 3, cfmTimesteps: 4, cfgValue: 2.0f, minTokens: 0,
            new Random(13));

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
        var latentChannelMajor = new float[vaeConfig.LatentDim * totalFrames];
        int fIdx = 0;
        foreach (var patch in result.Patches)
        {
            foreach (var frame in patch)
            {
                for (int c = 0; c < vaeConfig.LatentDim; c++) latentChannelMajor[c * totalFrames + fIdx] = frame[c];
                fIdx++;
            }
        }

        var waveform = VoxCpm2AudioVaeDecoder.Decode(vaeConfig, vaeWeights, latentChannelMajor, totalFrames);
        Assert.True(waveform.Length > 0);
        Assert.All(waveform, v => Assert.True(float.IsFinite(v)));
        Assert.All(waveform, v => Assert.InRange(v, -1f, 1f));
    }
}
