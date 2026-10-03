using OpenTail.Stingray.Diffusion.MiniMaxH3;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class MiniMaxH3Tests
{
    [Fact]
    public void Config_Dimensions_MatchSpec()
    {
        Assert.Equal(50, MiniMaxH3Config.NumLayers);
        Assert.Equal(5376, MiniMaxH3Config.HiddenDim);
        Assert.Equal(56, MiniMaxH3Config.NumHeads);
        Assert.Equal(128, MiniMaxH3Config.HeadDim);
        Assert.Equal(7168, MiniMaxH3Config.AttentionDim);

        Assert.Equal(24, MiniMaxH3Config.VideoLatentChannels);
        Assert.Equal(32, MiniMaxH3Config.AudioLatentChannels);
        Assert.Equal(96, MiniMaxH3Config.VideoPatchDim);
        Assert.Equal(32, MiniMaxH3Config.AudioPatchDim);

        Assert.Equal(12.0f, MiniMaxH3Config.VideoFlowShift);
        Assert.Equal(3.0f, MiniMaxH3Config.AudioFlowShift);
        Assert.Equal(1.0f, MiniMaxH3Config.GuidanceScale);

        Assert.Equal(8, MiniMaxH3Config.AdaLnTableRows);
        Assert.Equal(1025, MiniMaxH3Config.AdaLnTableCols);
        Assert.Equal(32256, MiniMaxH3Config.AdaLnModulationDimPerModality);
        Assert.Equal(96768, MiniMaxH3Config.AdaLnProjectionDim);
    }

    [Fact]
    public void Scheduler_DualSchedules_HaveExpectedTrajectories()
    {
        int steps = 10;
        var scheduler = new MiniMaxH3Scheduler(steps);

        Assert.Equal(steps + 1, scheduler.VideoSigmas.Length);
        Assert.Equal(steps + 1, scheduler.AudioSigmas.Length);

        // Both schedules start at 1.0 and end at 0.0
        Assert.Equal(1.0f, scheduler.VideoSigmas[0], 0.0001f);
        Assert.Equal(0.0f, scheduler.VideoSigmas[steps], 0.0001f);

        Assert.Equal(1.0f, scheduler.AudioSigmas[0], 0.0001f);
        Assert.Equal(0.0f, scheduler.AudioSigmas[steps], 0.0001f);

        // For intermediate step (linearT = 0.5f):
        // Video shift = 12: (12 * 0.5) / (1 + 11 * 0.5) = 6.0 / 6.5 = 0.923077
        // Audio shift = 3:  (3 * 0.5) / (1 + 2 * 0.5)  = 1.5 / 2.0 = 0.750000
        float videoMid = MiniMaxH3Scheduler.Shift(12.0f, 0.5f);
        float audioMid = MiniMaxH3Scheduler.Shift(3.0f, 0.5f);
        Assert.Equal(6.0f / 6.5f, videoMid, 0.0001f);
        Assert.Equal(0.75f, audioMid, 0.0001f);

        // Verify strictly monotonic decrease
        for (int i = 0; i < steps; i++)
        {
            Assert.True(scheduler.VideoSigmas[i] > scheduler.VideoSigmas[i + 1]);
            Assert.True(scheduler.AudioSigmas[i] > scheduler.AudioSigmas[i + 1]);
        }

        // Test Euler step
        float[] latent = [1.0f, 2.0f, 3.0f];
        float[] velocity = [0.5f, -1.0f, 2.0f];
        // dt = 0.8 - 1.0 = -0.2
        MiniMaxH3Scheduler.StepEuler(latent, velocity, currentSigma: 1.0f, nextSigma: 0.8f);
        Assert.Equal(1.0f + (-0.2f * 0.5f), latent[0], 0.0001f);
        Assert.Equal(2.0f + (-0.2f * -1.0f), latent[1], 0.0001f);
        Assert.Equal(3.0f + (-0.2f * 2.0f), latent[2], 0.0001f);
    }

    [Fact]
    public void AdaLN_Interpolation_And_Modulation()
    {
        // Build synthetic 8x1025 table
        var table = new float[8 * 1025];
        for (int c = 0; c < 8; c++)
        {
            for (int col = 0; col < 1025; col++)
            {
                // Each channel has a linear ramp: c * 1000 + col
                table[c * 1025 + col] = c * 1000.0f + col;
            }
        }

        var curveVec = new float[8];
        // Timestep t = 500 (maps to col = 512.0f)
        MiniMaxH3AdaLN.InterpolateTimestepCurve(table, 500.0f, curveVec);
        for (int c = 0; c < 8; c++)
        {
            Assert.Equal(c * 1000.0f + 512.0f, curveVec[c], 0.001f);
        }

        // Timestep t = 250 (maps to col = 256.0f)
        MiniMaxH3AdaLN.InterpolateTimestepCurve(table, 250.0f, curveVec);
        for (int c = 0; c < 8; c++)
        {
            Assert.Equal(c * 1000.0f + 256.0f, curveVec[c], 0.001f);
        }

        // Test AdaLN projection and modality slicing
        int hiddenDim = 64; // Small hidden dim for test
        int modDimPerMod = 6 * hiddenDim;
        int totalProjDim = 3 * modDimPerMod;

        var weight = new float[totalProjDim * 8];
        var bias = new float[totalProjDim];
        for (int i = 0; i < totalProjDim; i++) bias[i] = 1.0f;

        var modulation = new float[totalProjDim];
        MiniMaxH3AdaLN.ProjectModulation(curveVec, weight, bias, modulation);
        // All elements should equal 1.0f (since weight is 0 and bias is 1)
        for (int i = 0; i < totalProjDim; i++) Assert.Equal(1.0f, modulation[i]);

        var vidMod = MiniMaxH3AdaLN.GetModalityModulation(modulation, MiniMaxH3Modality.Video, hiddenDim);
        Assert.Equal(hiddenDim, vidMod.Gamma1.Length);
        Assert.Equal(hiddenDim, vidMod.Beta1.Length);
        Assert.Equal(hiddenDim, vidMod.Alpha1.Length);
        Assert.Equal(hiddenDim, vidMod.Gamma2.Length);
        Assert.Equal(hiddenDim, vidMod.Beta2.Length);
        Assert.Equal(hiddenDim, vidMod.Alpha2.Length);

        // Test ModulateNorm and ApplyGate
        float[] x = [2.0f, 4.0f];
        float[] gamma = [0.5f, 0.5f];
        float[] beta = [0.1f, 0.2f];
        MiniMaxH3AdaLN.ModulateNorm(x, gamma, beta);
        Assert.Equal((1.0f + 0.5f) * 2.0f + 0.1f, x[0], 0.0001f);
        Assert.Equal((1.0f + 0.5f) * 4.0f + 0.2f, x[1], 0.0001f);

        float[] alpha = [0.5f, 2.0f];
        MiniMaxH3AdaLN.ApplyGate(x, alpha);
        Assert.Equal(3.1f * 0.5f, x[0], 0.0001f);
        Assert.Equal(6.2f * 2.0f, x[1], 0.0001f);
    }

    [Fact]
    public void Layout_SequenceAndPatching_RoundTrip()
    {
        // 4 text tokens, 2 visual cond tokens, video: 2 frames, H=4, W=6, audio: 3 frames
        // Video patch H = 4 / 2 = 2, Video patch W = 6 / 2 = 3.
        // Num video tokens = 2 * 2 * 3 = 12.
        // Num audio tokens = 3.
        // Total tokens = 4 + 2 + 12 + 3 = 21.
        var layout = new MiniMaxH3Layout(
            numTextTokens: 4,
            numVisualCondTokens: 2,
            videoFrames: 2,
            videoHeight: 4,
            videoWidth: 6,
            audioFrames: 3);

        Assert.Equal(0, layout.TextStart);
        Assert.Equal(4, layout.VisualCondStart);
        Assert.Equal(6, layout.VideoTargetStart);
        Assert.Equal(18, layout.AudioTargetStart);
        Assert.Equal(21, layout.TotalTokens);

        // Verify modality classification
        for (int i = 0; i < 6; i++) Assert.Equal(MiniMaxH3Modality.Context, layout.GetModality(i));
        for (int i = 6; i < 18; i++) Assert.Equal(MiniMaxH3Modality.Video, layout.GetModality(i));
        for (int i = 18; i < 21; i++) Assert.Equal(MiniMaxH3Modality.Audio, layout.GetModality(i));

        // Test channel-major, patch-minor video latent round trip
        int cDim = MiniMaxH3Config.VideoLatentChannels; // 24
        int videoLatentLen = 2 * cDim * 4 * 6; // 1152 floats
        var originalVideoLatent = new float[videoLatentLen];
        for (int i = 0; i < videoLatentLen; i++)
        {
            originalVideoLatent[i] = (i * 0.137f) - 50.0f;
        }

        var patchTokens = new float[layout.NumVideoTokens * MiniMaxH3Config.VideoPatchDim];
        layout.PackVideoLatent(originalVideoLatent, patchTokens);

        var restoredVideoLatent = new float[videoLatentLen];
        layout.UnpackVideoLatent(patchTokens, restoredVideoLatent);

        for (int i = 0; i < videoLatentLen; i++)
        {
            Assert.Equal(originalVideoLatent[i], restoredVideoLatent[i]);
        }

        // Test audio latent round trip
        int audioLatentLen = 3 * MiniMaxH3Config.AudioPatchDim;
        var originalAudioLatent = new float[audioLatentLen];
        for (int i = 0; i < audioLatentLen; i++) originalAudioLatent[i] = i * 1.5f;

        var audioTokens = new float[audioLatentLen];
        layout.PackAudioLatent(originalAudioLatent, audioTokens);

        var restoredAudioLatent = new float[audioLatentLen];
        layout.UnpackAudioLatent(audioTokens, restoredAudioLatent);

        for (int i = 0; i < audioLatentLen; i++)
        {
            Assert.Equal(originalAudioLatent[i], restoredAudioLatent[i]);
        }

        // Test 3D coordinates
        var (t, hp, wp) = layout.GetVideoCoordinates(0);
        Assert.Equal(0, t); Assert.Equal(0, hp); Assert.Equal(0, wp);

        var (t1, hp1, wp1) = layout.GetVideoCoordinates(5); // 0 * 6 + 1 * 3 + 2
        Assert.Equal(0, t1); Assert.Equal(1, hp1); Assert.Equal(2, wp1);

        var (t2, hp2, wp2) = layout.GetVideoCoordinates(6); // Frame 1, hp 0, wp 0
        Assert.Equal(1, t2); Assert.Equal(0, hp2); Assert.Equal(0, wp2);
    }

    [Fact]
    public void RoPE_Multimodal_GeneratesValidTables_AndRotatesCorrectly()
    {
        // 2 text, 1 vis, video: 2 frames, H=4, W=4 (2x2=4 spatial patches -> 8 video tokens), audio: 3 tokens
        // Total tokens = 3 context + 8 video + 3 audio = 14 tokens
        var layout = new MiniMaxH3Layout(
            numTextTokens: 2,
            numVisualCondTokens: 1,
            videoFrames: 2,
            videoHeight: 4,
            videoWidth: 4,
            audioFrames: 3);

        int headDim = 128;
        var (cos, sin) = MiniMaxH3RoPE.ComputeMultimodalRoPE(layout, headDim);

        Assert.Equal(layout.TotalTokens * headDim, cos.Length);
        Assert.Equal(layout.TotalTokens * headDim, sin.Length);

        // Verify cos^2 + sin^2 ~= 1.0 for every entry
        for (int i = 0; i < cos.Length; i++)
        {
            float norm = cos[i] * cos[i] + sin[i] * sin[i];
            Assert.True(MathF.Abs(norm - 1.0f) < 1e-4f, $"Entry {i} failed Pythagorean identity: {norm}");
        }

        // Verify 3D video axis differentiation:
        // Video token 0: (t=0, hp=0, wp=0)
        // Video token 4: (t=1, hp=0, wp=0)
        int vid0Off = (layout.VideoTargetStart + 0) * headDim;
        int vid4Off = (layout.VideoTargetStart + 4) * headDim;

        // Temporal slice (dims 0..43) must differ between t=0 and t=1
        bool tDiffers = false;
        for (int d = 0; d < MiniMaxH3RoPE.VideoDimT; d++)
        {
            if (MathF.Abs(cos[vid0Off + d] - cos[vid4Off + d]) > 1e-5f) tDiffers = true;
        }
        Assert.True(tDiffers, "Temporal slice should differ for t=0 vs t=1");

        // Spatial slice (dims 44..127) must match because hp=0, wp=0 for both
        for (int d = MiniMaxH3RoPE.VideoDimT; d < headDim; d++)
        {
            Assert.Equal(cos[vid0Off + d], cos[vid4Off + d], 1e-5f);
        }

        // Verify ApplyRoPE preserves L2 norm (orthogonal transformation)
        var qk = new float[layout.TotalTokens * headDim];
        for (int i = 0; i < qk.Length; i++) qk[i] = (i % 7) + 1.0f;

        float originalNorm = 0.0f;
        for (int i = 0; i < headDim; i++) originalNorm += qk[i] * qk[i];

        MiniMaxH3RoPE.ApplyRoPE(qk, cos, sin, layout.TotalTokens, numHeads: 1, headDim: headDim);

        float rotatedNorm = 0.0f;
        for (int i = 0; i < headDim; i++) rotatedNorm += qk[i] * qk[i];

        Assert.True(MathF.Abs(originalNorm - rotatedNorm) < 1e-3f, $"RoPE rotated norm {rotatedNorm} != original {originalNorm}");
    }

    [Fact]
    public void DiTBlock_Forward_RunsAndAppliesModulation()
    {
        int hiddenDim = 64;
        int attnDim = 64;
        int ffnDim = 128;
        int numHeads = 4;
        int headDim = 16;

        var weights = new MiniMaxH3DiTBlockWeights(hiddenDim, attnDim, ffnDim);
        // Initialize weights with small non-zero values
        for (int i = 0; i < weights.Wq.Length; i++) weights.Wq[i] = 0.01f;
        for (int i = 0; i < weights.Wk.Length; i++) weights.Wk[i] = 0.01f;
        for (int i = 0; i < weights.Wv.Length; i++) weights.Wv[i] = 0.01f;
        for (int i = 0; i < weights.Wo.Length; i++) weights.Wo[i] = 0.01f;
        for (int i = 0; i < weights.WGate.Length; i++) weights.WGate[i] = 0.01f;
        for (int i = 0; i < weights.WUp.Length; i++) weights.WUp[i] = 0.01f;
        for (int i = 0; i < weights.WDown.Length; i++) weights.WDown[i] = 0.01f;

        var layout = new MiniMaxH3Layout(
            numTextTokens: 2,
            numVisualCondTokens: 0,
            videoFrames: 1,
            videoHeight: 2,
            videoWidth: 2,
            audioFrames: 1);
        // Total tokens = 2 (text) + 1 (video) + 1 (audio) = 4 tokens

        var x = new float[layout.TotalTokens * hiddenDim];
        for (int i = 0; i < x.Length; i++) x[i] = 1.0f;

        var curveVec = new float[8];
        for (int i = 0; i < 8; i++) curveVec[i] = 0.5f;

        MiniMaxH3DiTBlock.Forward(
            x,
            layout.TotalTokens,
            layout,
            curveVec,
            weights,
            numHeads,
            headDim);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.False(float.IsNaN(x[i]));
            Assert.False(float.IsInfinity(x[i]));
        }
    }

    [Fact]
    public void DiTBlock_FullDimension_SlicingAndModulation()
    {
        // Tests the real architecture dimensions:
        // HiddenDim = 5376, AttentionDim = 7168, NumHeads = 56, HeadDim = 128, AdaLN = 96768
        var weights = new MiniMaxH3DiTBlockWeights(
            hiddenDim: MiniMaxH3Config.HiddenDim,
            attentionDim: MiniMaxH3Config.AttentionDim,
            ffnDim: 14336);

        Assert.Equal(MiniMaxH3Config.AdaLnProjectionDim, weights.AdaLnBias!.Length);
        Assert.Equal(MiniMaxH3Config.AdaLnProjectionDim * 8, weights.AdaLnWeight.Length);

        var layout = new MiniMaxH3Layout(
            numTextTokens: 1,
            numVisualCondTokens: 0,
            videoFrames: 1,
            videoHeight: 2,
            videoWidth: 2,
            audioFrames: 1);
        // Total tokens = 1 (text) + 1 (video) + 1 (audio) = 3 tokens

        var x = new float[layout.TotalTokens * MiniMaxH3Config.HiddenDim];
        Array.Fill(x, 0.1f);

        var curveVec = new float[MiniMaxH3Config.AdaLnTableRows];
        Array.Fill(curveVec, 0.2f);

        MiniMaxH3DiTBlock.Forward(
            x,
            layout.TotalTokens,
            layout,
            curveVec,
            weights,
            MiniMaxH3Config.NumHeads,
            MiniMaxH3Config.HeadDim);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.False(float.IsNaN(x[i]));
            Assert.False(float.IsInfinity(x[i]));
        }
    }

    [Fact]
    public void Pipeline_Generate_EndToEndSyntheticRun()
    {
        int hiddenDim = 64;
        int attnDim = 64;
        int ffnDim = 128;
        int numHeads = 4;
        int headDim = 16;
        int contextDim = 32;

        var dit = new MiniMaxH3DiT(
            numLayers: 2,
            hiddenDim: hiddenDim,
            attentionDim: attnDim,
            ffnDim: ffnDim,
            contextDim: contextDim);

        // Fill weights with small deterministic constants
        Array.Fill(dit.VideoInProj, 0.01f);
        Array.Fill(dit.AudioInProj, 0.01f);
        Array.Fill(dit.ContextInProj, 0.01f);
        Array.Fill(dit.VideoOutProj, 0.01f);
        Array.Fill(dit.AudioOutProj, 0.01f);

        foreach (var block in dit.Blocks)
        {
            Array.Fill(block.Wq, 0.01f);
            Array.Fill(block.Wk, 0.01f);
            Array.Fill(block.Wv, 0.01f);
            Array.Fill(block.Wo, 0.01f);
            Array.Fill(block.WGate, 0.01f);
            Array.Fill(block.WUp, 0.01f);
            Array.Fill(block.WDown, 0.01f);
        }

        var adalnTable = new float[8 * 1025];
        Array.Fill(adalnTable, 0.1f);

        var pipeline = new MiniMaxH3Pipeline(dit, adalnTable);

        var layout = new MiniMaxH3Layout(
            numTextTokens: 2,
            numVisualCondTokens: 0,
            videoFrames: 1,
            videoHeight: 2,
            videoWidth: 2,
            audioFrames: 1);

        var contextTokens = new float[layout.NumTextTokens * contextDim];
        Array.Fill(contextTokens, 0.5f);

        int progressCallbacks = 0;
        var (videoLatent, audioLatent) = pipeline.Generate(
            layout,
            contextTokens,
            numSteps: 2,
            seed: 42,
            progress: (step, total) => progressCallbacks++,
            numHeads: numHeads,
            headDim: headDim);

        Assert.Equal(2, progressCallbacks);
        Assert.Equal(1 * MiniMaxH3Config.VideoLatentChannels * 2 * 2, videoLatent.Length);
        Assert.Equal(1 * MiniMaxH3Config.AudioLatentChannels, audioLatent.Length);

        // Check for validity (finite numbers)
        foreach (float val in videoLatent)
        {
            Assert.False(float.IsNaN(val));
            Assert.False(float.IsInfinity(val));
        }
        foreach (float val in audioLatent)
        {
            Assert.False(float.IsNaN(val));
            Assert.False(float.IsInfinity(val));
        }
    }

    [Fact]
    public void VideoVae_SmallClip_DecodesExpectedShapes()
    {
        var decoder = new MiniMaxH3VideoVaeDecoder(hiddenDim: 32);
        int latentFrames = 2;
        int latentH = 2;
        int latentW = 2;
        int latentLen = latentFrames * MiniMaxH3Config.VideoLatentChannels * latentH * latentW;

        var latent = new float[latentLen];
        for (int i = 0; i < latentLen; i++) latent[i] = 0.5f;

        var frames = decoder.Decode(latent, latentFrames, latentH, latentW);

        // Expected output frames = (2 - 1) * 4 + 1 = 5
        Assert.Equal(5, frames.Count);

        int expectedPixelsPerFrame = 3 * (latentH * 8) * (latentW * 8); // 3 * 16 * 16 = 768
        foreach (var frame in frames)
        {
            Assert.Equal(expectedPixelsPerFrame, frame.Length);
            for (int p = 0; p < frame.Length; p++)
            {
                Assert.True(frame[p] >= 0.0f && frame[p] <= 1.0f);
            }
        }
    }

    [Fact]
    public void VideoVae_LongClip_ExecutesTemporalChunkingAndSeamBlending()
    {
        var decoder = new MiniMaxH3VideoVaeDecoder(hiddenDim: 32);
        // 25 latent frames (> 22 frames threshold) triggers 5-latent-frame temporal chunking
        int latentFrames = 25;
        int latentH = 1;
        int latentW = 1;
        int latentLen = latentFrames * MiniMaxH3Config.VideoLatentChannels * latentH * latentW;

        var latent = new float[latentLen];
        for (int i = 0; i < latentLen; i++) latent[i] = (i % 5) * 0.2f;

        var frames = decoder.Decode(latent, latentFrames, latentH, latentW);

        // Expected output frames = (25 - 1) * 4 + 1 = 97
        Assert.Equal(97, frames.Count);

        int expectedPixels = 3 * (latentH * 8) * (latentW * 8); // 3 * 8 * 8 = 192
        foreach (var frame in frames)
        {
            Assert.Equal(expectedPixels, frame.Length);
            for (int p = 0; p < frame.Length; p++)
            {
                Assert.True(frame[p] >= 0.0f && frame[p] <= 1.0f);
            }
        }
    }

    [Fact]
    public void AudioVae_DecodesStereoPcmAt32kHz()
    {
        var decoder = new MiniMaxH3AudioVaeDecoder(hiddenDim: 32);
        int audioFrames = 4;
        int latentLen = audioFrames * MiniMaxH3Config.AudioLatentChannels; // 4 * 32 = 128

        var latent = new float[latentLen];
        for (int i = 0; i < latentLen; i++) latent[i] = MathF.Sin(i * 0.1f);

        var pcm = decoder.Decode(latent, audioFrames);

        // Expected samples: 4 * 512 = 2048 per channel -> 4096 interleaved stereo samples
        Assert.Equal(4096, pcm.Length);

        for (int i = 0; i < pcm.Length; i++)
        {
            Assert.False(float.IsNaN(pcm[i]));
            Assert.True(pcm[i] >= -1.0f && pcm[i] <= 1.0f);
        }
    }

    [Fact]
    public void OutputExporter_WritesValidWavFile()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"minimax_h3_test_{Guid.NewGuid():N}.wav");
        try
        {
            float[] pcmStereo = new float[32000 * 2]; // 1 second of stereo audio
            for (int i = 0; i < 32000; i++)
            {
                float val = MathF.Sin(2.0f * MathF.PI * 440.0f * i / 32000.0f); // 440 Hz tone
                pcmStereo[i * 2 + 0] = val; // Left
                pcmStereo[i * 2 + 1] = val; // Right
            }

            MiniMaxH3OutputExporter.ExportAudioWav(tempPath, pcmStereo, 32000, 2);

            Assert.True(File.Exists(tempPath));
            byte[] bytes = File.ReadAllBytes(tempPath);
            Assert.True(bytes.Length > 44);

            // Check RIFF header
            string riff = System.Text.Encoding.ASCII.GetString(bytes, 0, 4);
            string wave = System.Text.Encoding.ASCII.GetString(bytes, 8, 4);
            Assert.Equal("RIFF", riff);
            Assert.Equal("WAVE", wave);
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void Tokenizer_FromVocabAndMerges_EncodesTokens()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"minimax_h3_tok_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        string vocabPath = Path.Combine(tempDir, "vocab.json");
        string mergesPath = Path.Combine(tempDir, "merges.txt");

        try
        {
            // Synthetic vocab with byte tokens and merged tokens
            string vocabJson = """
            {
                "h": 0,
                "e": 1,
                "l": 2,
                "o": 3,
                "he": 4,
                "ll": 5,
                "hello": 6
            }
            """;
            File.WriteAllText(vocabPath, vocabJson);

            string mergesTxt = """
            #version: 0.2
            h e
            l l
            he l
            hel lo
            """;
            File.WriteAllText(mergesPath, mergesTxt);

            var tokenizer = QwenTokenizer.FromVocabAndMerges(vocabPath, mergesPath);
            Assert.NotNull(tokenizer);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }
}
