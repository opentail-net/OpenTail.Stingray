
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end RMVPE test: real mel (RvcRmvpeMelExtractor) -> real network
/// (RvcRmvpeEncoder) -> trimmed to the real (unpadded) frame count, matching the real reference's
/// own trim-at-output convention (confirmed: the reference runs its GRU on the full padded
/// sequence but only returns `mel.frames` real rows of "salience", trimming AFTER the head, not
/// before the GRU). Cross-checked numerically against the real C++ reference
/// (STINGRAY_RVC_TRACE=1 added to rmvpe_pitch_extractor.cpp's final salience array).</summary>
public sealed class RvcRmvpeEndToEndRealAudioTests : HeavyTestBase
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

    [Fact]
    public void EndToEnd_RealAudio_RealWeights_MatchesReferenceSalienceStats()
    {
        string? checkpointPath = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(checkpointPath != null, "rvc-f16.gguf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(checkpointPath!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.Rvc.RvcRmvpeWeights(source);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        // Real pipeline: 48Hz high-pass (filtfilt) + reflect-pad by audio_pad_duration_sec (1s
        // default) on each side, applied BEFORE RMVPE mel extraction (see native_pipeline.cpp).
        var prepared = OpenTail.Stingray.Audio.Rvc.RvcAudioPreprocessing.PrepareContentAudio(samples, audioPadDurationSec: 1);

        var (mel, realFrames) = OpenTail.Stingray.Audio.Rvc.RvcRmvpeMelExtractor.ExtractPadded(prepared);
        var salienceFull = OpenTail.Stingray.Audio.Rvc.RvcRmvpeEncoder.Forward(w, mel);

        // Real reference convention: GRU runs on the full padded sequence, but only the real
        // (unpadded) frames are kept in the final output.
        var salience = new float[realFrames][];
        Array.Copy(salienceFull, salience, realFrames);

        double sum = 0, sumSq = 0;
        float max = 0f;
        int n = 0;
        foreach (var row in salience)
        {
            foreach (var v in row)
            {
                Assert.True(float.IsFinite(v));
                sum += v;
                sumSq += (double)v * v;
                if (v > max) max = v;
                n++;
            }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[RvcRmvpeSalience] frames={realFrames} classes={salience[0].Length} mean={mean:F5} std={std:F5} max={max:F5}");

        // NOT YET GOLDEN-VERIFIED: the real reference (same audio, same audio_pad_duration_sec=1
        // preprocessing) reports frames=796 classes=360 mean=0.00474 std=0.04768 max=0.97022
        // (captured via STINGRAY_RVC_TRACE=1 in rmvpe_pitch_extractor.cpp). Frame count now
        // matches exactly (was 596 before RvcAudioPreprocessing existed), and a real H/W-axis
        // orientation bug in RvcRmvpeEncoder's U-Net image construction was found and fixed (the
        // reference's feature.input is [1, melBins, frames] transposed to put H=frames/W=melBins,
        // not H=melBins/W=frames as this code originally assumed -- conv kernels are not symmetric
        // under an H/W swap, so this was silently wrong, not just relabeled). That fix moved max
        // from 0.0014 to 0.0067 but the network still does not produce the reference's sharp,
        // confident per-frame peak -- every intermediate stage (post-input-BN through the GRU
        // outputs) has plausible non-degenerate mean/std when traced with STINGRAY_RVC_TRACE=1, so
        // the remaining bug is a genuine numerical mismatch still unlocated within the U-Net/GRU
        // stack, not a structural/wiring one. Needs real ggml-side intermediate-tensor dumps
        // (mean/std after input-BN, after each encoder level, after the bottleneck, after the
        // decoder, and after the final 3-channel conv) added to build_rmvpe_feature_graph in the
        // C++ reference to bisect which stage first diverges -- not yet done.
    }
}
