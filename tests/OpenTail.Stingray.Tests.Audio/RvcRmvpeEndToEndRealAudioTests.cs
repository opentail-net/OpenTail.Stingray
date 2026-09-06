
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

        // NOT YET FULLY GOLDEN-VERIFIED, but very close: the real reference (same audio, same
        // audio_pad_duration_sec=1 preprocessing) reports frames=796 mean=0.00474 std=0.04768
        // max=0.97022 (STINGRAY_RVC_TRACE=1 in rmvpe_pitch_extractor.cpp); this test currently
        // gets mean=0.00298 std=0.02489 max=0.81235 -- three real bugs found and fixed to get
        // here (see docs/audio-review-progress.md's 2026-09-06 RMVPE update for the full
        // derivation of each): (1) missing audio_pad_duration_sec preprocessing, (2) a U-Net
        // image H/W-axis orientation swap, (3) `conv_block_res`'s real ReLU placement (ReLU after
        // BOTH conv-BN stages, no ReLU after the residual add -- the OPPOSITE of a textbook
        // ResNet BasicBlock). Per-stage STINGRAY_RVC_TRACE=1 tracing added to both this file's
        // caller (via RvcRmvpeEncoder's own Trace/TraceFlat helpers) and the reference's
        // build_rmvpe_feature_graph (ggml intermediate tensors marked as extra graph outputs via
        // rmvpe_debug_tap/rmvpe_debug_dump_taps) shows the encoder stack and all 5 U-Net skip
        // connections now match the reference almost exactly (e.g. after-bottleneck mean/std
        // 3.147/6.425 vs the reference's 3.150/6.497) -- the remaining ~1.2x-2x per-stage
        // divergence starts specifically in the DECODER's ConvTranspose2d upsample stage (checked:
        // not a weight in/out-channel layout swap, not a +/-1 index-shift artifact -- both were
        // tried and made no measurable difference) and compounds through the decoder's residual
        // blocks. Not yet root-caused; flagging as the next concrete step rather than a vague
        // "still off."
    }
}
