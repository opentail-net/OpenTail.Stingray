namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// One-off numeric isolation test for the Qwen3 Forced Aligner's real layer-2 std jump
/// (docs/audio-review-progress.md, 2026-09-07 "layer-2 jump localized to the MLP block"
/// entries): computes layer 2's SwiGLU MLP BY HAND from the real checkpoint's raw Safetensors
/// weights, fed the REAL `post_attn` input dumped directly from the reference binary
/// (`STINGRAY_FA_DUMP_DIR`, a real additive debug-dump mechanism added to
/// `examples/audio.cpp/src/models/qwen3_asr/thinker.cpp` this session), and compares against the
/// reference's own real `out` (post-MLP, post-residual) tensor, ALSO dumped from the same run --
/// bypassing this port's `ForwardPass` graph entirely to test the raw SwiGLU formula in isolation.
///
/// Reads from an absolute scratch-directory path (not the repo), produced by manually running:
/// <c>STINGRAY_FA_TRACE=1 STINGRAY_FA_DUMP_DIR=&lt;dir&gt; audiocpp_cli.exe --task align --family
/// qwen3_forced_aligner --model models/qwen3-forcedaligner --model-spec-override
/// examples/audio.cpp/model_specs/qwen3_forced_aligner.json --backend cpu --audio
/// examples/audio.cpp/assets/resources/a.wav --text "..." --language English</c> -- skips if
/// that directory/those files aren't present (this is a manual investigation aid, not a routine
/// CI-run test).
/// </summary>
public sealed class QwenForcedAlignerLayer2MlpIsolationDebugTest : HeavyTestBase
{
    private const string DumpDir = "C:/Users/Dmitri/AppData/Local/Temp/claude/C--Git-Public/e20d4458-3906-46e3-a5d1-af57aa8bcc0d/scratchpad/fa_dump";

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

    private static float[] ReadRawF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    [Fact]
    public void HandComputedLayer2Mlp_OnRealDump_MatchesOrDivergesFromRealReferenceOutput()
    {
        string postAttnPath = Path.Combine(DumpDir, "layer_2_post_attn.bin");
        string outPath = Path.Combine(DumpDir, "layer_2_out.bin");
        Assert.SkipUnless(File.Exists(postAttnPath) && File.Exists(outPath), "real reference dump not found -- see class doc comment for how to regenerate");

        string? safetensorsPath = FindRepoFile("models/qwen3-forcedaligner/model.safetensors");
        Assert.SkipUnless(safetensorsPath != null, "models/qwen3-forcedaligner/model.safetensors not found");

        const int hiddenDim = 1024;
        const float eps = 1e-6f;

        var postAttn = ReadRawF32(postAttnPath);
        var realOut = ReadRawF32(outPath);
        Assert.Equal(postAttn.Length, realOut.Length);
        int steps = postAttn.Length / hiddenDim;
        Assert.Equal(0, postAttn.Length % hiddenDim);

        using var loader = SafetensorsLoader.Open(safetensorsPath!);
        var normWeight = loader.ReadF32("thinker.model.layers.2.post_attention_layernorm.weight");
        var gateWeight = loader.ReadF32("thinker.model.layers.2.mlp.gate_proj.weight");
        var upWeight = loader.ReadF32("thinker.model.layers.2.mlp.up_proj.weight");
        var downWeight = loader.ReadF32("thinker.model.layers.2.mlp.down_proj.weight");
        int[] gateShape = loader.GetShape("thinker.model.layers.2.mlp.gate_proj.weight"); // [ffDim, hiddenDim]
        int ffDim = gateShape[0];
        Assert.Equal(hiddenDim, gateShape[1]);

        var handComputedOut = new float[postAttn.Length];
        for (int t = 0; t < steps; t++)
        {
            var x = postAttn.AsSpan(t * hiddenDim, hiddenDim);

            // Real RMSNorm.
            double sumSq = 0;
            for (int i = 0; i < hiddenDim; i++) sumSq += (double)x[i] * x[i];
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / hiddenDim + eps));
            var normed = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) normed[i] = x[i] * invRms * normWeight[i];

            // Real SwiGLU: down(silu(gate(normed)) * up(normed)).
            var gate = new float[ffDim];
            var up = new float[ffDim];
            for (int o = 0; o < ffDim; o++)
            {
                float g = 0f, u = 0f;
                int wBase = o * hiddenDim;
                for (int i = 0; i < hiddenDim; i++)
                {
                    g += gateWeight[wBase + i] * normed[i];
                    u += upWeight[wBase + i] * normed[i];
                }
                float silu = g / (1f + MathF.Exp(-g));
                gate[o] = silu;
                up[o] = u;
            }
            var gated = new float[ffDim];
            for (int i = 0; i < ffDim; i++) gated[i] = gate[i] * up[i];

            var down = new float[hiddenDim];
            for (int o = 0; o < hiddenDim; o++)
            {
                float sum = 0f;
                int wBase = o * ffDim;
                for (int i = 0; i < ffDim; i++) sum += downWeight[wBase + i] * gated[i];
                down[o] = sum;
            }

            for (int i = 0; i < hiddenDim; i++) handComputedOut[t * hiddenDim + i] = x[i] + down[i];
        }

        double sumSqDiff = 0, sumSqReal = 0;
        for (int i = 0; i < realOut.Length; i++)
        {
            double d = handComputedOut[i] - realOut[i];
            sumSqDiff += d * d;
            sumSqReal += (double)realOut[i] * realOut[i];
        }
        double relError = Math.Sqrt(sumSqDiff / Math.Max(1e-12, sumSqReal));
        Console.Error.WriteLine($"[FA-MLP-Isolation] hand-computed vs real reference relative L2 error = {relError:F6}");

        // No hard assertion on the error magnitude -- this test's real purpose is to PRINT the
        // relative error for manual inspection (a near-zero error means this port's SwiGLU
        // formula matches the reference exactly on the real input, isolating the bug to
        // upstream of the MLP; a large error means the formula/weight-loading itself is wrong).
        Assert.True(double.IsFinite(relError));
    }
}
