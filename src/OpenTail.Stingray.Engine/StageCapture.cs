namespace OpenTail.Stingray.Engine;

/// <summary>
/// Diagnostic sink for intra-layer activation snapshots, written by BOTH the CPU
/// <see cref="ForwardPass"/> and the Vulkan <see cref="GpuForwardPass"/> at identically named
/// points so the two can be compared stage by stage.
///
/// <para>
/// Hidden-state taps (<see cref="Core.IForwardPass.EnableHiddenTaps"/>) already localise a
/// cross-backend divergence to a layer. This localises it <i>within</i> a layer, which is the
/// granularity at which the answer is a specific dispatch rather than a suspect list. The
/// existing <c>STINGRAY_GEMMA4_PROBE</c> stages report magnitudes for the Vulkan side only;
/// magnitudes cannot distinguish "slightly different" from "pointing elsewhere", and a one-sided
/// dump cannot be diffed at all.
/// </para>
///
/// <para>
/// Off by default and never allocated in normal use. Capture is inherently expensive on the GPU
/// path — each snapshot has to close and reopen the command buffer to make the download
/// host-visible — so this is a debugging tool, not something to leave enabled.
/// </para>
///
/// <para>
/// Not thread-safe: single-sequence, single-threaded diagnosis only. This project's
/// <c>xunit.runner.json</c> sets <c>parallelizeTestCollections=false</c>, so a test that enables
/// it cannot race another.
/// </para>
/// </summary>
internal static class StageCapture
{
    /// <summary>Names of the snapshot points, shared so the two backends cannot drift apart.</summary>
    internal static class Stages
    {
        /// <summary>Token embedding after any embedding scale, before layer 0.</summary>
        public const string Embed = "embed";

        /// <summary>Normalised input to the attention block (pre-attention RMSNorm output).</summary>
        public const string AttnNorm = "attn_norm";

        /// <summary>V after its projection, before Gemma 4's plain per-head V norm.</summary>
        public const string VProj = "v_proj";

        /// <summary>V after Gemma 4's plain per-head RMSNorm — what actually enters the KV cache.</summary>
        public const string VNorm = "v_norm";

        /// <summary>Attention output before the output projection. At position 0 this must equal V.</summary>
        public const string AttnOut = "attn_out";

        /// <summary>Hidden straight after the attention output projection, before the post-attn norm.</summary>
        public const string OProj = "o_proj";

        /// <summary>Hidden after the attention output projection, post-attn norm and residual add.</summary>
        public const string PostAttnResidual = "post_attn_resid";

        /// <summary>Hidden after the FFN and its residual add, before PLE injection.</summary>
        public const string PostFfnResidual = "post_ffn_resid";

        /// <summary>Hidden after Gemma 4 per-layer-embedding injection.</summary>
        public const string PostPle = "post_ple";

        /// <summary>Hidden after the per-layer output scale — the layer's final output.</summary>
        public const string LayerOutput = "layer_out";
    }

    /// <summary>Enables recording. Callers must reset <see cref="Records"/> themselves.</summary>
    public static bool Enabled { get; set; }

    /// <summary>Everything captured since the last <see cref="Reset"/>, in execution order.</summary>
    public static List<(string Backend, int Layer, string Stage, float[] Data)> Records { get; } = [];

    public static void Reset()
    {
        Records.Clear();
        Enabled = false;
    }

    /// <summary>
    /// Record one snapshot. <paramref name="layer"/> is -1 for pre-trunk stages such as
    /// <see cref="Stages.Embed"/>.
    /// </summary>
    public static void Record(string backend, int layer, string stage, ReadOnlySpan<float> data)
    {
        if (!Enabled) return;
        Records.Add((backend, layer, stage, data.ToArray()));
    }

    /// <summary>The recorded vector for one (backend, layer, stage), or null when absent.</summary>
    public static float[]? Find(string backend, int layer, string stage)
    {
        foreach (var r in Records)
            if (r.Layer == layer && r.Stage == stage && r.Backend == backend)
                return r.Data;
        return null;
    }
}
