namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// Architectural hyperparameters for the FLUX.2 (Klein &amp; Kontext) multi-reference foundation model.
/// Supports multi-image conditioning, contextual editing, and 3D Context RoPE.
/// </summary>
/// <remarks>
/// Values below are confirmed against the real downloaded checkpoint's own tensor shapes
/// (models/_models/flux2-dev-Q4_K_S.gguf, 299 tensors, via `stingray list-tensors`) and
/// cross-checked against an external report of BFL's real flux2/src/flux2/model.py source
/// (not yet independently re-verified against a vendored reference under examples/ per this
/// project's CLAUDE.md rule 8 -- do that before relying on this for numeric golden-parity work).
/// See docs/087-flux2-implementation-plan.md for the full derivation.
/// </remarks>
public sealed record Flux2Params
{
    /// <summary>Image latent channels per 2x2 patch (128 = 2x2 patch x 32 latent channels -- FLUX.2's VAE has 32 latent channels, double FLUX.1's 16; confirmed via img_in.weight [128,6144]).</summary>
    public int InChannels { get; init; } = 128;

    /// <summary>Output velocity channels (128, matches InChannels -- confirmed via final_layer.linear.weight [6144,128]).</summary>
    public int OutChannels { get; init; } = 128;

    /// <summary>FLUX.2 has no CLIP pooled-vector conditioning at all (no vector_in/y_in-style tensor in the checkpoint) -- unlike FLUX.1. Kept at 0/unused; do not wire a CLIP pooled embed for FLUX.2.</summary>
    public int VecInDim { get; init; } = 0;

    /// <summary>Sequence text embedding dimension: 3 concatenated Mistral-Small-3.2-24B-Instruct hidden-layer activations (5120 each, layers 10/20/30 per external report), NOT T5. Confirmed via txt_in.weight [15360,6144].</summary>
    public int ContextInDim { get; init; } = 15360;

    /// <summary>Transformer hidden embedding dimension D. Confirmed via img_attn.proj.weight [6144,6144] and every block tensor.</summary>
    public int HiddenSize { get; init; } = 6144;

    /// <summary>MLP expansion ratio for the gated FFN's hidden width (mlp_hidden_dim = HiddenSize * MlpRatio = 18432). Confirmed via img_mlp up-proj [6144,36864]=2x18432 and down-proj [18432,6144] (SiLU-gated: up-proj produces 2x mlp_hidden_dim, split into two 18432-wide halves, SiLU(a)*b gated down to 18432 before the down-projection) -- NOT a plain GELU MLP like FLUX.1.</summary>
    public float MlpRatio { get; init; } = 3.0f;

    /// <summary>Number of attention heads (HeadDim = 128, confirmed via key_norm.scale/query_norm.scale [128]; 6144/128=48).</summary>
    public int NumHeads { get; init; } = 48;

    /// <summary>Number of double-stream transformer blocks (confirmed by direct count: double_blocks.{0..7}).</summary>
    public int DepthDoubleBlocks { get; init; } = 8;

    /// <summary>Number of single-stream unified blocks (confirmed by direct count: single_blocks.{0..47}).</summary>
    public int DepthSingleBlocks { get; init; } = 48;

    /// <summary>4D Context RoPE axial dimensions [t, h, w, l] (frame/time, height, width, sequence-local) -- NOT FLUX.1's 3-axis [16,56,56] scheme. Confirmed against real BFL source (examples/flux2/src/flux2/model.py: `axes_dim: list[int] = [32, 32, 32, 32]`, sum=128=HeadDim). The extra "t" axis is used for multi-reference-image temporal offsetting (examples/flux2/src/flux2/sampling.py's `encode_image_refs`/`scatter_ids`); plain text-to-image (no reference images) uses a constant/zero t-coordinate.</summary>
    public int[] AxesDim { get; init; } = [32, 32, 32, 32];

    /// <summary>Base theta for RoPE frequency calculation. Confirmed against real BFL source: `theta: int = 2000` -- NOT FLUX.1's 10000.</summary>
    public float Theta { get; init; } = 2000.0f;

    /// <summary>FLUX.2's linears are bias-free -- no .bias tensor exists alongside any .qkv.weight/.proj.weight/mlp linear/final_layer.linear in the checkpoint, unlike FLUX.1.</summary>
    public bool QkvBias { get; init; } = false;

    /// <summary>Whether guidance embedding MLP is active (confirmed: guidance_in.in_layer.weight/out_layer.weight both present).</summary>
    public bool GuidanceEmbed { get; init; } = true;

    public int HeadDim => HiddenSize / Math.Max(1, NumHeads);
}
