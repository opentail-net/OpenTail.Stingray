
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Weight loader for MusicGen's text conditioning encoder. **Correction, 2026-09-02**: the
/// initial version of this file assumed the text encoder had to be loaded from a SEPARATE stock
/// `t5-base` checkpoint (reasoning: HF's `MusicgenForConditionalGeneration` documentation
/// describes composing sub-models via `AutoModel.from_pretrained`). Real inspection of
/// musicgen-small's own `model.safetensors` header disproved that: it contains a full,
/// self-contained `text_encoder.*` tensor tree (`text_encoder.shared.weight`,
/// `text_encoder.encoder.block.{i}.*`, `text_encoder.encoder.final_layer_norm.weight`) --
/// EXACTLY the same "bundled, not composed" convention Parler-TTS's
/// <see cref="Parler.T5EncoderWeights"/> already documented. Loads from the SAME loader as
/// <see cref="MusicGenTransformerWeights"/> (musicgen-small's single checkpoint file), not a
/// separate `t5-base` download.
///
/// <para><b>DRY pass, 2026-09-02</b>: now a thin wrapper over the shared, dimension-parameterized
/// <see cref="Primitives.T5EncoderKernels"/> (extracted once AudioGen's external, frozen
/// `t5-large` conditioner turned out to need the byte-for-byte identical non-gated T5 algorithm,
/// just with different dims) -- real t5-base is NOT gated (`is_gated_act: false`,
/// `feed_forward_proj: "relu"`, one `wi` matrix), see <see cref="Primitives.T5EncoderKernels"/>'s
/// doc comment for the shared math; do not reuse <see cref="Parler.T5Encoder"/>'s gated-GELU FFN
/// here.</para>
/// </summary>
public static class MusicGenTextEncoderWeights
{
    public static readonly T5EncoderDims Dims = new(
        DModel: MusicGenConfig.TextDModel,
        DFf: MusicGenConfig.TextDFf,
        DKv: MusicGenConfig.TextDKv,
        NumLayers: MusicGenConfig.TextNumLayers,
        NumHeads: MusicGenConfig.TextNumHeads,
        RelativeAttentionNumBuckets: MusicGenConfig.TextRelativeAttentionNumBuckets,
        RelativeAttentionMaxDistance: MusicGenConfig.TextRelativeAttentionMaxDistance,
        LayerNormEps: MusicGenConfig.TextLayerNormEps);

    public static NonGatedT5EncoderWeights Load(SafetensorsLoader loader) => T5EncoderKernels.Load(loader, Dims, prefix: "text_encoder.");
}
