
namespace OpenTail.Stingray.Audio.AudioGen;

/// <summary>
/// Weight loader for AudioGen's text conditioner: a real, STOCK `t5-large` checkpoint, loaded
/// SEPARATELY from `t5-large`'s own `model.safetensors` -- unlike MusicGen's bundled `t5-base`.
/// Confirmed from real source (`audiocraft.modules.conditioners.T5Conditioner.__init__`,
/// pip-installed and read directly 2026-09-02): `finetune=False` for the released checkpoints
/// means `self.t5` is explicitly excluded from the module's own `state_dict` (`self.__dict__['t5']
/// = t5.to(device)`, bypassing `nn.Module`'s parameter registration) -- so AudioGen's real LM
/// checkpoint genuinely never contains T5 weights at all, confirmed independently by their
/// absence from the real `state_dict.bin`'s key list. Standard stock T5 tensor names, no prefix.
/// </summary>
public static class AudioGenTextEncoderWeights
{
    public static readonly T5EncoderDims Dims = new(
        DModel: AudioGenConfig.TextDModel,
        DFf: AudioGenConfig.TextDFf,
        DKv: AudioGenConfig.TextDKv,
        NumLayers: AudioGenConfig.TextNumLayers,
        NumHeads: AudioGenConfig.TextNumHeads,
        RelativeAttentionNumBuckets: AudioGenConfig.TextRelativeAttentionNumBuckets,
        RelativeAttentionMaxDistance: AudioGenConfig.TextRelativeAttentionMaxDistance,
        LayerNormEps: AudioGenConfig.TextLayerNormEps);

    public static NonGatedT5EncoderWeights Load(SafetensorsLoader loader) => T5EncoderKernels.Load(loader, Dims, prefix: "");
}
