namespace OpenTail.Stingray.Engine;

/// <summary>
/// The qwen4exp MTP draft head (Phase 4). It ships as a separate GGUF with one extra transformer block (index
/// <c>NumLayers</c>), its own input projection and HC head. The shared variant borrows the token embedding and the output
/// projection from the trunk. Tensor names and shapes follow the reference graph_mtp (qwen4exp.cpp ~529-612).
/// </summary>
public sealed class Qwen4ExpMtpHead
{
    /// <summary>The head's transformer block (full attention, routed MoE, HC mixers).</summary>
    public Qwen4ExpLayerTensors Layer { get; }

    /// <summary>Projection of concat(enorm(token), hnorm(hidden)): [2 * embd, embd].</summary>
    public Qwen4ExpTensorRef EhProj { get; }
    public Qwen4ExpTensorRef EnNorm { get; }
    public Qwen4ExpTensorRef HNorm { get; }

    /// <summary>The head's own output mixer, read before the shared output projection.</summary>
    public Qwen4ExpTensorRef HcHeadNorm { get; }
    public Qwen4ExpTensorRef HcHeadDown { get; }
    public Qwen4ExpTensorRef HcHeadUp { get; }

    /// <summary>Shared with the trunk (the shared head files do not carry them).</summary>
    public Qwen4ExpTensorRef TokEmbd { get; }
    public Qwen4ExpTensorRef Output { get; }

    /// <summary>Index of the head block: the trunk's block count.</summary>
    public int BlockIndex { get; }

    private Qwen4ExpMtpHead(
        int blockIndex, Qwen4ExpLayerTensors layer, Qwen4ExpTensorRef ehProj, Qwen4ExpTensorRef enNorm,
        Qwen4ExpTensorRef hNorm, Qwen4ExpTensorRef hcHeadNorm, Qwen4ExpTensorRef hcHeadDown,
        Qwen4ExpTensorRef hcHeadUp, Qwen4ExpTensorRef tokEmbd, Qwen4ExpTensorRef output)
    {
        BlockIndex = blockIndex;
        Layer = layer;
        EhProj = ehProj;
        EnNorm = enNorm;
        HNorm = hNorm;
        HcHeadNorm = hcHeadNorm;
        HcHeadDown = hcHeadDown;
        HcHeadUp = hcHeadUp;
        TokEmbd = tokEmbd;
        Output = output;
    }

    /// <summary>
    /// Loads the head from an MTP GGUF. <paramref name="trunk"/> supplies the shared token embedding and output.
    /// Throws when a required tensor is missing or has an unexpected shape.
    /// </summary>
    public static Qwen4ExpMtpHead Load(GgufModel mtpModel, Qwen4ExpTensorSet trunk, int blockIndex, int embedDim, int hc)
    {
        ArgumentNullException.ThrowIfNull(mtpModel);
        ArgumentNullException.ThrowIfNull(trunk);

        var layer = Qwen4ExpTensorSet.LoadLayer(mtpModel, blockIndex, isRecr: false, isPle: false);
        var ehProj = Qwen4ExpTensorSet.Require(mtpModel, $"blk.{blockIndex}.nextn.eh_proj.weight");
        var enNorm = Qwen4ExpTensorSet.Require(mtpModel, $"blk.{blockIndex}.nextn.enorm.weight");
        var hNorm = Qwen4ExpTensorSet.Require(mtpModel, $"blk.{blockIndex}.nextn.hnorm.weight");
        var hcHeadNorm = Qwen4ExpTensorSet.Require(mtpModel, $"blk.{blockIndex}.nextn.hc_head_norm.weight");
        var hcHeadDown = Qwen4ExpTensorSet.Require(mtpModel, $"blk.{blockIndex}.nextn.hc_head_down.weight");
        var hcHeadUp = Qwen4ExpTensorSet.Require(mtpModel, $"blk.{blockIndex}.nextn.hc_head_up.weight");

        int hcDim = hc * embedDim;
        int lowRank = (int)hcHeadDown.Info.Dimensions[1];
        Expect(ehProj, 2L * embedDim, embedDim);
        Expect(enNorm, embedDim);
        Expect(hNorm, hcDim);
        Expect(hcHeadNorm, hcDim);
        Expect(hcHeadDown, hcDim, lowRank);
        Expect(hcHeadUp, lowRank, hcDim);

        return new Qwen4ExpMtpHead(
            blockIndex, layer, ehProj, enNorm, hNorm, hcHeadNorm, hcHeadDown, hcHeadUp,
            trunk.TokEmbd, trunk.Output);
    }

    private static void Expect(Qwen4ExpTensorRef t, params long[] dims)
    {
        for (int i = 0; i < dims.Length; i++)
        {
            long actual = i < t.Info.Dimensions.Length ? (long)t.Info.Dimensions[i] : 1;
            if (actual != dims[i])
                throw new InvalidOperationException(
                    $"MTP tensor '{t.Name}' dimension {i} is {actual}, expected {dims[i]}.");
        }
    }
}
