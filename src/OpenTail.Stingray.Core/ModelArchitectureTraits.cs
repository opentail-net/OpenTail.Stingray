namespace OpenTail.Stingray.Core;

/// <summary>The recurrent/attention layout family of a hybrid model.</summary>
public enum HybridKind
{
    /// <summary>Attention-only (or a dedicated pass handles the layout).</summary>
    None,
    /// <summary>Gated DeltaNet + attention (Qwen3.5 family).</summary>
    GatedDeltaNet,
    /// <summary>Mamba-2 selective-scan + attention (Granite-H, Nemotron-H); a layer is recurrent iff its head_count_kv entry is 0.</summary>
    Mamba2,
    /// <summary>Gated short-convolution + attention (Liquid LFM2).</summary>
    ShortConv,
}

/// <summary>
/// Structural facts about an architecture that the generic baseline parser needs while it reads the file: how layers are laid out,
/// where the head size comes from, which metadata a multimodal rope reads. They are inputs, not policy.
///
/// <para>Descriptors supply them on the normal load path (<c>ArchitectureDescriptor.Traits</c>), so a new architecture declares its
/// layout in its own file and needs no edit here. <see cref="Legacy"/> encodes the same facts by architecture name for callers that
/// bypass the registry (the direct <c>FromGgufMetadata</c> shim) and for the one architecture without a descriptor
/// (<c>qwen3vlmoe</c>). A test pins every descriptor's traits to <see cref="Legacy"/> so the two cannot drift; delete the table
/// once the shim callers are gone. Core is below Engine and cannot see descriptors, which is why traits are passed in.</para>
/// </summary>
public sealed record ModelArchitectureTraits
{
    public static readonly ModelArchitectureTraits None = new();

    public HybridKind Hybrid { get; init; } = HybridKind.None;

    /// <summary>Each layer carries exactly one sublayer (Nemotron-H: Mamba-2, attention or MLP, never a mix).</summary>
    public bool SingleSublayerBlocks { get; init; }

    /// <summary>Head size comes from <c>{arch}.wkv.head_size</c> when no attention key length is present (RWKV).</summary>
    public bool HeadDimFromWkvHeadSize { get; init; }

    /// <summary>Default for <c>expert_gating_func</c> when the file omits it (GLM-4 MoE: 2 = sigmoid).</summary>
    public int DefaultExpertGatingFunc { get; init; }

    /// <summary>MoE top-k weights are normalized unless the file says otherwise (llama.cpp's Mixtral-style llama MoE).</summary>
    public bool NormalizeMoeTopKWeightsByDefault { get; init; }

    /// <summary>Reads <c>{arch}.rope.dimension_sections</c> (multi-axis rope: Qwen2/3-VL, PaddleOCR-VL).</summary>
    public bool ReadsRopeDimensionSections { get; init; }

    /// <summary>The multi-axis rope sections are interleaved (Qwen3-VL MoE).</summary>
    public bool RopeSectionsInterleaved { get; init; }

    /// <summary>Builds the deepstack mapping from <c>{arch}.n_deepstack_layers</c> when no explicit mapping is present (Qwen3-VL).</summary>
    public bool DeepstackFromLayerCount { get; init; }

    /// <summary>
    /// The traits the parser applied before they became an input, keyed by metadata architecture name (case-sensitive, as the old
    /// literal comparisons were). Anything not listed has <see cref="None"/>.
    /// </summary>
    public static ModelArchitectureTraits Legacy(string? arch) => arch switch
    {
        "qwen35moe" => new() { Hybrid = HybridKind.GatedDeltaNet },
        "granitehybrid" => new() { Hybrid = HybridKind.Mamba2 },
        "nemotron_h" => new() { Hybrid = HybridKind.Mamba2, SingleSublayerBlocks = true },
        "lfm2" or "lfm2moe" => new() { Hybrid = HybridKind.ShortConv },
        "rwkv6" or "rwkv7" => new() { HeadDimFromWkvHeadSize = true },
        "glm4moe" => new() { DefaultExpertGatingFunc = 2 },
        "llama" => new() { NormalizeMoeTopKWeightsByDefault = true },
        "qwen2vl" or "paddleocr" => new() { ReadsRopeDimensionSections = true },
        "qwen3vl" => new() { ReadsRopeDimensionSections = true, DeepstackFromLayerCount = true },
        "qwen3vlmoe" => new() { ReadsRopeDimensionSections = true, RopeSectionsInterleaved = true, DeepstackFromLayerCount = true },
        _ => None,
    };
}
