
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Loaded Llama 4 (E4B Scout/Maverick) vision transformer projector
/// (<c>clip.vision.projector_type=llama4</c>). Ported from <c>llama.cpp</c>'s
/// <c>clip_graph_llama4::build()</c> (<c>tools/mtmd/models/llama4.cpp</c>) plus the shared
/// <c>clip_graph::build_vit</c>/<c>build_attn</c>/<c>build_ffn</c>/<c>build_rope_2d</c> helpers.
/// Distinctive relative to <see cref="Gemma4VVisionModel"/>/<see cref="Gemma3VisionModel"/>: a
/// real [CLS] token concatenated after patch embedding (dropped again before the merger), a
/// learned absolute position table AND 2D half-RoPE active simultaneously (not either/or), a
/// pixel-shuffle (space-to-depth) merger instead of average-pooling, and a rope pairing
/// convention that is NORM/interleaved (<c>clip_graph::build_rope_2d</c>'s default), NOT the NEOX
/// split-half convention <see cref="Gemma4VVisionEncoder"/> uses -- confirmed directly from
/// <c>clip.cpp</c>: <c>gemma4v.cpp</c> hand-rolls its own NEOX-mode rope function
/// ("similar to build_rope_2d, but use neox ordering"), while <c>llama4.cpp</c> calls the shared
/// <c>build_rope_2d</c> as-is, which issues <c>ggml_rope_ext</c> with mode 0 (interleaved pairs)
/// on each half-of-head-dim slice. See docs/06-llama4-vision-plan.md.
/// </summary>
/// <remarks>
/// This type is the validated ownership and tensor-name boundary for the Llama 4 ViT, same
/// caveat as <see cref="Gemma4VVisionModel"/>: it does not claim decoder splice or image-token
/// mask semantics are implemented; multi-tile ("llava-uhd") preprocessing is also out of scope --
/// this encoder processes one fixed-square tile per <see cref="Llama4VisionEncoder.Forward"/> call.
/// </remarks>
public sealed class Llama4VisionModel : IDisposable
{
    public const string ProjectorType = "llama4";

    /// <summary>2D-RoPE frequency base, hardcoded per <c>clip.cpp</c>'s
    /// <c>PROJECTOR_TYPE_LLAMA4</c> case (<c>hparams.rope_theta = 10000.0f</c>) -- NOT the same
    /// value as gemma4v's (100f) and NOT read from mmproj metadata.</summary>
    public const float RopeTheta = 10000f;

    private readonly GgufModel _gguf;
    private bool _disposed;

    private Llama4VisionModel(GgufModel gguf) => _gguf = gguf;

    public required int ImageSize { get; init; }
    public required int PatchSize { get; init; }
    public required int EmbeddingLength { get; init; }
    public required int FeedForwardLength { get; init; }
    public required int BlockCount { get; init; }
    public required int HeadCount { get; init; }
    public required float LayerNormEps { get; init; }

    /// <summary>Pixel-shuffle scale factor (<c>clip.vision.projector.scale_factor</c>, required --
    /// <c>llama4.cpp</c>'s merger asserts <c>scale_factor &gt; 0</c> with no code-level default).</summary>
    public required int NMerge { get; init; }

    /// <summary>Projection output width, derived from the merger tensor chain
    /// (<see cref="MmModelProj"/>'s output dim), not a separate metadata key.</summary>
    public required int ProjectionDim { get; init; }

    /// <summary>FFN activation for the per-block gated FFN, from <c>clip.use_gelu</c>/
    /// <c>clip.use_silu</c> (defaults to quick-GELU if neither is set, per <c>clip.cpp</c>'s
    /// shared hparam prologue -- the same mechanism <see cref="Gemma4VVisionModel"/> uses).
    /// The merger's own 2-layer MLP (<see cref="Llama4VisionEncoder"/>) always uses plain GELU
    /// regardless of this value -- <c>Llama4VisionMLP2</c> hardcodes it.</summary>
    public required Llama4FfnActivation FfnActivation { get; init; }

    public required float[] ImageMean { get; init; }
    public required float[] ImageStd { get; init; }

    internal GgufModel Gguf => _gguf;
    internal GgufTensorInfo PatchEmbdWeight { get; init; }
    internal GgufTensorInfo? PatchEmbdBias { get; init; }
    internal GgufTensorInfo ClassEmbedding { get; init; }
    internal GgufTensorInfo PositionEmbedding { get; init; }
    internal GgufTensorInfo? PreLnWeight { get; init; }
    internal GgufTensorInfo? PreLnBias { get; init; }
    internal GgufTensorInfo? PostLnWeight { get; init; }
    internal GgufTensorInfo? PostLnBias { get; init; }
    internal GgufTensorInfo MmModelMlp1Weight { get; init; }
    internal GgufTensorInfo MmModelMlp2Weight { get; init; }
    internal GgufTensorInfo MmModelProj { get; init; }
    internal Llama4BlockWeights[] Blocks { get; init; } = [];

    /// <summary>
    /// Opens and validates the complete tensor inventory for a Llama 4 <c>llama4</c> ViT vision
    /// encoder. Strict-by-construction, same rationale as the other two vision loaders in this
    /// codebase: a partial or differently-shaped export must fail at load time, not produce
    /// plausible-but-wrong image embeddings later. Biases and pre/post-layernorm are treated as
    /// genuinely optional (not required-then-guessed), matching <c>clip.cpp</c>'s own
    /// <c>get_tensor(name, /*required=*/false)</c> loading discipline for these tensors.
    /// </summary>
    public static Llama4VisionModel Open(string mmprojPath)
    {
        var gguf = GgufModel.Open(mmprojPath);
        try
        {
            if (gguf.GetMetadata("general.architecture", "") != "clip")
                throw new NotSupportedException($"'{mmprojPath}' is not a clip mmproj.");
            // NOTE: this mmproj declares the key as "clip.projector_type" (no "vision." segment),
            // same surprise as Gemma3VisionModel -- NOT "clip.vision.projector_type" like the
            // gemma4v/gemma4uv mmprojs use. Confirmed directly against the real file (list-metadata).
            if (gguf.GetMetadata("clip.projector_type", "") != ProjectorType)
                throw new NotSupportedException(
                    $"'{mmprojPath}' is not a {ProjectorType} vision projector.");
            if (!gguf.GetMetadata("clip.has_vision_encoder", false))
                throw new NotSupportedException($"'{mmprojPath}' declares no vision encoder.");

            var blockCount = RequiredPositiveInt("clip.vision.block_count");
            var imageSize = RequiredPositiveInt("clip.vision.image_size");
            var patchSize = RequiredPositiveInt("clip.vision.patch_size");
            var embeddingLength = RequiredPositiveInt("clip.vision.embedding_length");
            var feedForwardLength = RequiredPositiveInt("clip.vision.feed_forward_length");
            var headCount = RequiredPositiveInt("clip.vision.attention.head_count");
            var layerNormEps = RequiredPositiveFiniteFloat("clip.vision.attention.layer_norm_epsilon");
            if (embeddingLength % headCount != 0)
                throw new InvalidDataException(
                    $"mmproj '{mmprojPath}' has invalid ViT geometry: embedding={embeddingLength} " +
                    $"is not divisible by heads={headCount}.");
            if (imageSize % patchSize != 0)
                throw new InvalidDataException(
                    $"mmproj '{mmprojPath}' has image_size={imageSize}, which is not divisible by patch_size={patchSize}.");

            var nMerge = RequiredPositiveInt("clip.vision.projector.scale_factor");
            var gridSize = imageSize / patchSize;
            if (gridSize % nMerge != 0)
                throw new InvalidDataException(
                    $"mmproj '{mmprojPath}' has grid={gridSize} not divisible by scale_factor={nMerge} " +
                    "-- llama4.cpp's pixel-shuffle merger requires an exact divide.");

            var imageMean = FloatArray("clip.vision.image_mean", requireNonZero: false);
            var imageStd = FloatArray("clip.vision.image_std", requireNonZero: true);

            var useGelu = gguf.GetMetadata("clip.use_gelu", false);
            var useSilu = gguf.GetMetadata("clip.use_silu", false);
            if (useGelu && useSilu)
                throw new InvalidDataException($"mmproj '{mmprojPath}' sets both clip.use_gelu and clip.use_silu.");
            var ffnActivation = useGelu ? Llama4FfnActivation.Gelu
                : useSilu ? Llama4FfnActivation.Silu
                : Llama4FfnActivation.GeluQuick;

            GgufTensorInfo Required(string name) => gguf.FindTensor(name)
                ?? throw new InvalidDataException($"mmproj '{mmprojPath}' is missing tensor '{name}'.");
            GgufTensorInfo? Optional(string name) => gguf.FindTensor(name);

            var blocks = new Llama4BlockWeights[blockCount];
            for (var i = 0; i < blocks.Length; i++)
            {
                var prefix = $"v.blk.{i}.";
                var attnQ = Required(prefix + "attn_q.weight");
                var attnK = Required(prefix + "attn_k.weight");
                var attnV = Required(prefix + "attn_v.weight");
                var attnOut = Required(prefix + "attn_out.weight");
                var ffnUp = Required(prefix + "ffn_up.weight");
                var ffnDown = Required(prefix + "ffn_down.weight");
                var ffnGate = Optional(prefix + "ffn_gate.weight");
                blocks[i] = new Llama4BlockWeights(
                    Required(prefix + "ln1.weight"), Optional(prefix + "ln1.bias"),
                    Required(prefix + "ln2.weight"), Optional(prefix + "ln2.bias"),
                    attnQ, Optional(prefix + "attn_q.bias"),
                    attnK, Optional(prefix + "attn_k.bias"),
                    attnV, Optional(prefix + "attn_v.bias"),
                    attnOut, Optional(prefix + "attn_out.bias"),
                    ffnGate, ffnGate is null ? null : Optional(prefix + "ffn_gate.bias"),
                    ffnUp, Optional(prefix + "ffn_up.bias"),
                    ffnDown, Optional(prefix + "ffn_down.bias"));

                RequireVector(blocks[i].Ln1W, embeddingLength);
                RequireVector(blocks[i].Ln2W, embeddingLength);
                RequireMatrix(blocks[i].AttnQ, embeddingLength, embeddingLength, DType.Float16);
                RequireMatrix(blocks[i].AttnK, embeddingLength, embeddingLength, DType.Float16);
                RequireMatrix(blocks[i].AttnV, embeddingLength, embeddingLength, DType.Float16);
                RequireMatrix(blocks[i].AttnOut, embeddingLength, embeddingLength, DType.Float16);
                RequireMatrix(blocks[i].FfnUp, embeddingLength, feedForwardLength, DType.Float16);
                RequireMatrix(blocks[i].FfnDown, feedForwardLength, embeddingLength, DType.Float16);
                if (blocks[i].FfnGate is { } fg) RequireMatrix(fg, embeddingLength, feedForwardLength, DType.Float16);
            }

            var patchEmbdWeight = Required("v.patch_embd.weight");
            // Stored as a flat 2D [patchSize*patchSize*3, embeddingLength] matrix (Float16), NOT
            // the 4D [patchSize,patchSize,3,embd] Float32 shape gemma4v/gemma3 use -- confirmed
            // directly against the real mmproj (list-tensors). Llama4UnfoldConvolution reshapes it
            // to 4D in-graph for im2col, but the on-disk/loaded shape is already flat.
            RequireMatrix(patchEmbdWeight, patchSize * patchSize * 3, embeddingLength, DType.Float16);
            var patchEmbdBias = Optional("v.patch_embd.bias");

            var classEmbedding = Required("v.class_embd");
            RequireVector(classEmbedding, embeddingLength);
            var expectedPatches = gridSize * gridSize;
            var positionEmbedding = Required("v.position_embd.weight");
            // n_pos = n_patches + 1 for [CLS]; added directly (ggml_add) inside build_vit, no
            // gather -- same direct-add scheme as Gemma3VisionModel's single table, just one row
            // longer for the CLS slot.
            RequireShapeAndDType(positionEmbedding, [embeddingLength, expectedPatches + 1], DType.Float32);

            var preLnW = Optional("v.pre_ln.weight");
            var preLnB = Optional("v.pre_ln.bias");
            var postLnW = Optional("v.post_ln.weight");
            var postLnB = Optional("v.post_ln.bias");

            // Merger tail: mm.model.mlp.{1,2}.weight (no bias -- Llama4VisionMLP2 is bias-free)
            // then mm.model.fc.weight (final projector, also no bias). Widths are NOT re-derived
            // from a separate "projection_dim" metadata key (llama4 doesn't declare one) -- they
            // are validated by chaining each tensor's own shape into the next, self-consistently.
            var mlp1 = Required("mm.model.mlp.1.weight");
            if (mlp1.NDimensions != 2 || mlp1.DType != DType.Float16)
                throw new InvalidDataException($"mmproj tensor '{mlp1.Name}' has shape/dtype unexpected for a 2D F16 merger weight.");
            var pixelShuffleWidth = embeddingLength * nMerge * nMerge;
            if (mlp1.Dimensions[0] != pixelShuffleWidth)
                throw new InvalidDataException(
                    $"mmproj tensor '{mlp1.Name}' has input width {mlp1.Dimensions[0]}; expected {pixelShuffleWidth} " +
                    $"(embeddingLength={embeddingLength} * nMerge^2={nMerge * nMerge}).");
            var mlpHidden = (int)mlp1.Dimensions[1];

            var mlp2 = Required("mm.model.mlp.2.weight");
            if (mlp2.NDimensions != 2 || mlp2.DType != DType.Float16)
                throw new InvalidDataException($"mmproj tensor '{mlp2.Name}' has shape/dtype unexpected for a 2D F16 merger weight.");
            if (mlp2.Dimensions[0] != mlpHidden)
                throw new InvalidDataException(
                    $"mmproj tensor '{mlp2.Name}' has input width {mlp2.Dimensions[0]}; expected mlp.1's output width {mlpHidden}.");
            var mlpOut = (int)mlp2.Dimensions[1];

            var proj = Required("mm.model.fc.weight");
            if (proj.NDimensions != 2 || proj.DType != DType.Float16)
                throw new InvalidDataException($"mmproj tensor '{proj.Name}' has shape/dtype unexpected for a 2D F16 merger weight.");
            if (proj.Dimensions[0] != mlpOut)
                throw new InvalidDataException(
                    $"mmproj tensor '{proj.Name}' has input width {proj.Dimensions[0]}; expected mlp.2's output width {mlpOut}.");
            var projectionDim = (int)proj.Dimensions[1];

            return new Llama4VisionModel(gguf)
            {
                ImageSize = imageSize,
                PatchSize = patchSize,
                EmbeddingLength = embeddingLength,
                FeedForwardLength = feedForwardLength,
                BlockCount = blockCount,
                HeadCount = headCount,
                LayerNormEps = layerNormEps,
                NMerge = nMerge,
                ProjectionDim = projectionDim,
                FfnActivation = ffnActivation,
                ImageMean = imageMean,
                ImageStd = imageStd,
                PatchEmbdWeight = patchEmbdWeight,
                PatchEmbdBias = patchEmbdBias,
                ClassEmbedding = classEmbedding,
                PositionEmbedding = positionEmbedding,
                PreLnWeight = preLnW,
                PreLnBias = preLnB,
                PostLnWeight = postLnW,
                PostLnBias = postLnB,
                MmModelMlp1Weight = mlp1,
                MmModelMlp2Weight = mlp2,
                MmModelProj = proj,
                Blocks = blocks,
            };

            int RequiredPositiveInt(string key)
            {
                var value = gguf.GetMetadata(key, 0);
                if (value <= 0)
                    throw new InvalidDataException($"mmproj '{mmprojPath}' has invalid '{key}'={value}; expected a positive integer.");
                return value;
            }

            float RequiredPositiveFiniteFloat(string key)
            {
                var value = gguf.GetMetadata(key, 0f);
                if (!float.IsFinite(value) || value <= 0f)
                    throw new InvalidDataException($"mmproj '{mmprojPath}' has invalid '{key}'={value}; expected a finite positive float.");
                return value;
            }

            float[] FloatArray(string key, bool requireNonZero)
            {
                if (!gguf.Metadata.TryGetValue(key, out var value) || value is not Array values)
                    throw new InvalidDataException($"mmproj '{mmprojPath}' is missing float-array metadata '{key}'.");
                if (values.Length != 3)
                    throw new InvalidDataException($"mmproj '{mmprojPath}' has '{key}' with {values.Length} values; expected RGB length 3.");

                var result = new float[values.Length];
                for (var i = 0; i < result.Length; i++)
                {
                    result[i] = Convert.ToSingle(values.GetValue(i));
                    if (!float.IsFinite(result[i]) || (requireNonZero && result[i] == 0f))
                        throw new InvalidDataException($"mmproj '{mmprojPath}' has invalid '{key}[{i}]'={result[i]}.");
                }
                return result;
            }

            void RequireVector(GgufTensorInfo tensor, int length) =>
                RequireShapeAndDType(tensor, [length], DType.Float32);

            void RequireMatrix(GgufTensorInfo tensor, int rows, int columns, DType type) =>
                RequireShapeAndDType(tensor, [rows, columns], type);

            void RequireShapeAndDType(GgufTensorInfo tensor, long[] expected, DType type)
            {
                if (tensor.DType != type)
                    throw new NotSupportedException(
                        $"mmproj tensor '{tensor.Name}' has dtype {tensor.DType}; expected {type}.");
                if (tensor.NDimensions != expected.Length ||
                    !tensor.Dimensions.Zip(expected).All(pair => pair.Second < 0 || pair.First == pair.Second))
                    throw new InvalidDataException(
                        $"mmproj tensor '{tensor.Name}' has shape [{string.Join(',', tensor.Dimensions)}]; " +
                        $"expected [{string.Join(',', expected.Select(dimension => dimension < 0 ? "positive" : dimension.ToString()))}].");
                if (tensor.Dimensions.Any(dimension => dimension <= 0))
                    throw new InvalidDataException($"mmproj tensor '{tensor.Name}' has a non-positive dimension.");
            }
        }
        catch
        {
            gguf.Dispose();
            throw;
        }
    }

    internal void EnsureNotDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <summary>Dequantizes a tensor's full contents to a fresh float array.</summary>
    internal float[] LoadFloats(GgufTensorInfo t)
    {
        var dst = new float[t.ElementCount];
        Cpu.Dequantize.ToFloat32(_gguf.GetTensorData(t), dst, t.DType, t.ElementCount);
        return dst;
    }

    /// <summary>Same as <see cref="LoadFloats(GgufTensorInfo)"/> but for optional tensors --
    /// returns an all-zero array of the given length when <paramref name="t"/> is absent, so
    /// callers can unconditionally add/use the result without a null branch at every call site.</summary>
    internal float[] LoadFloatsOrZero(GgufTensorInfo? t, int length) =>
        t is { } present ? LoadFloats(present) : new float[length];

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gguf.Dispose();
    }
}

/// <summary>FFN activation for the per-block gated FFN, resolved once at load time from
/// <c>clip.use_gelu</c>/<c>clip.use_silu</c> metadata (defaults to quick-GELU).</summary>
public enum Llama4FfnActivation { Gelu, Silu, GeluQuick }

/// <summary>Resolved tensor names for one <c>llama4</c> ViT transformer block. Biases are
/// genuinely optional here (unlike <see cref="Gemma3VisionModel"/>'s always-present biases or
/// <see cref="Gemma4VVisionModel"/>'s always-absent ones) -- <c>clip.cpp</c>'s generic per-block
/// loader fetches every bias with <c>required=false</c>, and llama4 doesn't override that.</summary>
internal sealed record Llama4BlockWeights(
    GgufTensorInfo Ln1W, GgufTensorInfo? Ln1B,
    GgufTensorInfo Ln2W, GgufTensorInfo? Ln2B,
    GgufTensorInfo AttnQ, GgufTensorInfo? AttnQBias,
    GgufTensorInfo AttnK, GgufTensorInfo? AttnKBias,
    GgufTensorInfo AttnV, GgufTensorInfo? AttnVBias,
    GgufTensorInfo AttnOut, GgufTensorInfo? AttnOutBias,
    GgufTensorInfo? FfnGate, GgufTensorInfo? FfnGateBias,
    GgufTensorInfo FfnUp, GgufTensorInfo? FfnUpBias,
    GgufTensorInfo FfnDown, GgufTensorInfo? FfnDownBias);
