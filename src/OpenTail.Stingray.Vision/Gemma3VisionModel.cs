
namespace OpenTail.Stingray.Vision;

/// <summary>
/// Loaded Gemma 3 SigLIP vision transformer projector (<c>clip.vision.projector_type=gemma3</c>).
/// This is <c>llama.cpp</c>'s <c>clip_graph_siglip</c> encoder body with the
/// <c>PROJECTOR_TYPE_GEMMA3</c> pooling/projection tail -- a genuinely simpler ViT than the
/// Gemma 4 E4B <c>gemma4v</c> encoder (<see cref="Gemma4VVisionModel"/>): no 2D RoPE, no per-head
/// QK-norm, no V-norm, no per-block QAT clamp, a single (not dual x/y) learned position table, a
/// plain (not gated) FFN, and standard <c>1/sqrt(head_dim)</c> attention scale. It has its own new
/// wrinkles instead: every per-block linear has a real bias, the final projection tensor is stored
/// transposed relative to this codebase's row-major convention, weights are F16 (not BF16), and
/// the input grid is much larger (896px / 14px patches = 4096 patches, 27 blocks, head_dim 72).
/// See docs/03-gemma4-e4b-vision-plan.md's Gemma 3 addendum for the full derivation, verified
/// against the real <c>models/mmproj-gemma-3-4b-it-f16.gguf</c>.
/// </summary>
public sealed class Gemma3VisionModel : IDisposable
{
    public const string ProjectorType = "gemma3";

    private readonly GgufModel _gguf;
    private bool _disposed;

    private Gemma3VisionModel(GgufModel gguf) => _gguf = gguf;

    public required int ImageSize { get; init; }
    public required int PatchSize { get; init; }
    public required int EmbeddingLength { get; init; }
    public required int ProjectionDim { get; init; }
    public required int FeedForwardLength { get; init; }
    public required int BlockCount { get; init; }
    public required int HeadCount { get; init; }
    public required float LayerNormEps { get; init; }
    public required int NMerge { get; init; }

    /// <summary>Vision input channel mean from the mmproj header.</summary>
    public required float[] ImageMean { get; init; }

    /// <summary>Vision input channel standard deviation from the mmproj header.</summary>
    public required float[] ImageStd { get; init; }

    internal GgufModel Gguf => _gguf;
    internal GgufTensorInfo PatchEmbdWeight { get; init; }
    internal GgufTensorInfo PatchEmbdBias { get; init; }
    internal GgufTensorInfo PositionEmbedding { get; init; }
    internal GgufTensorInfo PostLnWeight { get; init; }
    internal GgufTensorInfo PostLnBias { get; init; }
    internal GgufTensorInfo MmSoftEmbNorm { get; init; }
    internal GgufTensorInfo MmInputProjection { get; init; }
    internal Gemma3BlockWeights[] Blocks { get; init; } = [];

    /// <summary>
    /// Opens and validates the complete tensor inventory for a Gemma 3 <c>gemma3</c> SigLIP
    /// vision encoder. Strict by construction, same rationale as
    /// <see cref="Gemma4VVisionModel.Open"/>: a partial or differently-shaped export must fail at
    /// load time, not produce plausible-but-wrong image embeddings later.
    /// </summary>
    public static Gemma3VisionModel Open(string mmprojPath)
    {
        var gguf = GgufModel.Open(mmprojPath);
        try
        {
            if (gguf.GetMetadata("general.architecture", "") != "clip")
                throw new NotSupportedException($"'{mmprojPath}' is not a clip mmproj.");
            // NOTE: this mmproj declares the key as "clip.projector_type" (no "vision." segment),
            // NOT "clip.vision.projector_type" like the gemma4v/gemma4uv mmprojs use -- confirmed
            // directly against the real file (list-metadata). The two conventions genuinely
            // differ between exports; do not assume one implies the other.
            if (gguf.GetMetadata("clip.projector_type", "") != ProjectorType)
                throw new NotSupportedException(
                    $"'{mmprojPath}' is not a {ProjectorType} vision projector.");
            if (!gguf.GetMetadata("clip.has_vision_encoder", false))
                throw new NotSupportedException($"'{mmprojPath}' declares no vision encoder.");

            var blockCount = RequiredPositiveInt("clip.vision.block_count");
            var imageSize = RequiredPositiveInt("clip.vision.image_size");
            var patchSize = RequiredPositiveInt("clip.vision.patch_size");
            var embeddingLength = RequiredPositiveInt("clip.vision.embedding_length");
            var projectionDim = RequiredPositiveInt("clip.vision.projection_dim");
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

            // clip.cpp's PROJECTOR_TYPE_GEMMA3 case: hparams.n_merge = 4 (default), optionally
            // overridden by clip.vision.projector.scale_factor (KEY_PROJ_SCALE_FACTOR).
            var nMerge = gguf.GetMetadata("clip.vision.projector.scale_factor", 4);

            var imageMean = FloatArray("clip.vision.image_mean", requireNonZero: false);
            var imageStd = FloatArray("clip.vision.image_std", requireNonZero: true);

            GgufTensorInfo Required(string name) => gguf.FindTensor(name)
                ?? throw new InvalidDataException($"mmproj '{mmprojPath}' is missing tensor '{name}'.");

            var blocks = new Gemma3BlockWeights[blockCount];
            for (var i = 0; i < blocks.Length; i++)
            {
                var prefix = $"v.blk.{i}.";
                // NOTE: this checkpoint's GGUF tensor NAMES "ffn_up"/"ffn_down" are swapped
                // relative to their actual FUNCTIONAL role -- confirmed unambiguously via bias
                // length (a bias vector's length is exactly its projection's output width, no
                // axis-order ambiguity possible): the tensor named "ffn_up" has a
                // embeddingLength-wide bias (it is actually the SECOND/reducing step,
                // ffLen->embd), and the tensor named "ffn_down" has a feedForwardLength-wide bias
                // (it is actually the FIRST/expanding step, embd->ffLen). This is a naming quirk
                // in this specific export, not a storage-transpose issue -- both tensors use the
                // ordinary row-major [input,output] convention once bound to the correct role, so
                // Gemma3VisionEncoder needs no transpose for either (unlike mm.input_projection,
                // which genuinely IS stored transposed -- confirmed by siglip.cpp's own explicit
                // ggml_transpose call before using it). Bind by FUNCTION below, not by GGUF name.
                blocks[i] = new Gemma3BlockWeights(
                    Required(prefix + "ln1.weight"), Required(prefix + "ln1.bias"),
                    Required(prefix + "ln2.weight"), Required(prefix + "ln2.bias"),
                    Required(prefix + "attn_q.weight"), Required(prefix + "attn_q.bias"),
                    Required(prefix + "attn_k.weight"), Required(prefix + "attn_k.bias"),
                    Required(prefix + "attn_v.weight"), Required(prefix + "attn_v.bias"),
                    Required(prefix + "attn_out.weight"), Required(prefix + "attn_out.bias"),
                    FfnUp: Required(prefix + "ffn_down.weight"), FfnUpBias: Required(prefix + "ffn_down.bias"),
                    FfnDown: Required(prefix + "ffn_up.weight"), FfnDownBias: Required(prefix + "ffn_up.bias"));

                RequireVector(blocks[i].Ln1W, embeddingLength);
                RequireVector(blocks[i].Ln1B, embeddingLength);
                RequireVector(blocks[i].Ln2W, embeddingLength);
                RequireVector(blocks[i].Ln2B, embeddingLength);
                RequireMatrix(blocks[i].AttnQ, embeddingLength, embeddingLength, DType.Float16);
                RequireVector(blocks[i].AttnQBias, embeddingLength);
                RequireMatrix(blocks[i].AttnK, embeddingLength, embeddingLength, DType.Float16);
                RequireVector(blocks[i].AttnKBias, embeddingLength);
                RequireMatrix(blocks[i].AttnV, embeddingLength, embeddingLength, DType.Float16);
                RequireVector(blocks[i].AttnVBias, embeddingLength);
                RequireMatrix(blocks[i].AttnOut, embeddingLength, embeddingLength, DType.Float16);
                RequireVector(blocks[i].AttnOutBias, embeddingLength);
                RequireMatrix(blocks[i].FfnUp, embeddingLength, feedForwardLength, DType.Float16);
                RequireVector(blocks[i].FfnUpBias, feedForwardLength);
                RequireMatrix(blocks[i].FfnDown, feedForwardLength, embeddingLength, DType.Float16);
                RequireVector(blocks[i].FfnDownBias, embeddingLength);
            }

            var patchEmbdWeight = Required("v.patch_embd.weight");
            RequireShapeAndDType(patchEmbdWeight, [patchSize, patchSize, 3, embeddingLength], DType.Float32);
            var patchEmbdBias = Required("v.patch_embd.bias");
            RequireVector(patchEmbdBias, embeddingLength);
            var positionEmbedding = Required("v.position_embd.weight");
            // One row per patch, NOT the dual x/y factorised scheme gemma4v uses -- confirmed by
            // siglip.cpp adding it directly (ggml_add) with no ggml_get_rows lookup at all.
            var expectedPatches = (imageSize / patchSize) * (imageSize / patchSize);
            RequireShapeAndDType(positionEmbedding, [embeddingLength, expectedPatches], DType.Float32);
            var postLnW = Required("v.post_ln.weight");
            RequireVector(postLnW, embeddingLength);
            var postLnB = Required("v.post_ln.bias");
            RequireVector(postLnB, embeddingLength);
            var mmSoftEmbNorm = Required("mm.soft_emb_norm.weight");
            RequireVector(mmSoftEmbNorm, embeddingLength);
            var mmInputProjection = Required("mm.input_projection.weight");
            // Stored TRANSPOSED relative to every other weight in this codebase: ne=[projDim,embd]
            // (projDim contiguous), not [embd,projDim]. siglip.cpp explicitly transposes it before
            // use (ggml_cont(ggml_transpose(...))) -- the encoder must do the same, not feed it
            // straight into the usual row-major MatVec convention.
            RequireMatrix(mmInputProjection, projectionDim, embeddingLength, DType.Float16);

            return new Gemma3VisionModel(gguf)
            {
                ImageSize = imageSize,
                PatchSize = patchSize,
                EmbeddingLength = embeddingLength,
                ProjectionDim = projectionDim,
                FeedForwardLength = feedForwardLength,
                BlockCount = blockCount,
                HeadCount = headCount,
                LayerNormEps = layerNormEps,
                NMerge = nMerge,
                ImageMean = imageMean,
                ImageStd = imageStd,
                PatchEmbdWeight = patchEmbdWeight,
                PatchEmbdBias = patchEmbdBias,
                PositionEmbedding = positionEmbedding,
                PostLnWeight = postLnW,
                PostLnBias = postLnB,
                MmSoftEmbNorm = mmSoftEmbNorm,
                MmInputProjection = mmInputProjection,
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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gguf.Dispose();
    }
}

/// <summary>Resolved tensor names for one <c>gemma3</c> SigLIP ViT transformer block.
/// Every projection here carries a real bias, unlike <see cref="Gemma4VBlockWeights"/>.</summary>
internal sealed record Gemma3BlockWeights(
    GgufTensorInfo Ln1W, GgufTensorInfo Ln1B,
    GgufTensorInfo Ln2W, GgufTensorInfo Ln2B,
    GgufTensorInfo AttnQ, GgufTensorInfo AttnQBias,
    GgufTensorInfo AttnK, GgufTensorInfo AttnKBias,
    GgufTensorInfo AttnV, GgufTensorInfo AttnVBias,
    GgufTensorInfo AttnOut, GgufTensorInfo AttnOutBias,
    GgufTensorInfo FfnUp, GgufTensorInfo FfnUpBias,
    GgufTensorInfo FfnDown, GgufTensorInfo FfnDownBias);
