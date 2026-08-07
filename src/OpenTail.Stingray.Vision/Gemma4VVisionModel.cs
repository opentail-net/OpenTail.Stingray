using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Loaded Gemma 4 E-model vision transformer projector
/// (<c>clip.vision.projector_type=gemma4v</c>). This is deliberately separate from
/// <see cref="VisionModel"/>, whose <c>gemma4uv</c> path is encoder-free.
/// </summary>
/// <remarks>
/// This type is the validated ownership and tensor-name boundary for the E4B ViT. It does not
/// claim that encoder forward, token reduction, or image-token mask semantics are implemented;
/// those must be parity-derived before embeddings are admitted to the text decoder.
/// </remarks>
public sealed class Gemma4VVisionModel : IDisposable
{
    public const string ProjectorType = "gemma4v";

    private readonly GgufModel _gguf;
    private bool _disposed;

    private Gemma4VVisionModel(GgufModel gguf) => _gguf = gguf;

    public required int ImageSize { get; init; }
    public required int PatchSize { get; init; }
    public required int EmbeddingLength { get; init; }
    public required int ProjectionDim { get; init; }
    public required int FeedForwardLength { get; init; }
    public required int BlockCount { get; init; }
    public required int HeadCount { get; init; }
    public required float LayerNormEps { get; init; }

    /// <summary>Vision input channel mean from the mmproj header.</summary>
    public required float[] ImageMean { get; init; }

    /// <summary>Vision input channel standard deviation from the mmproj header.</summary>
    public required float[] ImageStd { get; init; }

    internal GgufModel Gguf => _gguf;
    internal GgufTensorInfo PatchEmbedding { get; init; }
    internal GgufTensorInfo PositionEmbedding { get; init; }
    internal GgufTensorInfo InputProjection { get; init; }
    internal Gemma4VBlockWeights[] Blocks { get; init; } = [];

    /// <summary>
    /// Opens and validates the complete tensor inventory for an E-model <c>gemma4v</c> vision
    /// encoder. Validation is intentionally strict: a partial or differently named export must
    /// fail at load time instead of producing plausible but wrong image embeddings later.
    /// </summary>
    public static Gemma4VVisionModel Open(string mmprojPath)
    {
        var gguf = GgufModel.Open(mmprojPath);
        try
        {
            if (gguf.GetMetadata("general.architecture", "") != "clip")
                throw new NotSupportedException($"'{mmprojPath}' is not a clip mmproj.");
            if (gguf.GetMetadata("clip.vision.projector_type", "") != ProjectorType)
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
            // Only the divisibility needs checking here; RequiredPositiveInt has already rejected
            // non-positive blockCount/embeddingLength/headCount above.
            if (embeddingLength % headCount != 0)
                throw new InvalidDataException(
                    $"mmproj '{mmprojPath}' has invalid ViT geometry: embedding={embeddingLength} " +
                    $"is not divisible by heads={headCount}.");
            if (imageSize % patchSize != 0)
                throw new InvalidDataException(
                    $"mmproj '{mmprojPath}' has image_size={imageSize}, which is not divisible by patch_size={patchSize}.");

            var imageMean = FloatArray("clip.vision.image_mean", requireNonZero: false);
            var imageStd = FloatArray("clip.vision.image_std", requireNonZero: true);

            GgufTensorInfo Required(string name) => gguf.FindTensor(name)
                ?? throw new InvalidDataException($"mmproj '{mmprojPath}' is missing tensor '{name}'.");

            var blocks = new Gemma4VBlockWeights[blockCount];
            for (var i = 0; i < blocks.Length; i++)
            {
                var prefix = $"v.blk.{i}.";
                blocks[i] = new Gemma4VBlockWeights(
                    Required(prefix + "ln1.weight"),
                    Required(prefix + "ln2.weight"),
                    Required(prefix + "attn_q.weight"),
                    Required(prefix + "attn_k.weight"),
                    Required(prefix + "attn_v.weight"),
                    Required(prefix + "attn_out.weight"),
                    Required(prefix + "attn_q_norm.weight"),
                    Required(prefix + "attn_k_norm.weight"),
                    Required(prefix + "attn_post_norm.weight"),
                    Required(prefix + "ffn_gate.weight"),
                    Required(prefix + "ffn_up.weight"),
                    Required(prefix + "ffn_down.weight"),
                    Required(prefix + "ffn_post_norm.weight"));

                RequireVector(blocks[i].Ln1, embeddingLength);
                RequireVector(blocks[i].Ln2, embeddingLength);
                RequireMatrix(blocks[i].AttnQ, embeddingLength, embeddingLength, DType.BFloat16);
                RequireMatrix(blocks[i].AttnK, embeddingLength, embeddingLength, DType.BFloat16);
                RequireMatrix(blocks[i].AttnV, embeddingLength, embeddingLength, DType.BFloat16);
                RequireMatrix(blocks[i].AttnOut, embeddingLength, embeddingLength, DType.BFloat16);
                RequireVector(blocks[i].AttnQNorm, embeddingLength / headCount);
                RequireVector(blocks[i].AttnKNorm, embeddingLength / headCount);
                RequireVector(blocks[i].AttnPostNorm, embeddingLength);
                RequireMatrix(blocks[i].FfnGate, embeddingLength, feedForwardLength, DType.BFloat16);
                RequireMatrix(blocks[i].FfnUp, embeddingLength, feedForwardLength, DType.BFloat16);
                RequireMatrix(blocks[i].FfnDown, feedForwardLength, embeddingLength, DType.BFloat16);
                RequireVector(blocks[i].FfnPostNorm, embeddingLength);
            }

            var patchEmbedding = Required("v.patch_embd.weight");
            RequireShapeAndDType(patchEmbedding, [patchSize, patchSize, 3, embeddingLength], DType.Float32);
            var positionEmbedding = Required("v.position_embd.weight");
            RequireShapeAndDType(positionEmbedding, [embeddingLength, -1, 2], DType.Float32);
            var inputProjection = Required("mm.input_projection.weight");
            RequireMatrix(inputProjection, embeddingLength, projectionDim, DType.BFloat16);

            return new Gemma4VVisionModel(gguf)
            {
                ImageSize = imageSize,
                PatchSize = patchSize,
                EmbeddingLength = embeddingLength,
                ProjectionDim = projectionDim,
                FeedForwardLength = feedForwardLength,
                BlockCount = blockCount,
                HeadCount = headCount,
                LayerNormEps = layerNormEps,
                ImageMean = imageMean,
                ImageStd = imageStd,
                PatchEmbedding = patchEmbedding,
                PositionEmbedding = positionEmbedding,
                InputProjection = inputProjection,
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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gguf.Dispose();
    }
}

/// <summary>Resolved tensor names for one <c>gemma4v</c> ViT transformer block.</summary>
internal sealed record Gemma4VBlockWeights(
    GgufTensorInfo Ln1,
    GgufTensorInfo Ln2,
    GgufTensorInfo AttnQ,
    GgufTensorInfo AttnK,
    GgufTensorInfo AttnV,
    GgufTensorInfo AttnOut,
    GgufTensorInfo AttnQNorm,
    GgufTensorInfo AttnKNorm,
    GgufTensorInfo AttnPostNorm,
    GgufTensorInfo FfnGate,
    GgufTensorInfo FfnUp,
    GgufTensorInfo FfnDown,
    GgufTensorInfo FfnPostNorm);
