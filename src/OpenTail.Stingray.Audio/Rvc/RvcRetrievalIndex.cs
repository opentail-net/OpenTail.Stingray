
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real k-NN retrieval index loader for RVC's optional feature-blending step, transcribed
/// directly from `examples/audio.cpp/src/models/rvc/retrieval_index.cpp`'s
/// `load_retrieval_sidecar` (not guessed) -- the "bundled sidecar" format, i.e. a
/// `centroids`/`vectors`/`list_offsets`/`list_lengths` tensor group under a voice's own
/// `..._index_vectors/` prefix in the packed `rvc-f16.gguf` (confirmed via the same real
/// tensor-name dump that found the multiple bundled voices -- see
/// `docs/audio-review-progress.md`'s RVC synthesizer update). The separate real FAISS
/// `IndexIVFFlat` binary-file format (`retrieval_index_path` pointing at a user's own
/// `.index` file) is NOT implemented here -- only the packaged/bundled sidecar path used by
/// this checkpoint's own voices.
/// </summary>
public sealed class RvcRetrievalIndex
{
    public int Dim { get; }
    public int NList { get; }
    public float[] Centroids { get; } // [NList, Dim]
    public float[] Vectors { get; } // [totalVectors, Dim]
    public int[] ListOffsets { get; } // [NList]
    public int[] ListLengths { get; } // [NList]

    public RvcRetrievalIndex(RvcPackedTensorSource source, string voicePrefix, int dim)
    {
        Dim = dim;
        string root = voicePrefix + "/";
        Centroids = source.GetTensor(root + "centroids");
        Vectors = source.GetTensor(root + "vectors");
        var offsetsF = source.GetTensor(root + "list_offsets");
        var lengthsF = source.GetTensor(root + "list_lengths");
        NList = offsetsF.Length;
        ListOffsets = new int[NList];
        ListLengths = new int[NList];
        long total = 0;
        for (int i = 0; i < NList; i++)
        {
            ListOffsets[i] = (int)MathF.Round(offsetsF[i]);
            ListLengths[i] = (int)MathF.Round(lengthsF[i]);
            total += ListLengths[i];
        }
        if (total * dim != Vectors.Length)
            throw new InvalidDataException($"RVC retrieval index list lengths ({total}) do not match vector count ({Vectors.Length / dim}).");
    }

    private const int Neighbors = 8;

    /// <summary>Real RVC retrieval blend, transcribed exactly from `native_pipeline.cpp`'s
    /// `apply_retrieval_blend`: for each content frame, find the nearest centroid (L2), keep the
    /// 8 nearest vectors within that centroid's list (a running worst-replacement scan, matching
    /// the reference's fixed-size top-8 array exactly rather than a full sort), weight each by
    /// `(1/distSquared)^2` (an exact-match vector, dist&lt;=0, is used directly with no
    /// averaging), then linearly blend `blended*retrievalBlend + original*(1-retrievalBlend)`.
    /// </summary>
    public void ApplyBlend(Span<float> features, ReadOnlySpan<float> sourceFeatures, int frames, int dim, float retrievalBlend)
    {
        if (retrievalBlend == 0f) return;
        var blended = new float[dim];
        Span<float> nearestDist = stackalloc float[Neighbors];
        Span<int> nearestIndex = stackalloc int[Neighbors];

        for (int frame = 0; frame < frames; frame++)
        {
            var query = sourceFeatures.Slice(frame * dim, dim);

            int bestList = 0;
            float bestCentroidDist = float.PositiveInfinity;
            for (int c = 0; c < NList; c++)
            {
                float dist = SquaredL2(query, Centroids.AsSpan(c * dim, dim));
                if (dist < bestCentroidDist) { bestCentroidDist = dist; bestList = c; }
            }

            int offset = ListOffsets[bestList];
            int length = ListLengths[bestList];
            int nearestCount = 0;
            for (int row = 0; row < length; row++)
            {
                int vectorIndex = offset + row;
                float dist = SquaredL2(query, Vectors.AsSpan(vectorIndex * dim, dim));
                if (nearestCount < Neighbors)
                {
                    nearestDist[nearestCount] = dist;
                    nearestIndex[nearestCount] = vectorIndex;
                    nearestCount++;
                    continue;
                }
                int worstSlot = 0;
                float worstDist = nearestDist[0];
                for (int slot = 1; slot < Neighbors; slot++)
                {
                    if (nearestDist[slot] > worstDist) { worstDist = nearestDist[slot]; worstSlot = slot; }
                }
                if (dist < worstDist)
                {
                    nearestDist[worstSlot] = dist;
                    nearestIndex[worstSlot] = vectorIndex;
                }
            }
            if (nearestCount == 0)
                throw new InvalidOperationException("RVC retrieval selected an empty IVF list");

            Array.Clear(blended, 0, dim);
            float weightSum = 0f;
            for (int slot = 0; slot < nearestCount; slot++)
            {
                float dist = nearestDist[slot];
                int vectorIndex = nearestIndex[slot];
                if (dist <= 0f)
                {
                    Vectors.AsSpan(vectorIndex * dim, dim).CopyTo(blended);
                    weightSum = -1f;
                    break;
                }
                float inv = 1f / dist;
                float weight = inv * inv;
                weightSum += weight;
                var src = Vectors.AsSpan(vectorIndex * dim, dim);
                for (int d = 0; d < dim; d++) blended[d] += src[d] * weight;
            }
            if (weightSum > 0f)
            {
                for (int d = 0; d < dim; d++) blended[d] /= weightSum;
            }

            var dst = features.Slice(frame * dim, dim);
            for (int d = 0; d < dim; d++)
                dst[d] = blended[d] * retrievalBlend + query[d] * (1f - retrievalBlend);
        }
    }

    private static float SquaredL2(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }
}
