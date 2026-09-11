using System.Globalization;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.ParaformerOnnx;

/// <summary>
/// Real ONNX wrapper for FunASR's Paraformer (non-autoregressive CIF encoder-predictor-decoder,
/// fully fused into ONE exported ONNX graph) -- ported from
/// `examples/sherpa-onnx/sherpa-onnx/csrc/offline-paraformer-model.h/.cc`. Confirmed against the
/// real local checkpoint `models/_models/paraformer-zh-small.int8.onnx`'s own embedded custom
/// metadata (real inspection, 2026-09-11): input names `speech` (float32 `[N,T,560]`) and
/// `speech_lengths` (int32 `[N]`); output names `logits` (float32 `[N,T',8359]`) and `token_num`
/// (int32 `[N,T']`, unused by the real greedy decoder -- see `ParaformerOnnxDecoder`); custom
/// metadata keys `vocab_size`=8359, `lfr_window_size`=7, `lfr_window_shift`=6, plus real
/// `neg_mean`/`inv_stddev` CMVN vectors (560 floats each = feature_dim(80) * lfr_window_size(7)).
/// This checkpoint carries no `us_alphas`/`us_cif_peak` timestamp outputs (real, not an omission --
/// the real reference's `Forward()` doc comment marks those as present only "if it is a model
/// supporting timestamps").
/// </summary>
public sealed class ParaformerOnnxModel : IDisposable
{
    private readonly OnnxModelSession _session;

    public int VocabSize { get; }
    public int LfrWindowSize { get; }
    public int LfrWindowShift { get; }
    public float[] NegMean { get; }
    public float[] InvStdDev { get; }

    private ParaformerOnnxModel(OnnxModelSession session, int vocabSize, int lfrWindowSize,
        int lfrWindowShift, float[] negMean, float[] invStdDev)
    {
        _session = session;
        VocabSize = vocabSize;
        LfrWindowSize = lfrWindowSize;
        LfrWindowShift = lfrWindowShift;
        NegMean = negMean;
        InvStdDev = invStdDev;
    }

    public static ParaformerOnnxModel? TryLoad(string modelPath)
    {
        var session = OnnxModelSession.TryLoad(modelPath);
        if (session is null) return null;

        var meta = session.CustomMetadata;
        int vocabSize = ReadInt(meta, "vocab_size");
        int lfrWindowSize = ReadInt(meta, "lfr_window_size");
        int lfrWindowShift = ReadInt(meta, "lfr_window_shift");
        float[] negMean = ReadFloatVec(meta, "neg_mean");
        float[] invStdDev = ReadFloatVec(meta, "inv_stddev");

        if (vocabSize <= 0 || lfrWindowSize <= 0 || lfrWindowShift <= 0 ||
            negMean.Length == 0 || invStdDev.Length == 0)
        {
            session.Dispose();
            return null;
        }

        return new ParaformerOnnxModel(session, vocabSize, lfrWindowSize, lfrWindowShift, negMean, invStdDev);
    }

    private static int ReadInt(IReadOnlyDictionary<string, string> meta, string key) =>
        meta.TryGetValue(key, out var s) && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : -1;

    private static float[] ReadFloatVec(IReadOnlyDictionary<string, string> meta, string key)
    {
        if (!meta.TryGetValue(key, out var s) || string.IsNullOrEmpty(s)) return [];
        var parts = s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var result = new float[parts.Length];
        for (int i = 0; i < parts.Length; i++)
            result[i] = float.Parse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture);
        return result;
    }

    /// <summary>
    /// Real forward pass: `features` is frame-major `[numFrames, featDim]` flattened row-major
    /// (featDim == feature_dim(80) * LfrWindowSize, i.e. already LFR-spliced and CMVN-normalized).
    /// Returns `logits` reshaped to `[T', VocabSize]` (frame-major). Matches
    /// `OfflineParaformerModel::Forward`'s real two-input/`log_probs`-output contract (this
    /// checkpoint's `token_num` output is real but unused, per the real greedy decoder -- see
    /// `offline-paraformer-greedy-search-decoder.cc` line 17, `Ort::Value /*token_num*/`).
    /// </summary>
    public float[][] Forward(float[] featuresFlat, int numFrames, int featDim)
    {
        var speech = featuresFlat;
        var speechShape = new[] { 1, numFrames, featDim };
        var speechLengths = new[] { numFrames };
        var speechLengthsShape = new[] { 1 };

        var outputs = _session.Run(
            ("speech", speech, speechShape),
            ("speech_lengths", speechLengths, speechLengthsShape));

        if (!outputs.TryGetValue("logits", out var logitsFlat))
        {
            throw new InvalidOperationException(
                "Paraformer ONNX forward pass did not produce a 'logits' output tensor.");
        }

        int tOut = logitsFlat.Length / VocabSize;
        var result = new float[tOut][];
        for (int t = 0; t < tOut; t++)
        {
            var row = new float[VocabSize];
            Array.Copy(logitsFlat, t * VocabSize, row, 0, VocabSize);
            result[t] = row;
        }
        return result;
    }

    public void Dispose() => _session.Dispose();
}
