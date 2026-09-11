using System.Globalization;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.SenseVoice;

/// <summary>
/// Real metadata for Alibaba's SenseVoice-Small CTC ASR ONNX checkpoint, read directly from the
/// model's own embedded ONNX custom metadata (`InferenceSession.ModelMetadata.CustomMetadataMap`)
/// rather than hardcoded -- ported from sherpa-onnx's real
/// `sherpa-onnx/csrc/offline-sense-voice-model.cc` (`Impl::Init`, lines ~107-148) and cross-checked
/// against the real export script that actually writes these keys,
/// `examples/sherpa-onnx/scripts/sense-voice/export-onnx.py` lines 159-190
/// (`add_meta_data(filename, meta_data)`), which writes: `lfr_window_size`, `lfr_window_shift`,
/// `normalize_samples`, `neg_mean` (comma-separated floats, from the Kaldi `cmvn` file's
/// `&lt;LearnRateCoef&gt;` lines), `inv_stddev` (same format), `vocab_size`, `comment`,
/// `lang_auto`/`lang_zh`/`lang_en`/`lang_yue`/`lang_ja`/`lang_ko`, `with_itn`, `without_itn`.
/// `blank_id` is NOT written by the export script -- sherpa-onnx reads it with a default of 0
/// (`SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.blank_id, "blank_id", 0)`), so this port
/// does the same.
/// </summary>
public sealed class SenseVoiceMetaData
{
    public required int VocabSize { get; init; }
    public required int BlankId { get; init; }
    public required int LfrWindowSize { get; init; } // lfr_m
    public required int LfrWindowShift { get; init; } // lfr_n
    public required int NormalizeSamples { get; init; }
    public required int WithItnId { get; init; }
    public required int WithoutItnId { get; init; }
    public required IReadOnlyDictionary<string, int> Lang2Id { get; init; }
    public required float[] NegMean { get; init; }
    public required float[] InvStddev { get; init; }

    public static SenseVoiceMetaData FromCustomMetadata(IReadOnlyDictionary<string, string> meta)
    {
        int GetInt(string key, int? fallback = null)
        {
            if (meta.TryGetValue(key, out var v) && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i))
                return i;
            if (fallback.HasValue) return fallback.Value;
            throw new InvalidOperationException($"SenseVoice ONNX metadata missing required key '{key}'.");
        }

        float[] GetFloatVec(string key)
        {
            if (!meta.TryGetValue(key, out var v) || string.IsNullOrWhiteSpace(v))
                throw new InvalidOperationException($"SenseVoice ONNX metadata missing required key '{key}'.");
            var parts = v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var arr = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                arr[i] = float.Parse(parts[i], CultureInfo.InvariantCulture);
            return arr;
        }

        var lang2Id = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["auto"] = GetInt("lang_auto"),
            ["zh"] = GetInt("lang_zh"),
            ["en"] = GetInt("lang_en"),
            ["yue"] = GetInt("lang_yue"),
            ["ja"] = GetInt("lang_ja"),
            ["ko"] = GetInt("lang_ko"),
        };

        return new SenseVoiceMetaData
        {
            VocabSize = GetInt("vocab_size"),
            BlankId = GetInt("blank_id", fallback: 0),
            LfrWindowSize = GetInt("lfr_window_size"),
            LfrWindowShift = GetInt("lfr_window_shift"),
            NormalizeSamples = GetInt("normalize_samples"),
            WithItnId = GetInt("with_itn"),
            WithoutItnId = GetInt("without_itn"),
            Lang2Id = lang2Id,
            NegMean = GetFloatVec("neg_mean"),
            InvStddev = GetFloatVec("inv_stddev"),
        };
    }
}

/// <summary>
/// Real ONNX model wrapper for SenseVoice-Small, ported from sherpa-onnx's real
/// `OfflineSenseVoiceModel`/`Impl` (`offline-sense-voice-model.cc`/`.h`). The real (non-nano)
/// export declares exactly 4 named inputs and 1 named output -- confirmed from the actual real
/// export call in `examples/sherpa-onnx/scripts/sense-voice/export-onnx.py` lines 143-156:
/// `input_names=["x", "x_length", "language", "text_norm"]`, `output_names=["logits"]`. Inputs are
/// looked up by these exact names via <see cref="OnnxModelSession"/>'s name-filtered `Run`, not by
/// position, matching the real reference's `GetInputNames`/`GetOutputNames`-driven dispatch.
/// </summary>
public sealed class SenseVoiceModel : IDisposable
{
    private readonly OnnxModelSession _session;

    public SenseVoiceMetaData MetaData { get; }
    public bool IsAvailable => _session.IsAvailable;

    private SenseVoiceModel(OnnxModelSession session, SenseVoiceMetaData metaData)
    {
        _session = session;
        MetaData = metaData;
    }

    public static SenseVoiceModel? TryLoad(string onnxPath)
    {
        var session = OnnxModelSession.TryLoad(onnxPath);
        if (session is null) return null;

        var meta = SenseVoiceMetaData.FromCustomMetadata(session.CustomMetadata);
        return new SenseVoiceModel(session, meta);
    }

    /// <summary>
    /// Real 4-input forward pass. `features` is frame-major `[numFrames, featDim]` (already
    /// LFR-spliced + CMVN-normalized, `featDim = 80 * LfrWindowSize`), batch size is always 1.
    /// Returns flat `[numFrames', vocabSize]` logits (row-major) plus the real output frame count
    /// (which is `numFrames + 4` because the model itself prepends 4 special-token frames --
    /// language/emotion/event/text-norm -- confirmed from the real reference's own
    /// `DecodeOneStream`: `new_num_frames = num_frames + 4` is passed as the CTC decoder's
    /// `logits_length`; this port instead reads the real output tensor's own frame dimension
    /// directly from the ONNX result, which is equivalent and avoids hardcoding the "+4").
    /// </summary>
    public (float[] Logits, int NumFrames, int VocabSize)? Forward(
        float[] featuresFlat, int numFrames, int featDim, int language, int textNorm)
    {
        var xLength = new int[] { numFrames };
        var languageArr = new int[] { language };
        var textNormArr = new int[] { textNorm };

        var outputs = _session.Run(
            ("x", featuresFlat, new[] { 1, numFrames, featDim }),
            ("x_length", xLength, new[] { 1 }),
            ("language", languageArr, new[] { 1 }),
            ("text_norm", textNormArr, new[] { 1 }));

        if (!outputs.TryGetValue("logits", out var logits))
        {
            // Fall back to the first (only) output if the name doesn't match exactly for some
            // reason -- keeps this robust the same way OnnxModelSession.RunToFloatArray does.
            if (outputs.Count == 0) return null;
            logits = outputs.Values.First();
        }

        int vocabSize = MetaData.VocabSize;
        int outFrames = logits.Length / vocabSize;
        return (logits, outFrames, vocabSize);
    }

    public void Dispose() => _session.Dispose();
}
