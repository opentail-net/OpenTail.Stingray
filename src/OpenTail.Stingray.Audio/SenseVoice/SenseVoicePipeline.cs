using OpenTail.Stingray.Audio.FunASR;

namespace OpenTail.Stingray.Audio.SenseVoice;

/// <summary>Real transcript plus the optional language/emotion/event tags SenseVoice's first 3
/// decoded tokens encode (real, per <see cref="Transcribe"/>'s doc comment).</summary>
public sealed record SenseVoiceResult(string Text, string? Language, string? Emotion, string? Event);

/// <summary>
/// Real SenseVoice-Small CTC ASR pipeline: real Kaldi-style fbank + LFR splice + CMVN
/// (<see cref="FunAsrRealMelExtractor"/>, already real and shared with the FunASR/Paraformer
/// family -- same real frontend config, confirmed matching in
/// `offline-recognizer-sense-voice-impl.h`'s `InitFeatConfig`: `window_type="hamming"`,
/// `high_freq=0`, `snip_edges=true`, `sampling_rate=16000`, `feature_dim=80`), real 4-input ONNX
/// forward pass (<see cref="SenseVoiceModel"/>), real greedy CTC decode
/// (<see cref="SenseVoiceCtcDecoder"/>), real vocabulary lookup (<see cref="SenseVoiceVocab"/>).
///
/// <para>Ported from sherpa-onnx's real `OfflineRecognizerSenseVoiceImpl::DecodeOneStream` and
/// `ConvertSenseVoiceResult` (`offline-recognizer-sense-voice-impl.h` lines ~26-67 and ~297-370).
/// The non-nano SenseVoice model prepends 4 special-token frames to its real output sequence
/// (language, emotion, event, text-norm/ITN marker) -- <see cref="ConvertSenseVoiceResult"/>
/// mirrors the real reference's `start = 4` convention: the first 3 decoded tokens are looked up
/// as language/emotion/event tags, and the real transcript text starts from decoded-token index 4
/// onward.</para>
/// </summary>
public sealed class SenseVoicePipeline : IDisposable
{
    private readonly SenseVoiceModel _model;
    private readonly SenseVoiceVocab _vocab;
    private readonly FunAsrRealMelExtractor _melExtractor = new();

    public SenseVoicePipeline(SenseVoiceModel model, SenseVoiceVocab vocab)
    {
        _model = model;
        _vocab = vocab;
    }

    public static SenseVoicePipeline? TryLoad(string onnxPath, string tokensPath)
    {
        var model = SenseVoiceModel.TryLoad(onnxPath);
        if (model is null) return null;
        var vocab = SenseVoiceVocab.Load(tokensPath);
        return new SenseVoicePipeline(model, vocab);
    }

    /// <summary>
    /// Real end-to-end transcription. `pcm16k` must be mono 16kHz float samples in [-1, 1].
    /// `language` is one of "auto" (default), "zh", "en", "ja", "ko", "yue" -- real default
    /// matches sherpa-onnx's own `OfflineSenseVoiceModelConfig::language = "auto"`
    /// (`offline-sense-voice-model-config.h` line 20). `useItn` (inverse text normalization, e.g.
    /// spelling out numbers/punctuation) defaults to `false`, matching the real reference's own
    /// `use_itn = false` default (same file, line 24) -- note this port does NOT implement the
    /// real reference's separate downstream `ApplyInverseTextNormalization`/
    /// `ApplyHomophoneReplacer` text post-processing passes (regex-based ITN rule application,
    /// not part of the ONNX model itself); with `useItn:false` (the default) those passes are a
    /// no-op in the real reference too, so this only matters if a caller opts into ITN.
    /// </summary>
    public SenseVoiceResult Transcribe(ReadOnlySpan<float> pcm16k, string language = "auto", bool useItn = false)
    {
        var meta = _model.MetaData;

        // normalize_samples==0 means samples should be scaled to [-32768,32767] before the
        // feature extractor sees them (real convention, confirmed from both the meta-data
        // struct's doc comment in offline-sense-voice-model-meta-data.h line 27-29, and this
        // exact checkpoint's own real metadata value read at load time).
        float waveformScale = meta.NormalizeSamples == 0 ? 32768f : 1f;

        var logMel = _melExtractor.ExtractLogMel(pcm16k, waveformScale);
        if (logMel.Length == 0)
            return new SenseVoiceResult(string.Empty, null, null, null);

        var spliced = FunAsrRealMelExtractor.ApplyLfr(logMel, meta.LfrWindowSize, meta.LfrWindowShift);
        var cmvn = FunAsrRealMelExtractor.ApplyCmvn(spliced, meta.NegMean, meta.InvStddev);

        int numFrames = cmvn.Length;
        int featDim = cmvn[0].Length;
        var flat = new float[numFrames * featDim];
        for (int t = 0; t < numFrames; t++)
            Array.Copy(cmvn[t], 0, flat, t * featDim, featDim);

        int languageId = meta.Lang2Id.TryGetValue(language, out var lid) ? lid : meta.Lang2Id["auto"];
        int textNormId = useItn ? meta.WithItnId : meta.WithoutItnId;

        var fwd = _model.Forward(flat, numFrames, featDim, languageId, textNormId);
        if (fwd is null)
            return new SenseVoiceResult(string.Empty, null, null, null);

        var (logits, outFrames, vocabSize) = fwd.Value;
        var ids = SenseVoiceCtcDecoder.GreedyDecode(logits, outFrames, vocabSize, meta.BlankId);

        return ConvertResult(ids);
    }

    /// <summary>Real `ConvertSenseVoiceResult` port (see class doc comment): the first 3 decoded
    /// tokens are the language/emotion/event tags (only meaningful when at least 3 tokens were
    /// decoded, matching the real reference's `if (src.tokens.size() >= 3)` guard), and real
    /// transcript text starts from decoded-token index 4 (`start = 4` for non-nano models).</summary>
    private SenseVoiceResult ConvertResult(int[] ids)
    {
        string? lang = null, emotion = null, evt = null;
        if (ids.Length >= 3)
        {
            lang = _vocab.TokenAt(ids[0]);
            emotion = _vocab.TokenAt(ids[1]);
            evt = _vocab.TokenAt(ids[2]);
        }

        const int start = 4;
        var textIds = ids.Length > start ? ids[start..] : [];
        string text = _vocab.Decode(textIds);

        return new SenseVoiceResult(text, lang, emotion, evt);
    }

    public void Dispose() => _model.Dispose();
}
