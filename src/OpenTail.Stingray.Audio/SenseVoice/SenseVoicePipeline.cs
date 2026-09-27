using System.Runtime.CompilerServices;
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
public sealed class SenseVoicePipeline : ISpeechToTextPipeline
{
    public string Architecture => "SenseVoice-Small-ONNX";
    public int SampleRate => 16000;

    private readonly SenseVoiceModel _model;
    private readonly SenseVoiceVocab _vocab;
    private readonly FunAsrRealMelExtractor _melExtractor = new();

    public SenseVoicePipeline(SenseVoiceModel model, SenseVoiceVocab vocab)
    {
        _model = model;
        _vocab = vocab;
    }

    /// <summary>Tokens file next to the model: <c>&lt;name&gt;-tokens.txt</c> (e.g.
    /// <c>sensevoice-small.int8.onnx</c> -> <c>sensevoice-small-tokens.txt</c>) or <c>tokens.txt</c>.</summary>
    public static string? ResolveTokensPath(string onnxPath)
    {
        string dir = Path.GetDirectoryName(Path.GetFullPath(onnxPath)) ?? ".";
        string stem = Path.GetFileName(onnxPath).Split('.')[0];
        foreach (var candidate in new[] { Path.Combine(dir, $"{stem}-tokens.txt"), Path.Combine(dir, "tokens.txt") })
            if (File.Exists(candidate)) return candidate;
        return null;
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

    /// <summary><see cref="ISpeechToTextPipeline"/> entry: resamples to 16 kHz, maps <c>request.Language</c> (null =
    /// auto) and returns the whole utterance as one segment. The detected language tag (e.g. <c>&lt;|en|&gt;</c>) is
    /// reported as the result language.</summary>
    public SpeechToTextResult Transcribe(SpeechToTextRequest request)
    {
        if (request.AudioSamples is null || request.AudioSamples.Length == 0)
            return new SpeechToTextResult(string.Empty, request.Language ?? "auto", TimeSpan.Zero, []);
        float[] pcm16k = request.AudioSamples;
        if (request.SampleRate != SampleRate)
            pcm16k = AudioResampler.Resample(pcm16k, request.SampleRate, SampleRate);
        var duration = TimeSpan.FromSeconds((double)pcm16k.Length / SampleRate);

        var r = Transcribe(pcm16k, string.IsNullOrEmpty(request.Language) ? "auto" : request.Language);
        string lang = r.Language?.Trim('<', '|', '>') is { Length: > 0 } l ? l : request.Language ?? "auto";
        string text = r.Text.Trim();
        var segment = new SpeechSegment { Id = 0, Start = TimeSpan.Zero, End = duration, Text = text, Probability = 1.0f };
        return new SpeechToTextResult(text, lang, duration, [segment]);
    }

    /// <summary>SenseVoice-Small is an offline, whole-utterance CTC model: buffer the stream, then one pass.</summary>
    public async IAsyncEnumerable<SpeechSegment> TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audioStream,
        SpeechToTextRequest baseRequest,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var buffer = new List<float>();
        await foreach (var chunk in audioStream.WithCancellation(ct))
            buffer.AddRange(chunk.ToArray());
        if (buffer.Count == 0) yield break;
        foreach (var seg in Transcribe(baseRequest with { AudioSamples = [.. buffer] }).Segments)
            yield return seg;
    }

    public void Dispose() => _model.Dispose();
}
