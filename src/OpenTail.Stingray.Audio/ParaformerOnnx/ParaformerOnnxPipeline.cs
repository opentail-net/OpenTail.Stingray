using System.Runtime.CompilerServices;
using OpenTail.Stingray.Audio.FunASR;

namespace OpenTail.Stingray.Audio.ParaformerOnnx;

/// <summary>
/// Real, independent ONNX-based FunASR Paraformer ASR pipeline for the local
/// `models/_models/paraformer-zh-small.int8.onnx` checkpoint. Independent of the native GGUF
/// Paraformer path in `OpenTail.Stingray.Audio.FunASR` (that path is confirmed broken this
/// session -- `FunAsrWeights.cs` throws on missing `pf.vocab` GGUF metadata, a bad conversion, not
/// a code bug). This path calls the checkpoint's own fused ONNX graph directly instead.
///
/// Real pipeline stages, each cross-referenced against `examples/sherpa-onnx`'s real C++ source
/// (not guessed):
/// 1. Kaldi-style log-mel fbank -- reuses <see cref="FunAsrRealMelExtractor"/> read-only (real
///    fs=16000/hamming/80-mel/25ms-frame/10ms-shift, matches
///    `offline-recognizer-paraformer-impl.h::InitFeatConfig` exactly: `window_type="hamming"`,
///    `snip_edges=true`, `normalize_samples=false` i.e. int16-range waveform scale -- the
///    extractor's own default `waveformScale=32768` already matches this).
/// 2. LFR splice (`ApplyLfr`, real `lfr_window_size`/`lfr_window_shift` read from THIS
///    checkpoint's own ONNX custom metadata, not hardcoded -- see <see cref="ParaformerOnnxModel"/>).
/// 3. CMVN (`ApplyCmvn`, real `neg_mean`/`inv_stddev` read from THIS checkpoint's own metadata).
/// 4. One ONNX forward pass (`speech`+`speech_lengths` in, `logits` out) -- the whole
///    encoder+CIF-predictor+decoder is fused into the exported graph, no architecture to
///    reimplement.
/// 5. Real per-position-argmax-with-EOS-stop greedy decode (`ParaformerOnnxDecoder`, NOT
///    CTC-collapse -- a different algorithm from SenseVoice's CTC decode, ported from a different
///    real reference file).
/// 6. Real vocab lookup + BPE/CJK text assembly (`ParaformerOnnxVocab.Convert`).
/// </summary>
public sealed class ParaformerOnnxPipeline : ISpeechToTextPipeline
{
    public string Architecture => "FunASR-Paraformer-ONNX";
    public int SampleRate => 16000;

    private readonly ParaformerOnnxModel _model;
    private readonly ParaformerOnnxVocab _vocab;
    private readonly FunAsrRealMelExtractor _melExtractor;
    private const int FeatureDim = 80;

    public ParaformerOnnxPipeline(ParaformerOnnxModel model, ParaformerOnnxVocab vocab)
    {
        _model = model;
        _vocab = vocab;
        _melExtractor = new FunAsrRealMelExtractor();
    }

    /// <summary>
    /// Loads the real ONNX checkpoint plus its matching real `tokens.txt`. `tokensPath` defaults
    /// to `F:\_models\paraformer-zh-small-tokens.txt` (this session's real downloaded vocab for
    /// this exact checkpoint) if not given, falling back to a `.tokens.txt` sibling of the model
    /// file.
    /// </summary>
    public static ParaformerOnnxPipeline Load(string onnxPath, string? tokensPath = null)
    {
        var model = ParaformerOnnxModel.TryLoad(onnxPath)
            ?? throw new InvalidOperationException($"Failed to load Paraformer ONNX model or its custom metadata: {onnxPath}");

        tokensPath ??= ResolveTokensPath(onnxPath);
        if (tokensPath is null || !File.Exists(tokensPath))
        {
            model.Dispose();
            throw new FileNotFoundException(
                "Paraformer tokens.txt not found. Expected at F:\\_models\\paraformer-zh-small-tokens.txt " +
                "or as a sibling of the ONNX model file.", tokensPath);
        }

        var vocab = ParaformerOnnxVocab.Load(tokensPath);
        return new ParaformerOnnxPipeline(model, vocab);
    }

    private static string? ResolveTokensPath(string onnxPath)
    {
        const string wellKnown = @"F:\_models\paraformer-zh-small-tokens.txt";
        if (File.Exists(wellKnown)) return wellKnown;

        var dir = Path.GetDirectoryName(onnxPath);
        if (dir is null) return null;
        var sibling = Path.Combine(dir, "paraformer-zh-small-tokens.txt");
        return File.Exists(sibling) ? sibling : null;
    }

    public SpeechToTextResult Transcribe(SpeechToTextRequest request)
    {
        if (request.AudioSamples is null || request.AudioSamples.Length == 0)
            return new SpeechToTextResult(string.Empty, request.Language ?? "zh", TimeSpan.Zero, []);

        float[] pcm16k = request.AudioSamples;
        if (request.SampleRate != SampleRate)
            pcm16k = AudioResampler.Resample(pcm16k, request.SampleRate, SampleRate);

        var totalDuration = TimeSpan.FromSeconds((double)pcm16k.Length / SampleRate);

        var logMel = _melExtractor.ExtractLogMel(pcm16k);
        if (logMel.Length == 0)
            return new SpeechToTextResult(string.Empty, request.Language ?? "zh", totalDuration, []);

        var spliced = FunAsrRealMelExtractor.ApplyLfr(logMel, _model.LfrWindowSize, _model.LfrWindowShift);
        var normalized = FunAsrRealMelExtractor.ApplyCmvn(spliced, _model.NegMean, _model.InvStdDev);

        int numFrames = normalized.Length;
        int featDim = FeatureDim * _model.LfrWindowSize;
        var flat = new float[numFrames * featDim];
        for (int t = 0; t < numFrames; t++)
            Array.Copy(normalized[t], 0, flat, t * featDim, featDim);

        var logProbs = _model.Forward(flat, numFrames, featDim);
        var tokenIds = ParaformerOnnxDecoder.GreedyDecode(logProbs, _vocab.EosId);
        string text = _vocab.Convert(tokenIds).Trim();

        var segment = new SpeechSegment
        {
            Id = 0,
            Start = TimeSpan.Zero,
            End = totalDuration,
            Text = text,
            Tokens = [.. tokenIds],
            Probability = 1.0f,
        };

        return new SpeechToTextResult(text, request.Language ?? "zh", totalDuration, [segment]);
    }

    public async IAsyncEnumerable<SpeechSegment> TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audioStream,
        SpeechToTextRequest baseRequest,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        // Paraformer is a non-autoregressive, fixed-utterance model (the CIF predictor inside the
        // fused ONNX graph needs the whole utterance to determine output length) -- there is no
        // real incremental/streaming variant of this exact checkpoint's architecture. Buffer the
        // whole stream and run one real offline pass, matching the real reference's own
        // `OfflineRecognizerParaformerImpl` (offline-only, no streaming counterpart in sherpa-onnx
        // for this model class).
        var buffer = new List<float>();
        await foreach (var chunk in audioStream.WithCancellation(ct))
            buffer.AddRange(chunk.ToArray());

        if (buffer.Count == 0) yield break;

        var req = baseRequest with { AudioSamples = [.. buffer] };
        var res = Transcribe(req);
        foreach (var seg in res.Segments)
            yield return seg;
    }

    public void Dispose() => _model.Dispose();
}
