using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using OpenTail.Stingray.Diffusion.AceStep.Vae;

namespace OpenTail.Stingray.Diffusion.AceStep;

/// <summary>
/// Top-level ACE-Step Turbo text-to-music pipeline: real V1 end-to-end wiring (text+lyrics, plus a
/// real self-derived "no reference audio" timbre condition -- no actual reference-audio/cover/
/// repaint support) -- see docs/064-acestep-implementation-plan.md for each component's own
/// golden-parity verification against the real `diffusers` reference; this class only wires them
/// together, it introduces no new math of its own beyond the real SFT prompt template and the
/// latent-to-channel-major-PCM layout conversions each component already documents.
///
/// <para>Real flow (from `AceStepConditionGenerationModel.generate_audio`): encode prompt via
/// Qwen3 (causal, full model, final-RMSNorm'd `last_hidden_state`) -&gt; derive a real
/// `silence_latent` by encoding true digital silence through <see cref="AceStepOobleckEncoder"/>
/// (self-sufficient substitute for the real checkpoint's missing `silence_latent` buffer -- see
/// that class's doc comment) -&gt; condition encoder packs `[lyric, timbre, text]` into one
/// cross-attention sequence, where `timbre` is the real pooled embedding of that silence latent
/// (matches the real pipeline's own "no reference audio" path, NOT a placeholder) -&gt;
/// <see cref="AceStepFlowScheduler"/> runs the real hardcoded shift-1/2/3 Euler-ODE schedule
/// through the DiT, using that same silence latent as `src_latents` (also matching the real
/// pipeline's own "no reference audio" path) -&gt; the real `AutoencoderOobleck` VAE decodes the
/// resulting 25Hz latent to 48kHz stereo PCM.</para>
/// </summary>
public sealed class AceStepPipeline
{
    private readonly AceStepModel _model;

    // TEMPORARY diagnostic instrumentation (perf-sweep Phase 9.1b, docs/perf-sweep-plan.md) for
    // the ACE-Step Turbo CPU perf investigation (114.14x RTF, worst in PerformanceLeague.md) --
    // no STINGRAY_PROFILE_DECODE-equivalent exists for diffusion pipelines, so this mirrors
    // HybridGdnForwardPass's own STINGRAY_PROFILE_DECODE-gated temporary profiler pattern. Remove
    // once the real bottleneck is found and fixed.
    private static readonly bool s_profEnabled =
        Environment.GetEnvironmentVariable("STINGRAY_PROFILE_DECODE") == "1";

    public AceStepPipeline(AceStepModel model)
    {
        _model = model;
    }

    public StereoAudioBuffer Generate(AceStepGenerationParams parameters)
    {
        var swTotal = s_profEnabled ? System.Diagnostics.Stopwatch.StartNew() : null;
        var sw = s_profEnabled ? System.Diagnostics.Stopwatch.StartNew() : null;
        // Real SFT_GEN_PROMPT template, transcribed from the real diffusers ACE-Step pipeline --
        // see docs/064-acestep-implementation-plan.md's "Corrections and confirmations".
        string prompt =
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n" +
            $"# Caption\n{parameters.Prompt}\n\n" +
            $"# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {parameters.DurationSeconds:0} seconds\n<|endoftext|>\n";

        var textHidden = _model.TextEncoder.Encode(prompt);
        double msTextEncode = sw?.Elapsed.TotalMilliseconds ?? 0; sw?.Restart();

        int[] lyricTokenIds = parameters.Instrumental || string.IsNullOrWhiteSpace(parameters.Lyrics)
            ? []
            : _model.TextEncoder.Tokenize(parameters.Lyrics);

        int latentFrames = (int)MathF.Round(parameters.DurationSeconds * 25f); // real 25Hz acoustic latent rate

        // Real "no reference audio" path: encode true digital silence through the real VAE encoder
        // to derive src_latents/timbre input, matching the real pipeline's own silence_latent-based
        // fallback (see AceStepOobleckEncoder's doc comment for why this project derives it itself).
        const int timbreFixFrames = 750; // real `timbre_fix_frame = ceil(30 * 25Hz)`
        int silenceFrames = Math.Max(timbreFixFrames, latentFrames);
        var silenceRows = ComputeSilenceLatent(silenceFrames);
        double msSilenceVaeEncode = sw?.Elapsed.TotalMilliseconds ?? 0; sw?.Restart();

        var timbreInput = new float[Math.Min(timbreFixFrames, silenceFrames)][];
        Array.Copy(silenceRows, timbreInput, timbreInput.Length);
        var timbreRow = AceStepTimbreEncoder.Forward(_model.TimbreEncoder, timbreInput);

        var srcLatents = new float[latentFrames][];
        Array.Copy(silenceRows, srcLatents, latentFrames);

        var condition = AceStepConditionEncoder.Forward(
            _model.ConditionEncoder, textHidden, lyricTokenIds, _model.TextEncoder.TokenEmbeddingTable, timbreRow);
        double msTimbreAndCondition = sw?.Elapsed.TotalMilliseconds ?? 0; sw?.Restart();

        var latentRows = AceStepFlowScheduler.Generate(
            _model.Transformer, condition, latentFrames, parameters.Shift, parameters.Seed, srcLatents);
        double msDiT = sw?.Elapsed.TotalMilliseconds ?? 0; sw?.Restart();

        // AceStepFlowScheduler returns [t][acousticDim] (time-major); AceStepOobleckDecoder.Decode
        // wants [acousticDim, t] flat channel-major -- transpose.
        int acousticDim = AceStepConfig.AudioAcousticHiddenDim;
        var latentFlat = new float[acousticDim * latentFrames];
        for (int t = 0; t < latentFrames; t++)
            for (int c = 0; c < acousticDim; c++)
                latentFlat[c * latentFrames + t] = latentRows[t][c];

        var pcm = AceStepOobleckDecoder.Decode(_model.Vae, latentFlat, latentFrames);
        double msVaeDecode = sw?.Elapsed.TotalMilliseconds ?? 0;

        if (s_profEnabled)
        {
            double msTotal = swTotal!.Elapsed.TotalMilliseconds;
            Console.Error.WriteLine("[AceStepProfile] Stage split (TEMPORARY diagnostic, see docs/perf-sweep-plan.md Phase 9):");
            Console.Error.WriteLine($"  Text encoder (Qwen3)     {msTextEncode,10:F2}ms  {100.0 * msTextEncode / msTotal,6:F2}%");
            Console.Error.WriteLine($"  Silence VAE encode       {msSilenceVaeEncode,10:F2}ms  {100.0 * msSilenceVaeEncode / msTotal,6:F2}%");
            Console.Error.WriteLine($"  Timbre + condition enc.  {msTimbreAndCondition,10:F2}ms  {100.0 * msTimbreAndCondition / msTotal,6:F2}%");
            Console.Error.WriteLine($"  DiT flow scheduler       {msDiT,10:F2}ms  {100.0 * msDiT / msTotal,6:F2}%");
            Console.Error.WriteLine($"  VAE decode               {msVaeDecode,10:F2}ms  {100.0 * msVaeDecode / msTotal,6:F2}%");
            Console.Error.WriteLine($"  Total                    {msTotal,10:F2}ms");
        }

        int samplesPerChannel = pcm.Length / AceStepConfig.VaeAudioChannels;
        var left = new float[samplesPerChannel];
        var right = new float[samplesPerChannel];
        Array.Copy(pcm, 0, left, 0, samplesPerChannel);
        Array.Copy(pcm, samplesPerChannel, right, 0, samplesPerChannel);

        return new StereoAudioBuffer
        {
            SampleRate = AceStepConfig.VaeSampleRate,
            Left = left,
            Right = right,
        };
    }

    // perf-sweep Phase 9 (docs/perf-sweep-plan.md): real profiling found this stage alone was
    // 83-85% of total Generate() wall-clock (175-186s of a ~214s mean run) -- vastly more than
    // the actual DiT diffusion transformer (~10%). The input is ALWAYS the same all-zero silence
    // PCM for a given `frames` count (a pure function of duration alone, independent of prompt/
    // lyrics/seed/timbre) -- there is nothing to recompute across calls with the same duration.
    // Memoized per pipeline instance (correct: `_model.VaeEncoder` weights are fixed for the
    // instance's lifetime, so this is a true idempotent cache, not a correctness risk).
    private readonly Dictionary<int, float[][]> _silenceLatentCache = new();

    /// <summary>Encodes real true digital silence (all-zero stereo PCM) through the real VAE encoder to derive `frames` real latent rows -- see <see cref="AceStepOobleckEncoder"/>'s doc comment.</summary>
    private float[][] ComputeSilenceLatent(int frames)
    {
        if (_silenceLatentCache.TryGetValue(frames, out var cached)) return cached;

        int hopLength = AceStepConfig.VaeDownsamplingRatios.Aggregate(1, (a, b) => a * b);
        int sampleCount = frames * hopLength;
        var zeroPcm = new float[AceStepConfig.VaeAudioChannels * sampleCount]; // real true silence

        var flat = AceStepOobleckEncoder.EncodeMode(_model.VaeEncoder, zeroPcm, AceStepConfig.VaeAudioChannels, sampleCount);

        int latentDim = AceStepConfig.VaeDecoderInputChannels;
        var rows = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var row = new float[latentDim];
            for (int c = 0; c < latentDim; c++) row[c] = flat[c * frames + t];
            rows[t] = row;
        }
        _silenceLatentCache[frames] = rows;
        return rows;
    }
}

/// <summary>Shared stereo audio buffer shape -- intended to eventually also be used by MusicGen/AudioGen's mono output and any future stereo audio model, per this plan's recommendation, but not yet wired into those (both remain mono `float[]` today; retrofitting them is a separate, deliberate follow-up, not bundled into this scaffold).</summary>
public readonly struct StereoAudioBuffer
{
    public required int SampleRate { get; init; }
    public required float[] Left { get; init; }
    public required float[] Right { get; init; }

    public int SampleCount => Left.Length;
    public double DurationSeconds => SampleCount / (double)SampleRate;
}
