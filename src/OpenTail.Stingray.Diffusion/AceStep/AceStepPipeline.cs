using OpenTail.Stingray.Core;
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
public sealed class AceStepPipeline : IDisposable
{
    private readonly AceStepModel _model;
    private readonly IComputeBackend? _backend;
    private readonly bool _ownsBackend;
    private bool _disposed;

    // TEMPORARY diagnostic instrumentation (perf-sweep Phase 9.1b, docs/perf-sweep-plan.md) for
    // the ACE-Step Turbo CPU perf investigation (114.14x RTF, worst in PerformanceLeague.md) --
    // no STINGRAY_PROFILE_DECODE-equivalent exists for diffusion pipelines, so this mirrors
    // HybridGdnForwardPass's own STINGRAY_PROFILE_DECODE-gated temporary profiler pattern. Remove
    // once the real bottleneck is found and fixed.
    private static readonly bool s_profEnabled =
        Environment.GetEnvironmentVariable("STINGRAY_PROFILE_DECODE") == "1";

    public AceStepPipeline(AceStepModel model, IComputeBackend? backend = null)
    {
        _model = model;
        (_backend, _ownsBackend) = DiffusionBackendResolver.Resolve(backend);
    }

    public StereoAudioBuffer Generate(AceStepGenerationParams parameters)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var swTotal = System.Diagnostics.Stopwatch.StartNew();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        // Real SFT_GEN_PROMPT template, transcribed from the real diffusers ACE-Step pipeline --
        // see docs/064-acestep-implementation-plan.md's "Corrections and confirmations".
        string prompt =
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n" +
            $"# Caption\n{parameters.Prompt}\n\n" +
            $"# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {parameters.DurationSeconds:0} seconds\n<|endoftext|>\n";

        var textHidden = _model.TextEncoder.Encode(prompt);
        double msTextEncode = sw.Elapsed.TotalMilliseconds; sw.Restart();

        // Real ACE-Step lyric format: "# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>"
        // Even for instrumental or empty lyrics, this provides the standard 6-12 conditioning tokens
        // expected by the 8-layer bidirectional lyric encoder and cross-attention sequence.
        string lyricContent = parameters.Instrumental
            ? "[Instrumental]"
            : (string.IsNullOrWhiteSpace(parameters.Lyrics) ? "" : parameters.Lyrics);
        string lyricPrompt = $"# Languages\nen\n\n# Lyric\n{lyricContent}<|endoftext|>";
        int[] lyricTokenIds = _model.TextEncoder.Tokenize(lyricPrompt);

        int latentFrames = (int)MathF.Round(parameters.DurationSeconds * 25f); // real 25Hz acoustic latent rate

        // Real "no reference audio" path: encode true digital silence through the real VAE encoder
        // to derive src_latents/timbre input, matching the real pipeline's own silence_latent-based
        // fallback (see AceStepOobleckEncoder's doc comment for why this project derives it itself).
        EnsureSilenceBuffersLoaded();

        var silenceRows = ComputeSilenceLatent(latentFrames);
        double msSilenceVaeEncode = sw.Elapsed.TotalMilliseconds; sw.Restart();

        float[] timbreRow;
        if (s_precomputedTimbre != null)
        {
            timbreRow = s_precomputedTimbre;
        }
        else
        {
            const int timbreFixFrames = 750;
            int silenceFrames = Math.Max(timbreFixFrames, latentFrames);
            var silenceRowsForTimbre = ComputeSilenceLatent(silenceFrames);
            var timbreInput = new float[Math.Min(timbreFixFrames, silenceFrames)][];
            Array.Copy(silenceRowsForTimbre, timbreInput, timbreInput.Length);
            timbreRow = AceStepTimbreEncoder.Forward(_model.TimbreEncoder, timbreInput);
        }

        var srcLatents = new float[latentFrames][];
        Array.Copy(silenceRows, srcLatents, latentFrames);


        var condition = AceStepConditionEncoder.Forward(
            _model.ConditionEncoder, textHidden, lyricTokenIds, _model.TextEncoder.TokenEmbeddingTable, timbreRow);
        double msTimbreAndCondition = sw.Elapsed.TotalMilliseconds; sw.Restart();

        var latentRows = AceStepFlowScheduler.Generate(
            _model.Transformer, condition, latentFrames, parameters.Shift, parameters.Seed, srcLatents, _backend);
        double msDiT = sw.Elapsed.TotalMilliseconds; sw.Restart();

        // AceStepFlowScheduler returns [t][acousticDim] (time-major); AceStepOobleckDecoder.Decode
        // wants [acousticDim, t] flat channel-major -- transpose.
        int acousticDim = AceStepConfig.AudioAcousticHiddenDim;
        var latentFlat = new float[acousticDim * latentFrames];
        for (int t = 0; t < latentFrames; t++)
            for (int c = 0; c < acousticDim; c++)
                latentFlat[c * latentFrames + t] = latentRows[t][c];

        var pcm = AceStepOobleckDecoder.Decode(_model.Vae, latentFlat, latentFrames);
        double msVaeDecode = sw.Elapsed.TotalMilliseconds;

        double msTotal = swTotal.Elapsed.TotalMilliseconds;
        Console.WriteLine($"[AceStep Timing] Total: {msTotal:F1}ms | TextEncode: {msTextEncode:F1}ms | SilenceVAE: {msSilenceVaeEncode:F1}ms | Timbre&Cond: {msTimbreAndCondition:F1}ms | DiT(GPU={_backend is not null}): {msDiT:F1}ms | VaeDecode: {msVaeDecode:F1}ms");


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
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, float[][]> s_silenceLatentCache = new();
    private static float[]? s_precomputedTimbre;
    private static float[][]? s_precomputedSilenceLatent;


    private static string? FindFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static Stream? OpenSilenceStream(string filename)
    {
        string? diskPath = FindFile($"models/acestep-v15/{filename}")
            ?? FindFile($"src/OpenTail.Stingray.Diffusion/AceStep/{filename}");
        if (diskPath != null && File.Exists(diskPath))
            return File.OpenRead(diskPath);

        var asm = typeof(AceStepPipeline).Assembly;
        var resName = asm.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith(filename, StringComparison.OrdinalIgnoreCase));
        if (resName != null)
            return asm.GetManifestResourceStream(resName);

        return null;
    }

    private static void EnsureSilenceBuffersLoaded()
    {
        if (s_precomputedSilenceLatent != null && s_precomputedTimbre != null) return;
        lock (s_silenceLatentCache)
        {
            if (s_precomputedSilenceLatent != null && s_precomputedTimbre != null) return;

            using (var fs = OpenSilenceStream("silence_latent.bin"))
            {
                if (fs != null)
                {
                    using var br = new BinaryReader(fs);
                    int frames = br.ReadInt32();
                    int dim = br.ReadInt32();
                    var rows = new float[frames][];
                    for (int t = 0; t < frames; t++)
                    {
                        var row = new float[dim];
                        for (int c = 0; c < dim; c++) row[c] = br.ReadSingle();
                        rows[t] = row;
                    }
                    s_precomputedSilenceLatent = rows;
                    s_silenceLatentCache[frames] = rows;
                }
            }

            using (var fs = OpenSilenceStream("silence_timbre.bin"))
            {
                if (fs != null)
                {
                    using var br = new BinaryReader(fs);
                    int len = br.ReadInt32();
                    var tRow = new float[len];
                    for (int i = 0; i < len; i++) tRow[i] = br.ReadSingle();
                    s_precomputedTimbre = tRow;
                }
            }
        }
    }

    /// <summary>Encodes real true digital silence (all-zero stereo PCM) through the real VAE encoder to derive `frames` real latent rows -- see <see cref="AceStepOobleckEncoder"/>'s doc comment.</summary>
    private float[][] ComputeSilenceLatent(int frames)
    {
        EnsureSilenceBuffersLoaded();

        if (s_silenceLatentCache.TryGetValue(frames, out var cached)) return cached;

        // If a larger master silence latent is already computed (e.g. 750 frames), slice from it
        if (s_silenceLatentCache.TryGetValue(750, out var master) && frames <= 750)
        {
            var sliced = new float[frames][];
            Array.Copy(master, sliced, frames);
            s_silenceLatentCache[frames] = sliced;
            return sliced;
        }

        if (s_precomputedSilenceLatent != null && frames <= s_precomputedSilenceLatent.Length)
        {
            var sliced = new float[frames][];
            Array.Copy(s_precomputedSilenceLatent, sliced, frames);
            s_silenceLatentCache[frames] = sliced;
            return sliced;
        }

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
        s_silenceLatentCache[frames] = rows;
        return rows;
    }


    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsBackend && _backend is IDisposable d)
        {
            d.Dispose();
        }
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
