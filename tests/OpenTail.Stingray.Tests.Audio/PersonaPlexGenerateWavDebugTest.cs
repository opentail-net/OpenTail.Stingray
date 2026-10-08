using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: generates a real PersonaPlex wav end-to-end for informal
/// listening (no CLI wiring yet). Writes to docs/audio-samples (gitignored, local-only per
/// CLAUDE.md). Not a golden/parity test. Uses the real voice-id-conditioned bootstrap
/// (<see cref="PersonaPlexGenerator.GenerateWithVoicePrompt"/>) so the sample reflects a real
/// voice, not the plain self-predicting cold-start path.</summary>
public sealed class PersonaPlexGenerateWavDebugTest : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private const int NumLayers = 32, HiddenDim = 4096, NumHeads = 32, HeadDim = 128;
    private const int FfDim = 11264, TextVocabSize = 32000, LmCodebooks = 16, AudioCodebookSize = 2048;
    private const float RopeTheta = 10000f, RmsNormEps = 1e-8f;
    private const float MimiFrameRate = 12.5f;
    private const int OutputSampleRate = 24000;

    private static string ExtractVoicePrompt(GgufModel model)
    {
        const string wanted = "voices_safetensors/NATF0.safetensors";
        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, wanted, out var bytes))
        {
            string path = Path.Combine(Path.GetTempPath(), "stingray-personaplex-voice-natf0.safetensors");
            File.WriteAllBytes(path, bytes);
            return path;
        }
        throw new InvalidOperationException($"PersonaPlex packed GGUF is missing embedded file '{wanted}'.");
    }

    private static byte[] ExtractEmbeddedFile(GgufModel model, string fileName)
    {
        if (OpenTail.Stingray.Audio.AudioCppEmbeddedFiles.TryGet(model, fileName, out var bytes))
        {
            return bytes;
        }
        throw new InvalidOperationException($"PersonaPlex packed GGUF is missing embedded file '{fileName}'.");
    }

    [Fact]
    public void Generate_RealPersonaPlexWav()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");
        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/00-current-work.md"));
        Assert.NotNull(repoRoot);

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        string voicePath = ExtractVoicePrompt(model);
        var voicePrompt = PersonaPlexVoicePrompt.Load(voicePath, HiddenDim);

        using var llm = new PersonaPlexLmTensorSource(source, NumLayers, HiddenDim, NumHeads, HeadDim, FfDim, TextVocabSize, LmCodebooks, AudioCodebookSize, RopeTheta, RmsNormEps);
        var hp = OpenTail.Stingray.Engine.ArchitectureModelResolver.ResolveHyperparams(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var depformerWeights = PersonaPlexDepformerWeights.Load(TextVocabSize, AudioCodebookSize, source.GetTensor);
        var depformer = new PersonaPlexDepformer(depformerWeights);

        var tokenizerBytes = ExtractEmbeddedFile(model, "tokenizer_spm_32k_3.model");
        var tokenizer = PersonaPlexSentencePieceModel.Load(tokenizerBytes);

        var textOptions = new SamplingParams { Temperature = 0.7f, TopK = 25, TopP = 1.0f };
        var audioOptions = new SamplingParams { Temperature = 0.8f, TopK = 250, TopP = 1.0f };
        var rng = new Random(int.TryParse(Environment.GetEnvironmentVariable("PP_SEED"), out var sd) ? sd : 42424242);

        var frames = PersonaPlexGenerator.GenerateWithVoicePrompt(
            fwd, llm, depformer, voicePrompt, MimiFrameRate,
            systemPrompt: "You are a wise and friendly assistant. Speak clearly and introduce yourself.", tokenizer,
            numOutputFrames: int.TryParse(Environment.GetEnvironmentVariable("PP_FRAMES"), out var nf) ? nf : 50, TextVocabSize, AudioCodebookSize,
            textOptions: textOptions, audioOptions: audioOptions, rng: rng);

        Assert.True(frames.Length > 0);

        var mimiWeights = MimiCodecDecoderWeights.Load(source.GetTensor);
        var mimiCodes = frames.Select(f => f.AudioCodes).ToArray();
        var waveform = MimiCodecDecoder.Decode(mimiWeights, mimiCodes);
        Assert.True(waveform.Length > 0);

        var textTokens = frames.Select(f => f.TextToken).ToArray();
        string decodedText = string.Join("", textTokens.Where(id => id >= 0 && id < tokenizer.Pieces.Count).Select(id => tokenizer.Pieces[id])).Replace('\u2581', ' ').Trim();
        string txtPath = Path.Combine(repoRoot!, "audio-samples", "personaplex-real-check.txt");
        File.WriteAllText(txtPath, $"Decoded text: {decodedText}\nTokens: [{string.Join(", ", textTokens)}]");

        var wavResult = new OpenTail.Stingray.Audio.AudioGenerationResult(waveform, OutputSampleRate);
        string outPath = Environment.GetEnvironmentVariable("PP_OUT") ?? Path.Combine(repoRoot!, "audio-samples", "personaplex-real-check.wav");
        wavResult.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {waveform.Length} samples, {waveform.Length / (double)OutputSampleRate:F2}s. Decoded LM text: {decodedText}");
    }
}
