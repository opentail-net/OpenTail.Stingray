using OpenTail.Stingray.Audio.CosyVoice;

namespace OpenTail.Stingray.Tests.Audio;

// Scratch harness (untracked): CosyVoice2 resynthesis bisect. Real speech tokens of a clip (speech
// tokenizer v2) -> flow -> CFM -> HiFT, with no LLM, to isolate the acoustic stack. ZZ_CV2RS=<out.wav>.
public sealed class ZzCv2ResynthProfTmp
{
    [Fact]
    public void Resynth_RealTokens()
    {
        string? outPath = Environment.GetEnvironmentVariable("ZZ_CV2RS");
        if (outPath is null) return;
        string m = @"C:\Git-Public\OpenTail.Stingray\models\_models";
        string wav = Environment.GetEnvironmentVariable("ZZ_WAV") ?? @"C:\Git-Public\OpenTail.Stingray\examples\audio.cpp\assets\resources\a.wav";

        var (samples, sr, _) = WavReader.ReadWav(wav);
        var s16 = sr == 16000 ? samples : AudioResampler.Resample(samples, sr, 16000);
        var tokens = CosyVoiceSpeechTokenizer.Extract(Path.Combine(m, "cosyvoice_speech_tokenizer_v2.onnx"), s16)!;
        Console.WriteLine($"[ZZ cv2rs] {tokens.Length} tokens: {string.Join(",", tokens.Take(20))}...");

        using var flow = new CosyVoiceFlowWeights(Path.Combine(m, "cosyvoice2_flow.safetensors"));
        var cfm = new CosyVoiceCfmDecoderWeights(flow);
        using var hift = new CosyVoiceHiftWeights(Path.Combine(m, "cosyvoice2_hift.safetensors"));

        // ZZ_LLM=1: replace the real tokens with LLM-generated ones for ZZ_TEXT (no prompt), same acoustic stack.
        if (Environment.GetEnvironmentVariable("ZZ_LLM") == "1")
        {
            using var llm = new CosyVoiceLlmTensorSource(Path.Combine(m, "cosyvoice2_llm.safetensors"),
                numLayers: 24, hiddenDim: 896, numHeads: 14, numKvHeads: 2, headDim: 64, ffDim: 4864, vocabSize: 151936, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);
            string tokDir = @"C:\Git-Public\OpenTail.Stingray\models\cosyvoice2_tokenizer";
            tokens = CosyVoiceLlmGeneration.GenerateSpeechTokens(llm, tokDir, Environment.GetEnvironmentVariable("ZZ_TEXT") ?? "This is a test of voice synthesis.",
                rng: new Random(int.TryParse(Environment.GetEnvironmentVariable("ZZ_SEED"), out var sd) ? sd : 42),
                repetitionPenalty: float.TryParse(Environment.GetEnvironmentVariable("ZZ_REPPEN"), System.Globalization.CultureInfo.InvariantCulture, out var rp) ? rp : 1.0f);
            Console.WriteLine($"[ZZ cv2rs] LLM tokens {tokens.Length}: {string.Join(",", tokens.Take(20))}...");
        }
        var (mu, frames) = CosyVoiceFlowEncoder.Forward(flow, [], tokens);
        var xv = new float[flow.SpkEmbedDim];
        if (Environment.GetEnvironmentVariable("ZZ_XVEC") == "1") xv = CamPlusSpeakerEncoder.Extract(Path.Combine(m, "campplus.onnx"), s16)!;
        int steps = int.TryParse(Environment.GetEnvironmentVariable("ZZ_STEPS"), out var st) ? st : 10;
        var spk = CosyVoiceFlowEncoder.ProjectSpeakerEmbedding(flow, xv);
        var rng = new Random(1);
        var mel = CosyVoiceCfmDecoder.Generate(cfm, mu, spk, frames, rng, steps);
        var pcm = CosyVoiceHiftVocoder.Generate(hift, mel, frames, rng);
        new OpenTail.Stingray.Audio.AudioGenerationResult(pcm, 24000).SaveWav(outPath);
        Console.WriteLine($"[ZZ cv2rs] {frames} frames -> {pcm.Length / 24000.0:F2}s -> {outPath}");
    }
}
