using OpenTail.Stingray.Audio.CosyVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Diagnostic for the open CosyVoice2 drift: teacher-force the real speech tokens of a.wav (speech tokenizer v2) after
/// its own transcript and report how the LLM ranks each true next token, in position quartiles.
/// </summary>
public sealed class CosyVoice2TeacherForcingTests
{
    [Fact]
    public void RealSpeechTokens_AreRankedHighly_ThroughoutTheUtterance()
    {
        const string m = @"C:\Git-Public\OpenTail.Stingray\models\_models";
        string wav = @"C:\Git-Public\OpenTail.Stingray\examples\audio.cpp\assets\resources\a.wav";
        string llmPath = Path.Combine(m, "cosyvoice2_llm.safetensors");
        string tokDir = @"C:\Git-Public\OpenTail.Stingray\models\cosyvoice2_tokenizer";
        Assert.SkipUnless(File.Exists(llmPath) && File.Exists(wav) && Directory.Exists(tokDir), "CosyVoice2 checkpoint, tokenizer or a.wav not found");

        var (samples, sr, _) = WavReader.ReadWav(wav);
        var s16 = sr == 16000 ? samples : AudioResampler.Resample(samples, sr, 16000);
        var tokens = CosyVoiceSpeechTokenizer.Extract(Path.Combine(m, "cosyvoice_speech_tokenizer_v2.onnx"), s16)!;
        using var llm = new CosyVoiceLlmTensorSource(llmPath, numLayers: 24, hiddenDim: 896, numHeads: 14, numKvHeads: 2, headDim: 64,
            ffDim: 4864, vocabSize: 151936, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);
        var scores = CosyVoiceLlmGeneration.ScoreSpeechTokens(llm, tokDir,
            "This little work was finished in 1803 and intended for a media publication.", tokens);

        int q = Math.Max(1, scores.Length / 4);
        for (int k = 0; k < 4; k++)
        {
            var part = scores.Skip(k * q).Take(k == 3 ? scores.Length - 3 * q : q).ToArray();
            Console.WriteLine($"[Cv2TF] quartile {k + 1} ({part.Length} tokens): top1 {part.Count(s => s.Rank == 0) * 100 / part.Length}%, " +
                $"top10 {part.Count(s => s.Rank < 10) * 100 / part.Length}%, mean logp {part.Average(s => s.LogProb):F2}, median rank {part.Select(s => s.Rank).Order().ElementAt(part.Length / 2)}");
        }
        Console.WriteLine($"[Cv2TF] first 20 ranks: {string.Join(",", scores.Take(20).Select(s => s.Rank))}");
    }
}
