
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Throwaway dev utility: dumps the real af_heart StyleVector used by KokoroModel.Load to a raw float32 file for a Python-side end-to-end golden comparison. Not a real correctness test -- delete once the comparison is done.</summary>
public sealed class KokoroStyleVectorDumpTests
{
    [Fact]
    public void DumpAfHeartStyleVector()
    {
        string modelPath = @"C:\Git-Public\OpenTail.Stingray\models\kokoro-82m-q8_0.gguf";
        string voicePath = @"C:\Git-Public\OpenTail.Stingray\models\kokoro-voice-af_heart.gguf";
        if (!File.Exists(modelPath) || !File.Exists(voicePath)) return;

        using var weights = new KokoroWeights(modelPath, voicePath);
        Assert.NotNull(weights.VoiceTable);

        // "hello" -> 6 real phoneme characters (h,ə,l,ˈ,o,ʊ) -> real row index 6-1=5 (pack[len(ps)-1]).
        var row = weights.GetStyleVector(phonemeLength: 6)!;
        var bytes = new byte[row.Length * 4];
        Buffer.BlockCopy(row, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(@"C:\Git-Public\OpenTail.Stingray\scratch-llamacpp-ref\af_heart_style_real.f32", bytes);
    }
}
