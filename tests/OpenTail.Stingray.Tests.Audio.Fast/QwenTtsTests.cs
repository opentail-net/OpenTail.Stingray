using OpenTail.Stingray.Audio.QwenTTS;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class QwenTtsTests
{
    [Fact]
    public void QwenTtsTokenizer_FormatAndEncode_GeneratesChatMlPromptWithSpeakerAndDialect()
    {
        var tokenizer = new QwenTtsTokenizer();
        string text = "Hello from Qwen3-TTS native port!";

        // Test custom speaker dialect resolution
        string ericPrompt = tokenizer.FormatPrompt(text, "eric");
        Assert.Contains("sichuan", ericPrompt, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("<|im_start|>", ericPrompt, StringComparison.Ordinal);
        Assert.Contains("<|tts_bos|>", ericPrompt, StringComparison.Ordinal);

        int[] tokens = tokenizer.Encode(ericPrompt);
        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void QwenTtsTalkerLm_GenerateCode0_EmitsSemanticCode0AndHiddenStates()
    {
        using var talker = new QwenTtsTalkerLm();
        int[] promptTokens = [10, 20, 30, 40, 50];
        int[] refCodes = [100, 200];

        var result = talker.GenerateCode0(promptTokens, refCodes, maxFrames: 30, speed: 1.0f, seed: 42);
        int[] code0 = result.Code0Tokens;
        float[] hiddenStates = result.HiddenStates;

        Assert.NotNull(code0);
        Assert.NotEmpty(code0);
        Assert.NotNull(hiddenStates);
        Assert.Equal(code0.Length * talker.Config.HiddenDim, hiddenStates.Length);

        foreach (int c in code0)
        {
            Assert.InRange(c, 0, talker.Config.CodebookSize - 1);
        }
    }

    [Fact]
    public void QwenTtsCodePredictor_PredictAllCodebooks_Produces16RvqCodebooks()
    {
        using var predictor = new QwenTtsCodePredictor();
        int numFrames = 8;
        var code0 = new int[numFrames];
        for (int i = 0; i < numFrames; i++) code0[i] = (i * 37) % 2048;

        var hidden = new float[numFrames * 1024];

        int[] allCodes = predictor.PredictAllCodebooks(code0, hidden, talkerHiddenDim: 1024, seed: 99);

        Assert.NotNull(allCodes);
        Assert.Equal(16 * numFrames, allCodes.Length);

        for (int cb = 0; cb < 16; cb++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                int code = allCodes[cb * numFrames + f];
                Assert.InRange(code, 0, predictor.Config.CodebookSize - 1);
            }
        }
    }

    [Fact]
    public void QwenTtsDacDecoder_Decode_Synthesizes24kHzAudioWaveform()
    {
        using var decoder = new QwenTtsDacDecoder();
        int numFrames = 5;
        var rvqCodes = new int[16 * numFrames];
        for (int i = 0; i < rvqCodes.Length; i++)
        {
            rvqCodes[i] = (i * 19) % 2048;
        }

        float[] audio = decoder.Decode(rvqCodes, numFrames);

        Assert.NotNull(audio);
        Assert.Equal(numFrames * decoder.Config.TotalUpsampleFactor, audio.Length);

        for (int i = 0; i < audio.Length; i++)
        {
            Assert.False(float.IsNaN(audio[i]), $"NaN at sample {i}");
            Assert.False(float.IsInfinity(audio[i]), $"Infinity at sample {i}");
            Assert.InRange(audio[i], -1.0f, 1.0f);
        }
    }
}
