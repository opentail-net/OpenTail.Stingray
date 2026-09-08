using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Bit-for-bit and numerical parity tests against the reference C++ implementation
/// (examples/audio.cpp/src/models/moss/moss_tts_nano/) on real checkpoint weights.
/// Confirms tokenizer tokenization matches reference (92 prompt tokens), step 0 text logits
/// pick the assistant slot token, and step 0 local frame decoder codebook 0 produces greedy
/// token 234 matching reference audiocpp_cli execution.
/// </summary>
public sealed class MossTtsReferenceParityRealWeightsTests : HeavyTestBase
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

    private static byte[]? ExtractEmbeddedFile(GgufModel model, string fileName)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names) return null;
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
        for (int i = 0; i < names.Length; i++)
        {
            if ((string)names[i] != fileName) continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            return bytes[(int)start..(int)end];
        }
        return null;
    }

    [Fact]
    public void Step0_ReferenceNumericalParity_MatchesCppLogitsAndTokens()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");

        using var model = GgufModel.Open(path!);
        var tokBytes = ExtractEmbeddedFile(model, "tokenizer.model");
        Assert.NotNull(tokBytes);

        var tokenizer = SentencePieceBpeTokenizer.FromModelBytes(tokBytes!);
        var source = new RvcPackedTensorSource(model);
        var g = new MossTtsGlobalTransformerWeights(source);
        var l = new MossTtsLocalTransformerWeights(source);

        // 1. Prompt Tokenizer Parity: 92 tokens identical to C++ reference
        var prompt = MossTtsPromptBuilder.BuildZeroShotPrompt(tokenizer, "Hello there, this is a test of speech synthesis.");
        Assert.Equal(92, prompt.Count);
        Assert.Equal(4, prompt[0].TextId);   // im_start
        Assert.Equal(600, prompt[1].TextId); // "user"
        Assert.Equal(289, prompt[2].TextId); // "\n"
        Assert.Equal(6, prompt[^1].TextId);  // audio_start

        // 2. Global Transformer & Text Logits Parity:
        var allHidden = MossTtsGlobalTransformer.ForwardAll(g, prompt);
        var hidden = allHidden[^1];

        int bestText = MossTtsLocalFrameDecoder.PredictTextChoice(g, l, hidden);
        Assert.Equal(MossTtsGlobalTransformerWeights.AudioAssistantSlotTokenId, bestText);

        // Text logits parity: assistant logit ~ 14.0, end logit ~ 0.8
        var hiddenOutText = MossTtsGlobalTransformer.RunTransformerStack([hidden], l.Layers, l.FinalNormWeight, l.FinalNormBias);
        var textLogits = MossTtsGlobalTransformer.Linear(
            hiddenOutText[0], g.TextLmHeadWeight, bias: [],
            MossTtsGlobalTransformerWeights.HiddenDim, MossTtsGlobalTransformerWeights.VocabSize);
        Assert.InRange(textLogits[MossTtsGlobalTransformerWeights.AudioAssistantSlotTokenId], 13.5f, 14.5f);
        Assert.InRange(textLogits[MossTtsGlobalTransformerWeights.AudioEndTokenId], 0.5f, 1.2f);

        // 3. Local Frame Decoder Codebook 0 Parity:
        // C++ reference audiocpp_cli outputs max_idx = 234 (logit = 4.99)
        var frame = MossTtsLocalFrameDecoder.GenerateFrame(g, l, hidden, activeCodebooks: 1);
        Assert.NotNull(frame);
        Assert.Equal(234, frame![0]);
    }
}
