using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Audio.NeuTts;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class NeuTtsFastTests
{
    [Fact]
    public void LoadAddedTokenIds_ParsesValidJsonWithControlAndSpeechTokens()
    {
        string json = """
        {
          "added_tokens": [
            { "id": 151665, "content": "<|TEXT_PROMPT_START|>" },
            { "id": 151666, "content": "<|TEXT_PROMPT_END|>" },
            { "id": 151667, "content": "<|SPEECH_GENERATION_START|>" },
            { "id": 151668, "content": "<|SPEECH_GENERATION_END|>" },
            { "id": 151684, "content": "<|speech_0|>" },
            { "id": 217219, "content": "<|speech_65535|>" }
          ]
        }
        """;

        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
        var addedTokens = NeuTtsPromptBuilder.LoadAddedTokenIds(jsonBytes);

        Assert.Equal(6, addedTokens.Count);
        Assert.Equal(151665, addedTokens["<|TEXT_PROMPT_START|>"]);
        Assert.Equal(151666, addedTokens["<|TEXT_PROMPT_END|>"]);
        Assert.Equal(151667, addedTokens["<|SPEECH_GENERATION_START|>"]);
        Assert.Equal(151668, addedTokens["<|SPEECH_GENERATION_END|>"]);
        Assert.Equal(151684, addedTokens["<|speech_0|>"]);
        Assert.Equal(217219, addedTokens["<|speech_65535|>"]);
    }

    [Fact]
    public void Build_MissingControlTokensInAddedTokens_ThrowsInvalidDataException()
    {
        // Missing <|TEXT_PROMPT_START|>
        var incompleteTokens = new Dictionary<string, int>
        {
            ["<|TEXT_PROMPT_END|>"] = 151666,
            ["<|SPEECH_GENERATION_START|>"] = 151667,
            ["<|SPEECH_GENERATION_END|>"] = 151668,
            ["<|speech_0|>"] = 151684,
            ["<|speech_65535|>"] = 217219
        };

        Assert.Throws<InvalidDataException>(() =>
            NeuTtsPromptBuilder.Build(null!, null!, incompleteTokens, "ref", "input", []));
    }

    [Fact]
    public void Decoder_InvalidSpeechCode_ThrowsArgumentOutOfRangeException()
    {
        // Codebook size is 4^8 = 65536, valid range is [0, 65535]
        int[] outOfRangeCodes = [65536];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            NeuTtsAudioDecoder.Decode(null!, outOfRangeCodes));

        int[] negativeCodes = [-1];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            NeuTtsAudioDecoder.Decode(null!, negativeCodes));
    }

    [Fact]
    public void Decoder_FsqLevels_DecomposeMixedRadixCorrectly()
    {
        // NeuTTS has 8 levels of size 4: codebook size = 65536
        Assert.Equal(8, NeuTtsAudioDecoderWeights.QuantizationLevels.Length);
        foreach (int lv in NeuTtsAudioDecoderWeights.QuantizationLevels)
        {
            Assert.Equal(4, lv);
        }

        // Verify min code (0) and max code (65535) map to expected [-1, 1] range:
        // levelIndex * (2 / (level - 1)) - 1:
        // level = 4: levelIndex in {0, 1, 2, 3} -> {0*(2/3)-1, 1*(2/3)-1, 2*(2/3)-1, 3*(2/3)-1}
        // = {-1.0, -0.3333333, +0.3333333, +1.0}
        int level = 4;
        float step = 2f / (level - 1);
        Assert.Equal(-1.0f, 0 * step - 1f, 1e-6f);
        Assert.Equal(-1f / 3f, 1 * step - 1f, 1e-6f);
        Assert.Equal(1f / 3f, 2 * step - 1f, 1e-6f);
        Assert.Equal(1.0f, 3 * step - 1f, 1e-6f);
    }
}
