
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end smoke test for the full Fun-ASR-Nano-2512 pipeline: real audio, real
/// mel+LFR frontend, real SAN-M encoder, real adaptor, a real ChatML prompt (system + user with
/// the real audio-placeholder token expanded to one token per audio frame, matching
/// <c>prompt.cpp</c>'s real <c>build()</c>), a real audio-conditioned embedding splice
/// (<see cref="OpenTail.Stingray.Audio.FunASR.FunAsrNanoLlmGgufTensorSource.EnableAudioConditioning"/>,
/// same technique as OmniVoice/Qwen3-ASR this session), the existing engine's real
/// <c>ForwardPass</c> Qwen3 decode loop, and real tokenizer decode. Not yet compared against a
/// captured reference trace -- this is a real, non-degenerate output check, the natural next step
/// being golden verification.</summary>
public sealed class FunAsrNanoEndToEndTests : HeavyTestBase
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

    private static string? FindRepoDir(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Transcribe_RealAudio_RealWeights_ProducesNonDegenerateText()
    {
        string? modelPath = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(modelPath != null, "models/paraformer-q8.gguf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");
        string? tokenizerDir = FindRepoDir("models/_models/fun-asr-nano-hf-tokenizer");
        Assert.SkipUnless(tokenizerDir != null, "fun-asr-nano-hf-tokenizer directory not found");

        var tokenizerResult = OpenTail.Stingray.Core.HuggingFaceTokenizerSource.Load(tokenizerDir!);
        Assert.True(tokenizerResult.IsUsable, "tokenizer failed to load: " + string.Join("; ", tokenizerResult.Rejections));
        var tokenizer = OpenTail.Stingray.Core.GgufTokenizer.FromSource(tokenizerResult.Source!);

        using var model = OpenTail.Stingray.Core.GgufModel.Open(modelPath!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var encoderWeights = new OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoderWeights(source);
        var adaptorWeights = new OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptorWeights(source);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var melExtractor = new OpenTail.Stingray.Audio.FunASR.FunAsrRealMelExtractor();
        var logMel = melExtractor.ExtractLogMel(samples);
        var lfr = OpenTail.Stingray.Audio.FunASR.FunAsrRealMelExtractor.ApplyLfr(logMel);
        int frames = Math.Min(lfr.Length, 150);
        var trimmed = new float[frames][];
        Array.Copy(lfr, trimmed, frames);

        var encoded = OpenTail.Stingray.Audio.FunASR.FunAsrNanoEncoder.Forward(encoderWeights, trimmed);
        var adapted = OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptor.Forward(adaptorWeights, encoded);
        int numAudioTokens = adapted.Length;
        var audioEmbedFlat = new float[numAudioTokens * OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptorWeights.DModel];
        for (int i = 0; i < numAudioTokens; i++) Array.Copy(adapted[i], 0, audioEmbedFlat, i * OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptorWeights.DModel, OpenTail.Stingray.Audio.FunASR.FunAsrNanoAdaptorWeights.DModel);

        // Real ChatML prompt matching prompt.cpp's build(): system + user("语音转写：" + audio
        // placeholder) + assistant generation-prompt. audio_token_id=151646 real placeholder.
        const int audioTokenId = 151646;
        string chat = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
                      "<|im_start|>user\n语音转写：<|object_ref_start|><|im_end|>\n<|im_start|>assistant\n";
        var ids = tokenizer.Encode(chat).ToList();
        int placeholderCount = ids.Count(t => t == audioTokenId);
        Assert.Equal(1, placeholderCount);

        using var llmSource = new OpenTail.Stingray.Audio.FunASR.FunAsrNanoLlmGgufTensorSource(
            model, numLayers: 28, hiddenDim: 1024, numHeads: 16, numKvHeads: 8, headDim: 128, ffDim: 3072, vocabSize: 151936, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);
        llmSource.EnableAudioConditioning(audioEmbedFlat, numAudioTokens);

        var prompt = new List<int>();
        foreach (var id in ids)
        {
            if (id == audioTokenId)
                for (int i = 0; i < numAudioTokens; i++) prompt.Add(llmSource.AudioTokenIdOffset + i);
            else
                prompt.Add(id);
        }

        var hp = OpenTail.Stingray.Core.ModelHyperparams.FromGgufMetadata(llmSource.Metadata);
        using var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwd = new OpenTail.Stingray.Engine.ForwardPass(llmSource, backend, hp);

        const int eosTokenId = 151645;
        var sampleParams = new OpenTail.Stingray.Engine.SamplingParams { Temperature = 0f };
        var rng = new Random(0);

        var logits = fwd.Prefill(prompt.ToArray());
        var emitted = new List<int>();
        int position = prompt.Count;
        const int maxNewTokens = 64;
        for (int step = 0; step < maxNewTokens; step++)
        {
            int nextToken = OpenTail.Stingray.Engine.Sampler.Sample(logits, sampleParams, rng);
            if (nextToken == eosTokenId) break;
            emitted.Add(nextToken);
            logits = fwd.Forward(nextToken, position);
            position++;
        }

        string text = tokenizer.Decode(emitted);
        Console.Error.WriteLine($"[FunAsrNanoE2E] numAudioTokens={numAudioTokens} emittedTokens={emitted.Count}");
        Console.Error.WriteLine($"[FunAsrNanoE2E] text=\"{text}\"");
        Assert.True(text.Trim().Length > 0, "decoded text is empty");
    }
}
