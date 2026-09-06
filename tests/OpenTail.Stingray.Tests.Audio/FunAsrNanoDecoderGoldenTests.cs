
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real golden-verification test for the Fun-ASR-Nano-2512 LLM decode path in ISOLATION
/// from the audio encoder/adaptor -- uses `examples/audio.cpp/tests/fun_asr_nano/decoder_reference.
/// {json,bin}`'s real fixed audio embeddings (2 synthetic 1024-dim frames, NOT derived from real
/// audio) and real expected `input_ids`/`greedy_token_ids`/`generated_text`, generated directly
/// from the real `transformers` model. This isolates whether the prompt construction + audio-token
/// splice + `ForwardPass` decode loop are correct, independent of whether the encoder/adaptor
/// produce numerically-perfect audio embeddings.</summary>
public sealed class FunAsrNanoDecoderGoldenTests : HeavyTestBase
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
    public void Decode_RealAudioEmbeddingsFromFixture_MatchesRealGreedyTokenIds()
    {
        string? modelPath = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(modelPath != null, "models/paraformer-q8.gguf not found");
        string? jsonPath = FindRepoFile("examples/audio.cpp/tests/fun_asr_nano/decoder_reference.json");
        string? binPath = FindRepoFile("examples/audio.cpp/tests/fun_asr_nano/decoder_reference.bin");
        Assert.SkipUnless(jsonPath != null && binPath != null, "decoder_reference.{json,bin} not found");

        var data = ReadAllF32(binPath!);
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(jsonPath!));
        var root = doc.RootElement;

        var itn = root.GetProperty("itn");
        var inputIds = itn.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var audioPositions = itn.GetProperty("audio_token_positions").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int audioTokenId = root.GetProperty("audio_token_id").GetInt32();
        int eosTokenId = root.GetProperty("eos_token_id").GetInt32();

        var audioEmbedEntry = root.GetProperty("audio_embeddings");
        var audioEmbedShape = audioEmbedEntry.GetProperty("shape");
        int numAudioTokens = audioEmbedShape[0].GetInt32();
        int hiddenDim = audioEmbedShape[1].GetInt32();
        var audioEmbedFlat = ReadFlat(data, audioEmbedEntry);

        var expectedTokenIds = root.GetProperty("greedy_token_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        string expectedText = root.GetProperty("generated_text").GetString() ?? "";

        Assert.Equal(audioPositions.Length, numAudioTokens);
        foreach (var p in audioPositions) Assert.Equal(audioTokenId, inputIds[p]);

        using var model = OpenTail.Stingray.Core.GgufModel.Open(modelPath!);
        using var llmSource = new OpenTail.Stingray.Audio.FunASR.FunAsrNanoLlmGgufTensorSource(
            model, numLayers: 28, hiddenDim: hiddenDim, numHeads: 16, numKvHeads: 8, headDim: 128, ffDim: 3072, vocabSize: 151936 + numAudioTokens, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);
        llmSource.EnableAudioConditioning(audioEmbedFlat, numAudioTokens);

        var prompt = new int[inputIds.Length];
        int frame = 0;
        for (int i = 0; i < inputIds.Length; i++)
            prompt[i] = inputIds[i] == audioTokenId ? llmSource.AudioTokenIdOffset + frame++ : inputIds[i];

        var qInfo = llmSource.FindTensor("blk.0.attn_q.weight");
        var embedInfoDbg = llmSource.FindTensor("token_embd.weight");
        Console.Error.WriteLine($"[FunAsrNanoDecoderGolden] blk.0.attn_q.weight dims=[{string.Join(",", qInfo!.Value.Dimensions)}]");
        Console.Error.WriteLine($"[FunAsrNanoDecoderGolden] token_embd.weight dims=[{string.Join(",", embedInfoDbg!.Value.Dimensions)}]");

        var hp = OpenTail.Stingray.Core.ModelHyperparams.FromGgufMetadata(llmSource.Metadata);
        using var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwd = new OpenTail.Stingray.Engine.ForwardPass(llmSource, backend, hp);

        var logits = fwd.Prefill(prompt);
        var emitted = new List<int>();
        int position = prompt.Length;
        for (int step = 0; step < expectedTokenIds.Length + 2; step++)
        {
            int nextToken = ArgMax(logits);
            emitted.Add(nextToken);
            if (nextToken == eosTokenId) break;
            logits = fwd.Forward(nextToken, position);
            position++;
        }

        Console.Error.WriteLine($"[FunAsrNanoDecoderGolden] emitted=[{string.Join(",", emitted)}] expected=[{string.Join(",", expectedTokenIds)}]");

        // Diagnostic: plain text-only prompt (no audio splice at all) through the SAME
        // ForwardPass/tensor-source wiring, to isolate whether the bug is audio-specific or a
        // general LLM-wiring issue.
        var textOnlyPrompt = new int[] { 151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 105761, 46670, 61443, 5122, 151645, 198, 151644, 77091, 198 };
        var logits2 = fwd.Prefill(textOnlyPrompt);
        var emitted2 = new List<int>();
        int position2 = textOnlyPrompt.Length;
        for (int step = 0; step < 5; step++)
        {
            int nextToken = ArgMax(logits2);
            emitted2.Add(nextToken);
            if (nextToken == eosTokenId) break;
            logits2 = fwd.Forward(nextToken, position2);
            position2++;
        }
        Console.Error.WriteLine($"[FunAsrNanoDecoderGolden] text-only-diagnostic emitted=[{string.Join(",", emitted2)}]");

        Assert.Equal(expectedTokenIds, emitted);
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static float[] ReadAllF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static float[] ReadFlat(float[] data, System.Text.Json.JsonElement entry)
    {
        int offset = entry.GetProperty("offset_f32").GetInt32();
        int count = entry.GetProperty("count").GetInt32();
        var result = new float[count];
        Array.Copy(data, offset, result, 0, count);
        return result;
    }
}
