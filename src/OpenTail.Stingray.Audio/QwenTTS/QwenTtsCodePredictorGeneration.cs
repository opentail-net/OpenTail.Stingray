using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real Code Predictor autoregressive depth-expansion, transcribed from
/// `examples/qwentts.cpp/src/code-predictor-forward.h` (`code_predictor_pass_append`/
/// `code_predictor_frame_graph_build`): given one frame's Talker semantic code `c0` and the
/// Talker's own last-position hidden state (post final norm, captured via the real
/// <see cref="ForwardPass.LastHidden"/> this session added specifically for this), expands it
/// into the remaining 15 acoustic codes via a real 5-layer transformer.
///
/// <para>Real, exact sequence: pass 0 (T=2 prefill) reads `[talker_hidden, embed(c0)]` (`embed`
/// via the TALKER's own `codec_embd` table, NOT the Code Predictor's -- confirmed real, see
/// `code-predictor-forward.h`'s own comment: "prefill path concats each slot's resident talker
/// hidden ahead of embed(c0)... through `lm_head[0]`") and predicts `c1` via
/// `code_pred.lm_head.0`. Passes `g=1..14` each read `codes[g]` via
/// `code_pred.codec_embd.{g-1}` (real: table index is `g-1`, not `g`) and predict `codes[g+1]`
/// via `code_pred.lm_head.{g}`. Real per-frame cache is fresh (`fwd` constructed once per
/// frame, real per-source doc: "the predictor cache is local to a single frame").</para>
/// </summary>
public static class QwenTtsCodePredictorGeneration
{
    public const int NumAcousticCodebooks = 15; // c1..c15
    public const int AcousticVocabSize = 2048;
    public const int HiddenDim = 1024;

    public sealed class Weights
    {
        public required float[] TalkerCodecEmbd { get; init; } // [3072,1024] native, real talker.codec_embd.weight
        public required float[][] CodecEmbd { get; init; } // 15x [2048,1024] native, code_pred.codec_embd.{0..14}.weight
        public required float[][] LmHead { get; init; } // 15x [2048,1024] native, code_pred.lm_head.{0..14}.weight

        public static Weights Load(GgufModel model)
        {
            var codecEmbd = new float[NumAcousticCodebooks][];
            var lmHead = new float[NumAcousticCodebooks][];
            for (int i = 0; i < NumAcousticCodebooks; i++)
            {
                codecEmbd[i] = GetTensorF32(model, $"code_pred.codec_embd.{i}.weight");
                lmHead[i] = GetTensorF32(model, $"code_pred.lm_head.{i}.weight");
            }
            return new Weights
            {
                TalkerCodecEmbd = GetTensorF32(model, "talker.codec_embd.weight"),
                CodecEmbd = codecEmbd,
                LmHead = lmHead,
            };
        }

        private static float[] GetTensorF32(GgufModel model, string name)
        {
            var info = model.FindTensor(name) ?? throw new InvalidDataException($"QwenTTS code_pred GGUF missing required tensor '{name}'.");
            var bytes = model.GetTensorData(info);
            var dst = new float[info.ElementCount];
            Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
            return dst;
        }
    }

    /// <summary>
    /// Generates the 15 real acoustic codes for one frame. Returns `codes[0..14]` = `c1..c15`.
    /// `talkerLastHidden` must be the Talker's real `ForwardPass.LastHidden` captured
    /// immediately after the `Forward` call that produced `c0` (before any subsequent Talker
    /// step overwrites the shared buffer).
    /// </summary>
    public static int[] GenerateAcousticCodes(GgufModel rawModel, Weights weights, int numLayers, int c0, ReadOnlySpan<float> talkerLastHidden, Random rng)
    {
        using var source = new QwenTtsCodePredictorTensorSource(rawModel, numLayers);

        var promptRows = new float[2 * HiddenDim];
        talkerLastHidden.CopyTo(promptRows.AsSpan(0, HiddenDim));
        Array.Copy(weights.TalkerCodecEmbd, (long)c0 * HiddenDim, promptRows, HiddenDim, HiddenDim);
        source.SetPromptEmbedding(promptRows, 2);
        source.SetOutputHead(weights.LmHead[0], AcousticVocabSize);

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata, source);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var logits = fwd.Prefill([0, 1]).ToArray();

        var codes = new int[NumAcousticCodebooks];
        codes[0] = SampleTopK(logits, temperature: 0.9f, topK: 50, rng); // c1, from lm_head.0

        int pos = 2;
        for (int g = 1; g < NumAcousticCodebooks; g++)
        {
            int prevCode = codes[g - 1]; // codes[g] in the real 1-indexed doc numbering
            var inputEmb = new float[HiddenDim];
            Array.Copy(weights.CodecEmbd[g - 1], (long)prevCode * HiddenDim, inputEmb, 0, HiddenDim);

            source.SetPromptEmbedding(inputEmb, 1);
            source.SetOutputHead(weights.LmHead[g], AcousticVocabSize);

            logits = fwd.Forward(0, pos).ToArray();
            pos++;

            codes[g] = SampleTopK(logits, temperature: 0.9f, topK: 50, rng); // c_{g+1}
        }

        return codes;
    }

    /// <summary>
    /// Real Qwen3-TTS subtalker sampling: <c>do_sample=True, top_k=50, top_p=1.0,
    /// temperature=0.9</c>, no repetition penalty (confirmed from the local reference source,
    /// `examples/qwen-tts-py/qwen_tts/core/models/modeling_qwen3_tts.py` -- the subtalker/code-
    /// predictor generation kwargs never pass a `subtalker_repetition_penalty`, unlike the main
    /// talker's `repetition_penalty=1.05`). Was plain greedy <c>Argmax</c> before this fix -- same
    /// class of bug as Parler-TTS's "drill noise" collapse, see
    /// docs/audio-review-progress.md. <c>top_p=1.0</c> makes nucleus filtering a no-op, so it is
    /// not implemented here.
    /// </summary>
    private static int SampleTopK(float[] logits, float temperature, int topK, Random rng)
    {
        int k = Math.Min(topK, logits.Length);
        Span<int> topIdx = stackalloc int[k];
        Span<float> topVal = stackalloc float[k];
        int filled = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i] / temperature;
            if (filled < k)
            {
                int pos = filled++;
                while (pos > 0 && topVal[pos - 1] > v) { topVal[pos] = topVal[pos - 1]; topIdx[pos] = topIdx[pos - 1]; pos--; }
                topVal[pos] = v; topIdx[pos] = i;
            }
            else if (v > topVal[0])
            {
                int pos = 0;
                while (pos < k - 1 && topVal[pos + 1] < v) { topVal[pos] = topVal[pos + 1]; topIdx[pos] = topIdx[pos + 1]; pos++; }
                topVal[pos] = v; topIdx[pos] = i;
            }
        }

        float max = topVal[k - 1];
        double sum = 0.0;
        for (int i = 0; i < k; i++) sum += Math.Exp(topVal[i] - max);

        double r = rng.NextDouble() * sum;
        double cumulative = 0.0;
        for (int i = 0; i < k; i++)
        {
            cumulative += Math.Exp(topVal[i] - max);
            if (r < cumulative) return topIdx[i];
        }
        return topIdx[k - 1];
    }
}
