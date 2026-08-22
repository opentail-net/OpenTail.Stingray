using System;
using System.Collections.Generic;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real Talker prompt composition, transcribed exactly from the real
/// `examples/qwentts.cpp/src/prompt-builder.h` (`prompt_builder_build`), restricted to the
/// "base" case: `auto` language (no language tag), no named speaker, no voice-design instruct,
/// no in-context-learning (ICL) reference audio. This is the simplest real path (matches this
/// project's other TTS pipelines' scope), NOT a simplification of the math itself -- every row
/// this class produces is the same real sum-of-two-streams composition the full prompt builder
/// uses for this configuration.
///
/// <para>Real per-position composition (real, not guessed): two parallel streams --
/// `text(input_ids) -&gt; text_embedding -&gt; text_proj.fc1 -&gt; SiLU -&gt; text_proj.fc2` (151936-&gt;2048-&gt;1024)
/// and `codec_embd(codec_ids)` (3072-&gt;1024, or `tts_bos`/`tts_eos`/`tts_pad` for the text-stream's
/// special/pad rows) -- summed per row. Real layout for auto-language, no-speaker, no-ICL:</para>
/// <code>
/// row 0..2:  text_proj(role_ids[0:3])                         (no codec contribution)
/// row 3..6:  [tts_pad,tts_pad,tts_pad,tts_bos] + codec_embd([nothink,think_bos,think_eos,codec_pad])
/// row 7..7+N_text-1: text_proj(utterance_ids) + codec_pad
/// row next:  tts_eos + codec_pad
/// row next:  tts_pad + codec_bos
/// </code>
/// <para>Real special ids confirmed via this GGUF's own metadata (not guessed):
/// `qwen3-tts.text.tts_bos_id`=151672, `tts_eos_id`=151673, `tts_pad_id`=151671;
/// `qwen3-tts.codec.nothink_id`=2155, `think_bos_id`=2156, `think_eos_id`=2157, `pad_id`=2148,
/// `bos_id`=2149.</para>
/// </summary>
public static class QwenTtsTalkerPromptBuilder
{
    public const int TextEmbedDim = 2048;
    public const int TalkerHiddenDim = 1024;

    public sealed record SpecialTokenIds(
        int TtsBosId, int TtsEosId, int TtsPadId,
        int CodecNothinkId, int CodecThinkBosId, int CodecThinkEosId, int CodecPadId, int CodecBosId, int CodecEosId);

    /// <summary>Real Talker weights needed for prompt composition, loaded once (the text embedding table alone is ~300M elements -- must not be re-dequantized per call).</summary>
    public sealed class Weights
    {
        public required float[] TextEmbd { get; init; } // [151936,2048] native
        public required float[] Fc1Weight { get; init; } // [2048,2048] native [out,in]
        public required float[] Fc1Bias { get; init; }
        public required float[] Fc2Weight { get; init; } // [1024,2048] native [out,in]
        public required float[] Fc2Bias { get; init; }
        public required float[] CodecEmbd { get; init; } // [3072,1024] native
        public required SpecialTokenIds Specials { get; init; }

        public static Weights Load(GgufModel model) => new()
        {
            TextEmbd = GetTensorF32(model, "talker.text_embd.weight"),
            Fc1Weight = GetTensorF32(model, "talker.text_proj.fc1.weight"),
            Fc1Bias = GetTensorF32(model, "talker.text_proj.fc1.bias"),
            Fc2Weight = GetTensorF32(model, "talker.text_proj.fc2.weight"),
            Fc2Bias = GetTensorF32(model, "talker.text_proj.fc2.bias"),
            CodecEmbd = GetTensorF32(model, "talker.codec_embd.weight"),
            Specials = ReadSpecials(model),
        };
    }

    public static SpecialTokenIds ReadSpecials(GgufModel model) => new(
        TtsBosId: model.GetMetadata("qwen3-tts.text.tts_bos_id", 0),
        TtsEosId: model.GetMetadata("qwen3-tts.text.tts_eos_id", 0),
        TtsPadId: model.GetMetadata("qwen3-tts.text.tts_pad_id", 0),
        CodecNothinkId: model.GetMetadata("qwen3-tts.codec.nothink_id", 0),
        CodecThinkBosId: model.GetMetadata("qwen3-tts.codec.think_bos_id", 0),
        CodecThinkEosId: model.GetMetadata("qwen3-tts.codec.think_eos_id", 0),
        CodecPadId: model.GetMetadata("qwen3-tts.codec.pad_id", 0),
        CodecBosId: model.GetMetadata("qwen3-tts.codec.bos_id", 0),
        CodecEosId: model.GetMetadata("qwen3-tts.codec.eos_id", 0));

    /// <summary>Real text-side projection: text_embedding lookup -&gt; fc1 (2048-&gt;2048) -&gt; SiLU -&gt; fc2 (2048-&gt;1024).</summary>
    public static float[] ProjectTextIds(Weights w, IReadOnlyList<int> ids)
    {
        int n = ids.Count;
        var output = new float[n * TalkerHiddenDim];
        for (int i = 0; i < n; i++)
        {
            var embed = new float[TextEmbedDim];
            Array.Copy(w.TextEmbd, (long)ids[i] * TextEmbedDim, embed, 0, TextEmbedDim);

            var h = new float[TextEmbedDim];
            for (int o = 0; o < TextEmbedDim; o++)
            {
                float sum = w.Fc1Bias[o];
                int wBase = o * TextEmbedDim;
                for (int k = 0; k < TextEmbedDim; k++) sum += w.Fc1Weight[wBase + k] * embed[k];
                h[o] = sum / (1f + MathF.Exp(-sum)); // SiLU
            }

            for (int o = 0; o < TalkerHiddenDim; o++)
            {
                float sum = w.Fc2Bias[o];
                int wBase = o * TextEmbedDim;
                for (int k = 0; k < TextEmbedDim; k++) sum += w.Fc2Weight[wBase + k] * h[k];
                output[i * TalkerHiddenDim + o] = sum;
            }
        }
        return output;
    }

    public static float[] CodecEmbedRow(Weights w, int codecId)
    {
        var row = new float[TalkerHiddenDim];
        Array.Copy(w.CodecEmbd, (long)codecId * TalkerHiddenDim, row, 0, TalkerHiddenDim);
        return row;
    }

    /// <summary>
    /// Builds the full real prompt embedding matrix [T,1024] for the base/auto-language/
    /// no-speaker/no-ICL case.
    /// </summary>
    public static (float[] Embedding, int NumRows) BuildBasePrompt(Weights w, GgufTokenizer tokenizer, string utteranceText)
    {
        string fullText = "<|im_start|>assistant\n" + utteranceText + "<|im_end|>\n<|im_start|>assistant\n";
        var ids = tokenizer.Encode(fullText);
        int n = ids.Count;
        int nText = n - 3 - 5;
        if (nText <= 0)
            throw new InvalidOperationException($"QwenTTS prompt: tokenized prompt too short (N={n}) for any utterance text.");

        var specials = w.Specials;
        var codecLeft = new[] { specials.CodecNothinkId, specials.CodecThinkBosId, specials.CodecThinkEosId, specials.CodecPadId };

        int tCtx = 3 + codecLeft.Length + nText + 1 + 1;
        var embed = new float[tCtx * TalkerHiddenDim];
        int row = 0;

        // Role: text_proj(ids[0:3]), no codec contribution.
        var roleEmb = ProjectTextIds(w, [ids[0], ids[1], ids[2]]);
        Array.Copy(roleEmb, 0, embed, 0, roleEmb.Length);
        row += 3;

        // Codec prefix: tts_pad x(n-1) + tts_bos, summed with codec_embd(codecLeft[i]).
        for (int i = 0; i < codecLeft.Length; i++)
        {
            int rowOff = (row + i) * TalkerHiddenDim;
            int specialId = i == codecLeft.Length - 1 ? specials.TtsBosId : specials.TtsPadId;
            var textSpecial = ProjectTextIds(w, [specialId]);
            Array.Copy(textSpecial, 0, embed, rowOff, TalkerHiddenDim);

            var codecVec = CodecEmbedRow(w, codecLeft[i]);
            for (int d = 0; d < TalkerHiddenDim; d++) embed[rowOff + d] += codecVec[d];
        }
        row += codecLeft.Length;

        // Trailing utterance text + codec_pad.
        var textIds = new List<int>(nText);
        for (int i = 0; i < nText; i++) textIds.Add(ids[3 + i]);
        var textEmb = ProjectTextIds(w, textIds);
        var codecPadVec = CodecEmbedRow(w, specials.CodecPadId);
        for (int i = 0; i < nText; i++)
        {
            int rowOff = (row + i) * TalkerHiddenDim;
            Array.Copy(textEmb, i * TalkerHiddenDim, embed, rowOff, TalkerHiddenDim);
            for (int d = 0; d < TalkerHiddenDim; d++) embed[rowOff + d] += codecPadVec[d];
        }
        row += nText;

        // tts_eos + codec_pad
        {
            var e = ProjectTextIds(w, [specials.TtsEosId]);
            int rowOff = row * TalkerHiddenDim;
            Array.Copy(e, 0, embed, rowOff, TalkerHiddenDim);
            for (int d = 0; d < TalkerHiddenDim; d++) embed[rowOff + d] += codecPadVec[d];
            row++;
        }

        // tts_pad + codec_bos
        {
            var e = ProjectTextIds(w, [specials.TtsPadId]);
            int rowOff = row * TalkerHiddenDim;
            Array.Copy(e, 0, embed, rowOff, TalkerHiddenDim);
            var codecBosVec = CodecEmbedRow(w, specials.CodecBosId);
            for (int d = 0; d < TalkerHiddenDim; d++) embed[rowOff + d] += codecBosVec[d];
            row++;
        }

        if (row != tCtx)
            throw new InvalidOperationException($"QwenTTS prompt builder internal error: row={row} expected T_ctx={tCtx}");

        return (embed, tCtx);
    }

    private static float[] GetTensorF32(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new InvalidDataException($"QwenTTS talker GGUF missing required tensor '{name}'.");
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }
}
