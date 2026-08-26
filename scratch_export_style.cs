using System;
using System.IO;
using System.Collections.Generic;
using OpenTail.Stingray.Audio.Chatterbox;

class Program {
    static void Main() {
        var w = new ChatterboxWeights("models/chatterbox-turbo-t3-q4_k.gguf");
        string outDir = "examples/Chatterbox-turbo-cpp/style";

        // 1. cond_emb.bin: spkr_proj(1) + speech_emb(prompt_tokens) (375) => 376 * 1024 floats
        var condList = new List<float>();
        float[] speakerEmb = w.SpeakerEmbedding ?? new float[w.SpeakerEmbedSize];
        
        // SpkrEnc projection
        var spkrProj = new float[w.HiddenDim];
        for (int o = 0; o < w.HiddenDim; o++) {
            float sum = w.SpkrEncBias[o];
            int wBase = o * w.SpeakerEmbedSize;
            for (int i = 0; i < w.SpeakerEmbedSize; i++) {
                sum += speakerEmb[i] * w.SpkrEncWeight[wBase + i];
            }
            spkrProj[o] = sum;
        }
        condList.AddRange(spkrProj);

        if (w.SpeechPromptTokens is { } promptTokens) {
            foreach (int tok in promptTokens) {
                int rowBase = tok * w.HiddenDim;
                for (int d = 0; d < w.HiddenDim; d++) {
                    condList.Add(w.SpeechEmbWeight[rowBase + d]);
                }
            }
        }

        byte[] condBytes = new byte[condList.Count * 4];
        Buffer.BlockCopy(condList.ToArray(), 0, condBytes, 0, condBytes.Length);
        File.WriteAllBytes(Path.Combine(outDir, "cond_emb.bin"), condBytes);
        Console.WriteLine($"Exported cond_emb.bin: {condList.Count} floats ({condList.Count / 1024} tokens)");

        // 2. prompt_token.bin: int64 array
        if (w.GenPromptToken is { } genTokens) {
            byte[] tokBytes = new byte[genTokens.Length * 8];
            long[] i64 = new long[genTokens.Length];
            for (int i = 0; i < genTokens.Length; i++) i64[i] = genTokens[i];
            Buffer.BlockCopy(i64, 0, tokBytes, 0, tokBytes.Length);
            File.WriteAllBytes(Path.Combine(outDir, "prompt_token.bin"), tokBytes);
            Console.WriteLine($"Exported prompt_token.bin: {genTokens.Length} int64");
        }

        // 3. speaker_embeddings.bin: float array [192]
        if (w.GenEmbedding is { } genEmb) {
            byte[] embBytes = new byte[genEmb.Length * 4];
            Buffer.BlockCopy(genEmb, 0, embBytes, 0, embBytes.Length);
            File.WriteAllBytes(Path.Combine(outDir, "speaker_embeddings.bin"), embBytes);
            Console.WriteLine($"Exported speaker_embeddings.bin: {genEmb.Length} floats");
        }

        // 4. speaker_features.bin: float array [40000] = [500, 80] (row-major: 500 frames, 80 channels)
        if (w.GenPromptFeat is { } genFeat) {
            // GenPromptFeat in GGUF is channel-first [80, 500].
            // Chatterbox C++ expects [1, 500, 80] (time-first row-major):
            int mel = 80;
            int frames = genFeat.Length / mel;
            float[] timeFirst = new float[genFeat.Length];
            for (int ti = 0; ti < frames; ti++) {
                for (int c = 0; c < mel; c++) {
                    timeFirst[ti * mel + c] = genFeat[c * frames + ti];
                }
            }
            byte[] featBytes = new byte[timeFirst.Length * 4];
            Buffer.BlockCopy(timeFirst, 0, featBytes, 0, featBytes.Length);
            File.WriteAllBytes(Path.Combine(outDir, "speaker_features.bin"), featBytes);
            Console.WriteLine($"Exported speaker_features.bin: {timeFirst.Length} floats ({frames}x{mel})");
        }
    }
}
