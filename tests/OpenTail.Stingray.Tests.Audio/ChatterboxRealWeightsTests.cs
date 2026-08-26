using System;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Chatterbox;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.Fast;

public sealed class ChatterboxRealWeightsTests : HeavyTestBase
{
    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ChatterboxPipeline_LoadRealGgufModels_Synthesizes24kHzAudio()
    {
        string? t3Path = FindModelPath("chatterbox-turbo-t3-q4_k.gguf");
        string? s3GenPath = FindModelPath("chatterbox-turbo-s3gen-q4_k.gguf");

        if (t3Path is null) return;

        using var pipeline = ChatterboxPipeline.Load(t3Path, s3GenPath);
        Assert.NotNull(pipeline);
        Assert.Equal("Chatterbox-Turbo", pipeline.Architecture);
        Assert.Equal(24000, pipeline.DefaultSampleRate);

        var request = new AudioGenerationRequest
        {
            Text = "Chatterbox Turbo zero-shot expressive voice generation running natively in OpenTail Stingray.",
            Voice = "nova",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);
        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.NotEmpty(result.Samples);
    }

    [Fact]
    public void Chatterbox_ExportStyleBinFiles()
    {
        string? t3Path = FindModelPath("chatterbox-turbo-t3-q4_k.gguf");
        if (t3Path is null) return;

        using var w = new ChatterboxWeights(t3Path);
        string outDir = @"C:\Git-Public\OpenTail.Stingray\examples\Chatterbox-turbo-cpp\style";
        Directory.CreateDirectory(outDir);

        var condList = new System.Collections.Generic.List<float>();
        float[] speakerEmb = w.SpeakerEmbedding ?? new float[w.SpeakerEmbedSize];
        
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

        if (w.GenPromptToken is { } genTokens) {
            byte[] tokBytes = new byte[genTokens.Length * 8];
            long[] i64 = new long[genTokens.Length];
            for (int i = 0; i < genTokens.Length; i++) i64[i] = genTokens[i];
            Buffer.BlockCopy(i64, 0, tokBytes, 0, tokBytes.Length);
            File.WriteAllBytes(Path.Combine(outDir, "prompt_token.bin"), tokBytes);
        }

        if (w.GenEmbedding is { } genEmb) {
            byte[] embBytes = new byte[genEmb.Length * 4];
            Buffer.BlockCopy(genEmb, 0, embBytes, 0, embBytes.Length);
            File.WriteAllBytes(Path.Combine(outDir, "speaker_embeddings.bin"), embBytes);
        }

        if (w.GenPromptFeat is { } genFeat) {
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
        }
    }

    [Fact]
    public void Chatterbox_T3_GgufRealModelFile_LoadsAndInspectsTensors()
    {
        string? modelPath = FindModelPath("chatterbox-turbo-t3-q4_k.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Chatterbox T3 GGUF must have tensors");
        Assert.True(model.Metadata.Count > 0, "Chatterbox T3 GGUF must have metadata");
    }

    [Fact]
    public void Chatterbox_S3Gen_GgufRealModelFile_LoadsAndInspectsTensors()
    {
        string? modelPath = FindModelPath("chatterbox-turbo-s3gen-q4_k.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Chatterbox S3Gen GGUF must have tensors");
    }
}
