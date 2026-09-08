using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: real per-layer bisection of VibeVoice TTS's LLM backbone prefill,
/// matching <see cref="VoxCpm2PerLayerBisectDebugTest"/>'s exact successful technique (that
/// bisection found VoxCPM2's divergence was a single concrete RoPE-convention bug, not diffuse
/// floating-point noise -- worth checking whether the same pattern applies here). Uses the
/// existing <see cref="IForwardPass.EnableHiddenTaps"/>/<see cref="IForwardPass.HiddenTapsAt"/>
/// API to capture every layer's hidden state at the last prefill position, printed at the same
/// real reference sample indices the vendored reference's new per-layer
/// `vibevoice_tts.prefill.layer{i}.hidden` trace emits (`examples/audio.cpp/src/models/vibevoice/
/// decoder.cpp`, instrumented this session).</summary>
public sealed class VibeVoiceTtsPerLayerBisectDebugTest : HeavyTestBase
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

    private const int NumLayers = 28, HiddenDim = 1536, NumHeads = 12, NumKvHeads = 2, HeadDim = 128;
    private const int FfDim = 8960, VocabSize = 151936;
    private const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;

    [Fact]
    public void PrintPerLayerHiddenAtReferenceSampleIndices()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-1.5b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        string dir = Path.Combine(Path.GetTempPath(), "stingray-vibevoice-tts-tokenizer");
        var result = HuggingFaceTokenizerSource.Load(dir);
        if (result.Source is null)
        {
            // Extract tokenizer files from the packed GGUF (same convention as other VibeVoice tests).
            var namesObj = (object[])model.Metadata["audiocpp.embedded_files.names"];
            var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
            var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
            var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
            Directory.CreateDirectory(dir);
            string[] wanted = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"];
            for (int i = 0; i < namesObj.Length; i++)
            {
                string name = (string)namesObj[i];
                if (Array.IndexOf(wanted, name) < 0) continue;
                long start = Convert.ToInt64(offsets[i]);
                long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
                File.WriteAllBytes(Path.Combine(dir, name), bytes[(int)start..(int)end]);
            }
            result = HuggingFaceTokenizerSource.Load(dir);
        }
        var tokenizer = GgufTokenizer.FromSource(result.Source!);

        const int SpeechStartId = 151652;
        string script = "Speaker 1: Hello there, this is a real end to end test of speech synthesis.";
        var promptTokenIds = VibeVoiceTtsPromptBuilder.BuildPrompt(text => [.. tokenizer.Encode(text)], script, SpeechStartId);

        using var llm = new VibeVoiceLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        Assert.True(fwd.SupportsHiddenTaps, "CPU ForwardPass must support hidden taps for this bisection.");
        fwd.EnableHiddenTaps([.. Enumerable.Range(0, NumLayers)]);

        fwd.Prefill(promptTokenIds, startPos: 0);

        int lastPos = promptTokenIds.Length - 1;
        var allLayers = fwd.HiddenTapsAt(lastPos);

        // Real reference sample indices (same 40 points the reference's own trace prints).
        int[] indices = [0, 39, 78, 117, 157, 196, 235, 275, 314, 353, 393, 432, 471, 511, 512, 558, 604, 651, 697, 744, 790, 837, 883, 930, 976, 1023, 1024, 1063, 1102, 1141, 1181, 1220, 1259, 1299, 1338, 1377, 1417, 1456, 1495, 1535];

        var lines = new List<string>();
        for (int layer = 0; layer < NumLayers; layer++)
        {
            var row = allLayers.Slice(layer * HiddenDim, HiddenDim);
            var sb = new System.Text.StringBuilder($"[Ours] layer{layer} hidden samples=[");
            foreach (int idx in indices) sb.Append($"{idx}:{row[idx]:G6},");
            sb.Append(']');
            lines.Add(sb.ToString());
            Console.WriteLine(sb.ToString());
        }
        Directory.CreateDirectory("scratch");
        File.WriteAllLines("scratch/vibevoice_tts_ours_trace.log", lines);
    }
}
