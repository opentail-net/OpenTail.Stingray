using System.Text.Json;

namespace OpenTail.Stingray.Audio.MmsTts;

/// <summary>
/// Real config.json fields for a HuggingFace `transformers.VitsModel` MMS-TTS checkpoint (e.g.
/// `facebook/mms-tts-eng`). Only the fields this port's inference path actually needs are parsed.
/// </summary>
public sealed class MmsTtsConfig
{
    public int HiddenSize { get; init; } = 192;
    public int NumAttentionHeads { get; init; } = 2;
    public int WindowSize { get; init; } = 4;
    public int NumHiddenLayers { get; init; } = 6;
    public int FfnKernelSize { get; init; } = 3;
    public int SamplingRate { get; init; } = 16000;
    public float NoiseScale { get; init; } = 0.667f;
    public float NoiseScaleDuration { get; init; } = 0.8f;
    public float SpeakingRate { get; init; } = 1.0f;
    public int[] UpsampleRates { get; init; } = [8, 8, 2, 2];
    public int[] UpsampleKernelSizes { get; init; } = [16, 16, 4, 4];
    public int UpsampleInitialChannel { get; init; } = 512;
    public int[] ResblockKernelSizes { get; init; } = [3, 7, 11];

    public static MmsTtsConfig Load(string configJsonPath)
    {
        using var stream = File.OpenRead(configJsonPath);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        static int[] ReadIntArray(JsonElement root, string name, int[] fallback)
        {
            if (!root.TryGetProperty(name, out var el) || el.ValueKind != JsonValueKind.Array) return fallback;
            var arr = new int[el.GetArrayLength()];
            int i = 0;
            foreach (var v in el.EnumerateArray()) arr[i++] = v.GetInt32();
            return arr;
        }

        return new MmsTtsConfig
        {
            HiddenSize = root.TryGetProperty("hidden_size", out var hs) ? hs.GetInt32() : 192,
            NumAttentionHeads = root.TryGetProperty("num_attention_heads", out var nh) ? nh.GetInt32() : 2,
            WindowSize = root.TryGetProperty("window_size", out var ws) ? ws.GetInt32() : 4,
            NumHiddenLayers = root.TryGetProperty("num_hidden_layers", out var nl) ? nl.GetInt32() : 6,
            FfnKernelSize = root.TryGetProperty("ffn_kernel_size", out var fk) ? fk.GetInt32() : 3,
            SamplingRate = root.TryGetProperty("sampling_rate", out var sr) ? sr.GetInt32() : 16000,
            NoiseScale = root.TryGetProperty("noise_scale", out var ns) ? ns.GetSingle() : 0.667f,
            NoiseScaleDuration = root.TryGetProperty("noise_scale_duration", out var nsd) ? nsd.GetSingle() : 0.8f,
            SpeakingRate = root.TryGetProperty("speaking_rate", out var spr) ? spr.GetSingle() : 1.0f,
            UpsampleRates = ReadIntArray(root, "upsample_rates", [8, 8, 2, 2]),
            UpsampleKernelSizes = ReadIntArray(root, "upsample_kernel_sizes", [16, 16, 4, 4]),
            UpsampleInitialChannel = root.TryGetProperty("upsample_initial_channel", out var uic) ? uic.GetInt32() : 512,
            ResblockKernelSizes = ReadIntArray(root, "resblock_kernel_sizes", [3, 7, 11]),
        };
    }
}
