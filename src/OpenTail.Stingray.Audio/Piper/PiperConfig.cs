using System.Text.Json.Serialization;

namespace OpenTail.Stingray.Audio.Piper;

[JsonSourceGenerationOptions(WriteIndented = false, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(PiperConfig))]
[JsonSerializable(typeof(PiperAudioConfig))]
[JsonSerializable(typeof(PiperEspeakConfig))]
[JsonSerializable(typeof(PiperInferenceConfig))]
[JsonSerializable(typeof(Dictionary<string, List<int>>))]
[JsonSerializable(typeof(Dictionary<string, int>))]
internal sealed partial class PiperJsonContext : JsonSerializerContext
{
}

/// <summary>
/// Piper VITS model configuration and voice metadata (.onnx.json).
/// </summary>
public sealed record PiperConfig
{
    [JsonPropertyName("audio")]
    public PiperAudioConfig Audio { get; init; } = new();

    [JsonPropertyName("espeak")]
    public PiperEspeakConfig? Espeak { get; init; }

    [JsonPropertyName("inference")]
    public PiperInferenceConfig Inference { get; init; } = new();

    [JsonPropertyName("phoneme_id_map")]
    public Dictionary<string, List<int>> PhonemeIdMap { get; init; } = new(StringComparer.Ordinal);

    [JsonPropertyName("phoneme_type")]
    public string PhonemeType { get; init; } = "espeak";

    [JsonPropertyName("num_speakers")]
    public int NumSpeakers { get; init; } = 1;

    [JsonPropertyName("speaker_id_map")]
    public Dictionary<string, int>? SpeakerIdMap { get; init; }

    public static PiperConfig FromJson(string json)
    {
        return JsonSerializer.Deserialize(json, PiperJsonContext.Default.PiperConfig) ?? new PiperConfig();
    }

    public static PiperConfig Load(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"Piper config file not found: {path}", path);
        string json = File.ReadAllText(path);
        return FromJson(json);
    }
}

public sealed record PiperAudioConfig
{
    [JsonPropertyName("sample_rate")]
    public int SampleRate { get; init; } = 22050;

    [JsonPropertyName("quality")]
    public string Quality { get; init; } = "medium";
}

public sealed record PiperEspeakConfig
{
    [JsonPropertyName("voice")]
    public string Voice { get; init; } = "en-us";
}

public sealed record PiperInferenceConfig
{
    [JsonPropertyName("noise_scale")]
    public float NoiseScale { get; init; } = 0.667f;

    [JsonPropertyName("length_scale")]
    public float LengthScale { get; init; } = 1.0f;

    [JsonPropertyName("noise_w")]
    public float NoiseW { get; init; } = 0.8f;
}
