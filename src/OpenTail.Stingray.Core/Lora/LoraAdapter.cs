
namespace OpenTail.Stingray.Core.Lora;

/// <summary>
/// Container for a complete fine-tuned LoRA adapter comprising multiple layer weights.
/// </summary>
public sealed class LoraAdapter : IDisposable
{
    private readonly Dictionary<(int layer, string module), LoraLayer> _layers = new();
    private bool _disposed;

    public string Id { get; }
    public string Path { get; }
    public IReadOnlyDictionary<(int layer, string module), LoraLayer> Layers => _layers;
    public int LayerCount => _layers.Count;

    public LoraAdapter(string id, string path, IEnumerable<LoraLayer> layers)
    {
        Id = id;
        Path = path;
        foreach (var l in layers)
        {
            _layers[(l.LayerIndex, l.TargetName)] = l;
        }
    }

    public bool TryGetLayer(int layer, string module, out LoraLayer? loraLayer)
    {
        return _layers.TryGetValue((layer, module), out loraLayer);
    }

    /// <summary>
    /// Applies the low-rank delta for the specified layer and module if present in this adapter.
    /// </summary>
    public void ApplyDelta(int layer, string module, ReadOnlySpan<float> input, Span<float> output)
    {
        if (_layers.TryGetValue((layer, module), out var loraLayer))
        {
            loraLayer.ApplyDelta(input, output);
        }
    }

    /// <summary>
    /// Loads a LoRA adapter from a .safetensors file.
    /// </summary>
    public static LoraAdapter Load(string path, string? id = null)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"LoRA adapter file not found: {path}");

        id ??= System.IO.Path.GetFileNameWithoutExtension(path);
        var layers = LoadFromSafetensors(path);
        return new LoraAdapter(id, path, layers);
    }

    private static List<LoraLayer> LoadFromSafetensors(string path)
    {
        var result = new List<LoraLayer>();
        using var loader = SafetensorsLoader.Open(path);

        var aWeights = new Dictionary<string, (float[] data, int[] shape)>(StringComparer.Ordinal);
        var bWeights = new Dictionary<string, (float[] data, int[] shape)>(StringComparer.Ordinal);
        var alphas = new Dictionary<string, float>(StringComparer.Ordinal);

        foreach (var key in loader.TensorNames)
        {
            if (key.EndsWith(".alpha", StringComparison.Ordinal))
            {
                string baseKey = key[..^".alpha".Length];
                var alphaArr = loader.ReadF32(key);
                if (alphaArr.Length > 0) alphas[baseKey] = alphaArr[0];
            }
            else if (key.EndsWith(".lora_down.weight", StringComparison.Ordinal) || key.EndsWith(".lora_A.weight", StringComparison.Ordinal))
            {
                string baseKey = key.EndsWith(".lora_down.weight", StringComparison.Ordinal)
                    ? key[..^".lora_down.weight".Length]
                    : key[..^".lora_A.weight".Length];

                var data = loader.ReadF32(key);
                var shape = loader.GetShape(key);
                aWeights[baseKey] = (data, shape);
            }
            else if (key.EndsWith(".lora_up.weight", StringComparison.Ordinal) || key.EndsWith(".lora_B.weight", StringComparison.Ordinal))
            {
                string baseKey = key.EndsWith(".lora_up.weight", StringComparison.Ordinal)
                    ? key[..^".lora_up.weight".Length]
                    : key[..^".lora_B.weight".Length];

                var data = loader.ReadF32(key);
                var shape = loader.GetShape(key);
                bWeights[baseKey] = (data, shape);
            }
        }

        foreach (var (baseKey, (aData, aShape)) in aWeights)
        {
            if (!bWeights.TryGetValue(baseKey, out var bVal))
                continue;

            var (bData, bShape) = bVal;

            int rank = aShape.Length >= 2 ? aShape[0] : 16;
            int inDim = aShape.Length >= 2 ? aShape[1] : aData.Length / rank;
            int outDim = bShape.Length >= 2 ? bShape[0] : bData.Length / rank;

            float alpha = alphas.TryGetValue(baseKey, out float explicitAlpha) ? explicitAlpha : (float)rank;

            var (layerIdx, module) = ParseLayerAndModule(baseKey);
            if (layerIdx >= 0 && !string.IsNullOrEmpty(module))
            {
                result.Add(new LoraLayer(module, layerIdx, aData, bData, inDim, outDim, rank, alpha));
            }
        }

        return result;
    }

    private static readonly Regex LayerRegex = new(@"\b(?:layers|blk|blocks)\.(\d+)\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static (int layer, string module) ParseLayerAndModule(string key)
    {
        var match = LayerRegex.Match(key);
        int layer = match.Success ? int.Parse(match.Groups[1].Value) : -1;

        string normalized = key.ToLowerInvariant();
        string module = "";

        if (normalized.Contains("q_proj") || normalized.Contains("attn_q")) module = "q_proj";
        else if (normalized.Contains("k_proj") || normalized.Contains("attn_k")) module = "k_proj";
        else if (normalized.Contains("v_proj") || normalized.Contains("attn_v")) module = "v_proj";
        else if (normalized.Contains("o_proj") || normalized.Contains("attn_output") || normalized.Contains("attn_out")) module = "o_proj";
        else if (normalized.Contains("gate_proj") || normalized.Contains("ffn_gate")) module = "gate_proj";
        else if (normalized.Contains("up_proj") || normalized.Contains("ffn_up")) module = "up_proj";
        else if (normalized.Contains("down_proj") || normalized.Contains("ffn_down")) module = "down_proj";

        return (layer, module);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _layers.Clear();
        }
    }
}
