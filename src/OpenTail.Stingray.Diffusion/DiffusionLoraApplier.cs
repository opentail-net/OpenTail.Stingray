using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Native LoRA (Low-Rank Adaptation) adapter loader and inplace delta applier.
/// Supports standard Low-Rank safetensors checkpoints across Stable Diffusion 1.5, SDXL, SD3, and FLUX.
/// Reference: stable-diffusion.cpp:src/model_manager.cpp:apply_lora
/// </summary>
public static class DiffusionLoraApplier
{
    public sealed class LoraLayer
    {
        public string TargetName { get; }
        public float[] DownWeight { get; }
        public float[] UpWeight { get; }
        public int InDim { get; }
        public int OutDim { get; }
        public int Rank { get; }
        public float Alpha { get; }

        public LoraLayer(string targetName, float[] downWeight, float[] upWeight, int inDim, int outDim, int rank, float alpha)
        {
            TargetName = targetName;
            DownWeight = downWeight;
            UpWeight = upWeight;
            InDim = inDim;
            OutDim = outDim;
            Rank = rank;
            Alpha = alpha;
        }

        public float[] ComputeDelta(float multiplier = 1.0f)
        {
            float scale = multiplier * (Alpha / Rank);
            var delta = new float[OutDim * InDim];

            // delta[o * InDim + i] = scale * sum_r(Up[o * Rank + r] * Down[r * InDim + i])
            for (int o = 0; o < OutDim; o++)
            {
                int outOffset = o * InDim;
                int upOffset = o * Rank;
                for (int r = 0; r < Rank; r++)
                {
                    float u = UpWeight[upOffset + r] * scale;
                    if (u == 0f) continue;
                    int downOffset = r * InDim;
                    for (int i = 0; i < InDim; i++)
                    {
                        delta[outOffset + i] += u * DownWeight[downOffset + i];
                    }
                }
            }
            return delta;
        }
    }

    /// <summary>
    /// Loads LoRA layers from a .safetensors file.
    /// </summary>
    public static List<LoraLayer> Load(string loraPath)
    {
        var result = new List<LoraLayer>();
        using var loader = SafetensorsLoader.Open(loraPath);

        var tensorNames = new HashSet<string>(loader.TensorNames, StringComparer.Ordinal);

        foreach (var key in loader.TensorNames)
        {
            if (key.EndsWith(".lora_down.weight", StringComparison.Ordinal) || key.EndsWith(".lora_A.weight", StringComparison.Ordinal))
            {
                string baseKey = key.EndsWith(".lora_down.weight", StringComparison.Ordinal)
                    ? key[..^".lora_down.weight".Length]
                    : key[..^".lora_A.weight".Length];

                string upKey = key.EndsWith(".lora_down.weight", StringComparison.Ordinal)
                    ? $"{baseKey}.lora_up.weight"
                    : $"{baseKey}.lora_B.weight";

                if (!tensorNames.Contains(upKey)) continue;

                var downShape = loader.GetShape(key);
                var upShape = loader.GetShape(upKey);
                var downData = loader.ReadF32(key);
                var upData = loader.ReadF32(upKey);

                int rank = downShape[0];
                int inDim = downShape[1];
                int outDim = upShape[0];

                float alpha = (float)rank;
                string alphaKey = $"{baseKey}.alpha";
                if (tensorNames.Contains(alphaKey))
                {
                    var aData = loader.ReadF32(alphaKey);
                    if (aData.Length > 0) alpha = aData[0];
                }

                string targetName = NormalizeTargetName(baseKey);
                result.Add(new LoraLayer(targetName, downData, upData, inDim, outDim, rank, alpha));
            }
        }

        return result;
    }

    /// <summary>
    /// Applies LoRA layers directly onto target weight dictionary in-place.
    /// </summary>
    public static int ApplyToWeights(Dictionary<string, float[]> targetWeights, IEnumerable<LoraLayer> layers, float multiplier = 1.0f)
    {
        int appliedCount = 0;
        foreach (var layer in layers)
        {
            if (targetWeights.TryGetValue(layer.TargetName, out var baseWeight) ||
                targetWeights.TryGetValue($"{layer.TargetName}.weight", out baseWeight))
            {
                if (baseWeight.Length == layer.OutDim * layer.InDim)
                {
                    var delta = layer.ComputeDelta(multiplier);
                    for (int i = 0; i < baseWeight.Length; i++)
                        baseWeight[i] += delta[i];
                    appliedCount++;
                }
            }
        }
        return appliedCount;
    }

    private static string NormalizeTargetName(string loraKey)
    {
        // Standardize common LoRA prefix aliases (lora_unet_, lora_te_, model.diffusion_model.)
        string name = loraKey;
        if (name.StartsWith("lora_unet_", StringComparison.Ordinal))
            name = name["lora_unet_".Length..].Replace('_', '.');
        else if (name.StartsWith("lora_te_", StringComparison.Ordinal))
            name = name["lora_te_".Length..].Replace('_', '.');

        return name;
    }
}
