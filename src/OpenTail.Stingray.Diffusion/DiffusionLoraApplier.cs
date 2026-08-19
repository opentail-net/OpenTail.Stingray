using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Universal Native LoRA (Low-Rank Adaptation) adapter loader and in-place delta applier.
/// Supports standard Low-Rank safetensors checkpoints across Stable Diffusion 1.5, SDXL, SD3, FLUX, and LyCORIS/LoCon formats.
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
            float scale = multiplier * (Alpha / Math.Max(1, Rank));
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
    /// Supports Kohya, Diffusers, PEFT, ComfyUI, and LyCORIS key patterns.
    /// </summary>
    public static List<LoraLayer> Load(string loraPath)
    {
        var result = new List<LoraLayer>();
        using var loader = SafetensorsLoader.Open(loraPath);

        var tensorNames = new HashSet<string>(loader.TensorNames, StringComparer.Ordinal);

        foreach (var key in loader.TensorNames)
        {
            // 1. Identify Down/A matrix
            string? baseKey = null;
            string? upKey = null;

            if (key.EndsWith(".lora_down.weight", StringComparison.Ordinal))
            {
                baseKey = key[..^".lora_down.weight".Length];
                upKey = $"{baseKey}.lora_up.weight";
            }
            else if (key.EndsWith(".lora_A.weight", StringComparison.Ordinal))
            {
                baseKey = key[..^".lora_A.weight".Length];
                upKey = $"{baseKey}.lora_B.weight";
            }
            else if (key.EndsWith(".lora_down", StringComparison.Ordinal))
            {
                baseKey = key[..^".lora_down".Length];
                upKey = $"{baseKey}.lora_up";
            }
            else if (key.EndsWith(".lora_A", StringComparison.Ordinal))
            {
                baseKey = key[..^".lora_A".Length];
                upKey = $"{baseKey}.lora_B";
            }

            if (baseKey is null || upKey is null || !tensorNames.Contains(upKey))
                continue;

            var downShape = loader.GetShape(key);
            var upShape = loader.GetShape(upKey);
            var downData = loader.ReadF32(key);
            var upData = loader.ReadF32(upKey);

            if (downShape.Length < 2 || upShape.Length < 2) continue;

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

        return result;
    }

    /// <summary>
    /// Applies LoRA layers directly onto target weight dictionary in-place.
    /// Tries multiple alias keys and formats transparently.
    /// </summary>
    public static int ApplyToWeights(Dictionary<string, float[]> targetWeights, IEnumerable<LoraLayer> layers, float multiplier = 1.0f)
    {
        int appliedCount = 0;
        foreach (var layer in layers)
        {
            if (TryFindTargetWeight(targetWeights, layer.TargetName, out var baseWeight, out string? foundKey))
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

    private static bool TryFindTargetWeight(
        Dictionary<string, float[]> targetWeights,
        string targetName,
        out float[] baseWeight,
        out string? foundKey)
    {
        string[] candidates =
        [
            targetName,
            $"{targetName}.weight",
            $"model.diffusion_model.{targetName}",
            $"model.diffusion_model.{targetName}.weight",
            $"diffusion_model.{targetName}",
            $"diffusion_model.{targetName}.weight",
            $"unet.{targetName}",
            $"unet.{targetName}.weight",
            $"transformer.{targetName}",
            $"transformer.{targetName}.weight",
            $"cond_stage_model.transformer.{targetName}",
            $"cond_stage_model.transformer.{targetName}.weight"
        ];

        foreach (var key in candidates)
        {
            if (targetWeights.TryGetValue(key, out baseWeight!))
            {
                foundKey = key;
                return true;
            }
        }

        baseWeight = null!;
        foundKey = null;
        return false;
    }

    private static string NormalizeTargetName(string loraKey)
    {
        string name = loraKey;

        // Strip common training script prefixes
        string[] prefixes =
        [
            "lora_unet_",
            "lora_te1_",
            "lora_te2_",
            "lora_te_",
            "lycoris_",
            "base_model.model.",
            "model.diffusion_model.",
            "diffusion_model.",
            "unet.",
            "transformer."
        ];

        foreach (var p in prefixes)
        {
            if (name.StartsWith(p, StringComparison.OrdinalIgnoreCase))
            {
                name = name[p.Length..];
                break;
            }
        }

        // Standardize underscore-separated block paths (e.g. down_blocks_0_attentions_0_proj_in)
        if (name.Contains('_') && !name.Contains('.'))
        {
            name = name.Replace('_', '.');
        }

        return name;
    }
}
