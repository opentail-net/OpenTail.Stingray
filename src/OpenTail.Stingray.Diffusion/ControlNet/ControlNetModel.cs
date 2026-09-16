using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.ControlNet;

/// <summary>
/// Native C# ControlNet model for Stable Diffusion 1.5.
/// Computes 13 spatial residual feature maps (12 down-blocks + 1 middle-block)
/// from structural hint conditions (Canny, Depth, OpenPose, LineArt, etc.)
/// and merges them into UNet skip connections.
/// Reference: stable-diffusion.cpp:src/model/diffusion/control.hpp:ControlNetBlock
/// </summary>
public sealed class ControlNetModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly IComputeBackend? _backend;
    private readonly IImageOpsBackend? _imageOps;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly Dictionary<string, CoreTensor>? _gpuWeightsNative;
    private readonly Dictionary<string, CoreTensor> _gpuBiasCache = new(StringComparer.Ordinal);
    private readonly Dictionary<float[], CoreTensor> _cachedContextGpu = new(ReferenceEqualityComparer.Instance);
    private float[]? _cachedHintRgb;
    private float[]? _cachedHintFeatures;
    private bool? _controlNetForwardResidencySupported;
    private bool? _residentGpuAttentionSupported;
    private bool _gpuWeightsWarm;
    private bool _disposed;

    public void ClearContextCache()
    {
        lock (_cachedContextGpu)
        {
            if (_imageOps is not null)
            {
                foreach (var t in _cachedContextGpu.Values) _imageOps.Free(t);
            }
            _cachedContextGpu.Clear();
        }
    }

    public ControlNetModel(IWeightLoader weights, string prefix = "", IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
        _backend = backend;
        _imageOps = backend as IImageOpsBackend;
        if (backend is not null)
        {
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
            _gpuWeightsNative = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
        }
    }

    public static ControlNetModel Load(string path, IComputeBackend? backend = null)
    {
        IWeightLoader loader = path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(path)
            : SafetensorsLoader.Open(path);
        return new ControlNetModel(loader, prefix: "", backend: backend);
    }

    /// <summary>
    /// Pre-uploads all ControlNet conv weights, linear weights, groupnorm/layernorm biases, and zero-conv weights
    /// to GPU device memory so that forward passes can be recorded into a single Vulkan command buffer.
    /// </summary>
    public void EnsureGpuResident()
    {
        if (_imageOps is null || _gpuWeightsWarm) return;
        EnsureGpuResidentCore(_imageOps);
        _gpuWeightsWarm = true;
    }

    private void EnsureConvGpuResident(IImageOpsBackend imageOps, string name, int outCh)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");
        GetNativeConvWeights(imageOps, name, wF);
        GetGpuBias(imageOps, $"{name}.bias", bF ?? new float[outCh]);
    }

    private void EnsureLinGpuResident(string name)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");
        GetGpuWeight($"{name}.weight", wF);
        if (bF is not null && _imageOps is not null)
        {
            GetGpuBias(_imageOps, $"{name}.bias", bF);
        }
    }

    private void EnsureGroupNormGpuResident(IImageOpsBackend imageOps, string prefix, int c)
    {
        var (gnGamma, gnBeta) = GetNormWeights(prefix, c);
        GetGpuBias(imageOps, $"{prefix}.weight", gnGamma);
        GetGpuBias(imageOps, $"{prefix}.bias", gnBeta);
    }

    private void EnsureLayerNormGpuResident(IImageOpsBackend imageOps, string prefix, int c)
    {
        var (lnGamma, lnBeta) = GetLayerNormWeights(prefix, c);
        GetGpuBias(imageOps, $"{prefix}.weight", lnGamma);
        GetGpuBias(imageOps, $"{prefix}.bias", lnBeta);
    }

    private void EnsureResBlockGpuResident(IImageOpsBackend imageOps, string prefix, int inC, int outC)
    {
        EnsureGroupNormGpuResident(imageOps, $"{prefix}.in_layers.0", inC);
        EnsureConvGpuResident(imageOps, $"{prefix}.in_layers.2", outC);
        EnsureLinGpuResident($"{prefix}.emb_layers.1");
        EnsureGroupNormGpuResident(imageOps, $"{prefix}.out_layers.0", outC);
        EnsureConvGpuResident(imageOps, $"{prefix}.out_layers.3", outC);
        if (TryGetWeight($"{prefix}.skip_connection.weight") is not null || inC != outC)
        {
            EnsureConvGpuResident(imageOps, $"{prefix}.skip_connection", outC);
        }
    }

    private void EnsureSpatialTransformerGpuResident(IImageOpsBackend imageOps, string prefix, int c)
    {
        EnsureGroupNormGpuResident(imageOps, $"{prefix}.norm", c);
        EnsureLinGpuResident($"{prefix}.proj_in");

        string tb = $"{prefix}.transformer_blocks.0";
        EnsureLayerNormGpuResident(imageOps, $"{tb}.norm1", c);
        EnsureLinGpuResident($"{tb}.attn1.to_q");
        EnsureLinGpuResident($"{tb}.attn1.to_k");
        EnsureLinGpuResident($"{tb}.attn1.to_v");
        EnsureLinGpuResident($"{tb}.attn1.to_out.0");

        EnsureLayerNormGpuResident(imageOps, $"{tb}.norm2", c);
        EnsureLinGpuResident($"{tb}.attn2.to_q");
        EnsureLinGpuResident($"{tb}.attn2.to_k");
        EnsureLinGpuResident($"{tb}.attn2.to_v");
        EnsureLinGpuResident($"{tb}.attn2.to_out.0");

        EnsureLayerNormGpuResident(imageOps, $"{tb}.norm3", c);
        EnsureLinGpuResident($"{tb}.ff.net.0.proj");
        EnsureLinGpuResident($"{tb}.ff.net.2");

        EnsureLinGpuResident($"{prefix}.proj_out");
    }

    private void EnsureGpuResidentCore(IImageOpsBackend imageOps)
    {
        EnsureConvGpuResident(imageOps, "input_blocks.0.0", 320);
        EnsureResBlockGpuResident(imageOps, "input_blocks.1.0", 320, 320);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.1.1", 320);
        EnsureResBlockGpuResident(imageOps, "input_blocks.2.0", 320, 320);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.2.1", 320);
        EnsureConvGpuResident(imageOps, "input_blocks.3.0.op", 320);
        EnsureResBlockGpuResident(imageOps, "input_blocks.4.0", 320, 640);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.4.1", 640);
        EnsureResBlockGpuResident(imageOps, "input_blocks.5.0", 640, 640);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.5.1", 640);
        EnsureConvGpuResident(imageOps, "input_blocks.6.0.op", 640);
        EnsureResBlockGpuResident(imageOps, "input_blocks.7.0", 640, 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.7.1", 1280);
        EnsureResBlockGpuResident(imageOps, "input_blocks.8.0", 1280, 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "input_blocks.8.1", 1280);
        EnsureConvGpuResident(imageOps, "input_blocks.9.0.op", 1280);
        EnsureResBlockGpuResident(imageOps, "input_blocks.10.0", 1280, 1280);
        EnsureResBlockGpuResident(imageOps, "input_blocks.11.0", 1280, 1280);

        EnsureResBlockGpuResident(imageOps, "middle_block.0", 1280, 1280);
        EnsureSpatialTransformerGpuResident(imageOps, "middle_block.1", 1280);
        EnsureResBlockGpuResident(imageOps, "middle_block.2", 1280, 1280);

        int[] zeroConvChs = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280];
        for (int i = 0; i < 12; i++)
        {
            EnsureConvGpuResident(imageOps, $"zero_convs.{i}.0", zeroConvChs[i]);
        }
        EnsureConvGpuResident(imageOps, "middle_block_out.0", 1280);
    }

    private float[] GetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        var data = _weights.ReadF32(fullName);
        _weightCache[fullName] = data;
        return data;
    }

    private float[]? TryGetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        if (_weights.Contains(fullName))
        {
            var data = _weights.ReadF32(fullName);
            _weightCache[fullName] = data;
            return data;
        }
        return null;
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct) || _weights.Contains(direct + ".weight")) return direct;
        if (_weights.Contains("control_model." + direct) || _weights.Contains("control_model." + direct + ".weight")) return "control_model." + direct;
        if (_weights.Contains("controlnet." + direct) || _weights.Contains("controlnet." + direct + ".weight")) return "controlnet." + direct;

        string diffusers = MapCompVisToDiffusers(direct);
        if (_weights.Contains(diffusers) || _weights.Contains(diffusers + ".weight")) return diffusers;
        if (_weights.Contains("control_model." + diffusers) || _weights.Contains("control_model." + diffusers + ".weight")) return "control_model." + diffusers;
        if (_weights.Contains("controlnet." + diffusers) || _weights.Contains("controlnet." + diffusers + ".weight")) return "controlnet." + diffusers;

        return direct;
    }

    private static string MapCompVisToDiffusers(string k)
    {
        if (k.StartsWith("time_embed.0", StringComparison.Ordinal)) return "time_embedding.linear_1" + k["time_embed.0".Length..];
        if (k.StartsWith("time_embed.2", StringComparison.Ordinal)) return "time_embedding.linear_2" + k["time_embed.2".Length..];
        if (k.StartsWith("input_blocks.0.0", StringComparison.Ordinal)) return "conv_in" + k["input_blocks.0.0".Length..];
        if (k.StartsWith("input_hint_block.0", StringComparison.Ordinal)) return "controlnet_cond_embedding.conv_in" + k["input_hint_block.0".Length..];
        if (k.StartsWith("input_hint_block.2", StringComparison.Ordinal)) return "controlnet_cond_embedding.blocks.0" + k["input_hint_block.2".Length..];
        if (k.StartsWith("input_hint_block.4", StringComparison.Ordinal)) return "controlnet_cond_embedding.blocks.1" + k["input_hint_block.4".Length..];
        if (k.StartsWith("input_hint_block.6", StringComparison.Ordinal)) return "controlnet_cond_embedding.blocks.2" + k["input_hint_block.6".Length..];
        if (k.StartsWith("input_hint_block.8", StringComparison.Ordinal)) return "controlnet_cond_embedding.blocks.3" + k["input_hint_block.8".Length..];
        if (k.StartsWith("input_hint_block.10", StringComparison.Ordinal)) return "controlnet_cond_embedding.blocks.4" + k["input_hint_block.10".Length..];
        if (k.StartsWith("input_hint_block.12", StringComparison.Ordinal)) return "controlnet_cond_embedding.blocks.5" + k["input_hint_block.12".Length..];
        if (k.StartsWith("input_hint_block.14", StringComparison.Ordinal)) return "controlnet_cond_embedding.conv_out" + k["input_hint_block.14".Length..];
        if (k.StartsWith("middle_block_out.0", StringComparison.Ordinal)) return "controlnet_mid_block" + k["middle_block_out.0".Length..];

        for (int i = 0; i < 12; i++)
        {
            string prefix = $"zero_convs.{i}.0";
            if (k.StartsWith(prefix, StringComparison.Ordinal))
                return $"controlnet_down_blocks.{i}" + k[prefix.Length..];
        }

        string? mappedPrefix = null;
        string? remaining = null;

        if (k.StartsWith("input_blocks.1.0", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.0.resnets.0"; remaining = k["input_blocks.1.0".Length..]; }
        else if (k.StartsWith("input_blocks.1.1", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.0.attentions.0"; remaining = k["input_blocks.1.1".Length..]; }
        else if (k.StartsWith("input_blocks.2.0", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.0.resnets.1"; remaining = k["input_blocks.2.0".Length..]; }
        else if (k.StartsWith("input_blocks.2.1", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.0.attentions.1"; remaining = k["input_blocks.2.1".Length..]; }
        else if (k.StartsWith("input_blocks.3.0.op", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.0.downsamplers.0.conv"; remaining = k["input_blocks.3.0.op".Length..]; }
        else if (k.StartsWith("input_blocks.4.0", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.1.resnets.0"; remaining = k["input_blocks.4.0".Length..]; }
        else if (k.StartsWith("input_blocks.4.1", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.1.attentions.0"; remaining = k["input_blocks.4.1".Length..]; }
        else if (k.StartsWith("input_blocks.5.0", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.1.resnets.1"; remaining = k["input_blocks.5.0".Length..]; }
        else if (k.StartsWith("input_blocks.5.1", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.1.attentions.1"; remaining = k["input_blocks.5.1".Length..]; }
        else if (k.StartsWith("input_blocks.6.0.op", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.1.downsamplers.0.conv"; remaining = k["input_blocks.6.0.op".Length..]; }
        else if (k.StartsWith("input_blocks.7.0", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.2.resnets.0"; remaining = k["input_blocks.7.0".Length..]; }
        else if (k.StartsWith("input_blocks.7.1", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.2.attentions.0"; remaining = k["input_blocks.7.1".Length..]; }
        else if (k.StartsWith("input_blocks.8.0", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.2.resnets.1"; remaining = k["input_blocks.8.0".Length..]; }
        else if (k.StartsWith("input_blocks.8.1", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.2.attentions.1"; remaining = k["input_blocks.8.1".Length..]; }
        else if (k.StartsWith("input_blocks.9.0.op", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.2.downsamplers.0.conv"; remaining = k["input_blocks.9.0.op".Length..]; }
        else if (k.StartsWith("input_blocks.10.0", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.3.resnets.0"; remaining = k["input_blocks.10.0".Length..]; }
        else if (k.StartsWith("input_blocks.11.0", StringComparison.Ordinal)) { mappedPrefix = "down_blocks.3.resnets.1"; remaining = k["input_blocks.11.0".Length..]; }
        else if (k.StartsWith("middle_block.0", StringComparison.Ordinal)) { mappedPrefix = "mid_block.resnets.0"; remaining = k["middle_block.0".Length..]; }
        else if (k.StartsWith("middle_block.1", StringComparison.Ordinal)) { mappedPrefix = "mid_block.attentions.0"; remaining = k["middle_block.1".Length..]; }
        else if (k.StartsWith("middle_block.2", StringComparison.Ordinal)) { mappedPrefix = "mid_block.resnets.1"; remaining = k["middle_block.2".Length..]; }

        if (mappedPrefix is not null && remaining is not null)
        {
            remaining = remaining.Replace(".in_layers.0", ".norm1", StringComparison.Ordinal)
                                 .Replace(".in_layers.2", ".conv1", StringComparison.Ordinal)
                                 .Replace(".emb_layers.1", ".time_emb_proj", StringComparison.Ordinal)
                                 .Replace(".out_layers.0", ".norm2", StringComparison.Ordinal)
                                 .Replace(".out_layers.3", ".conv2", StringComparison.Ordinal)
                                 .Replace(".skip_connection.conv", ".conv_shortcut", StringComparison.Ordinal)
                                 .Replace(".skip_connection", ".conv_shortcut", StringComparison.Ordinal);
            return mappedPrefix + remaining;
        }

        return k;
    }

    private float[] ComputeHint(float[] hintRgb, int latH, int latW)
    {
        if (_cachedHintFeatures is not null && _cachedHintRgb is not null && ReferenceEquals(_cachedHintRgb, hintRgb))
            return _cachedHintFeatures;

        int hintH = latH * 8;
        int hintW = latW * 8;
        var hint = (float[])hintRgb.Clone();

        // 0: Conv 3 -> 16 (stride 1)
        hint = Conv("input_hint_block.0", hint, 3, hintH, hintW, 16, 3);
        DiffusionOps.SiluInPlace(hint);

        // 2: Conv 16 -> 16 (stride 1)
        hint = Conv("input_hint_block.2", hint, 16, hintH, hintW, 16, 3);
        DiffusionOps.SiluInPlace(hint);

        // 4: Conv 16 -> 32 (stride 2)
        hint = Conv("input_hint_block.4", hint, 16, hintH, hintW, 32, 3, stride: 2);
        hintH /= 2; hintW /= 2;
        DiffusionOps.SiluInPlace(hint);

        // 6: Conv 32 -> 32 (stride 1)
        hint = Conv("input_hint_block.6", hint, 32, hintH, hintW, 32, 3);
        DiffusionOps.SiluInPlace(hint);

        // 8: Conv 32 -> 96 (stride 2)
        hint = Conv("input_hint_block.8", hint, 32, hintH, hintW, 96, 3, stride: 2);
        hintH /= 2; hintW /= 2;
        DiffusionOps.SiluInPlace(hint);

        // 10: Conv 96 -> 96 (stride 1)
        hint = Conv("input_hint_block.10", hint, 96, hintH, hintW, 96, 3);
        DiffusionOps.SiluInPlace(hint);

        // 12: Conv 96 -> 256 (stride 2)
        hint = Conv("input_hint_block.12", hint, 96, hintH, hintW, 256, 3, stride: 2);
        hintH /= 2; hintW /= 2;
        DiffusionOps.SiluInPlace(hint);

        // 14: Conv 256 -> 320 (stride 1)
        hint = Conv("input_hint_block.14", hint, 256, hintH, hintW, 320, 3);

        _cachedHintRgb = hintRgb;
        _cachedHintFeatures = hint;
        return hint;
    }

    /// <summary>
    /// Computes ControlNet residuals from hint RGB image and current noisy latent.
    /// </summary>
    public (List<float[]> downResiduals, float[] midResidual) Forward(
        float[] latent,
        float timestep,
        float[] context,
        float[] hintRgb,
        int latH,
        int latW,
        float conditioningScale = 1.0f)
    {
        var hint = ComputeHint(hintRgb, latH, latW);
        var tEmb = ComputeTimeEmbedding(timestep, 320, 1280);

        if (_imageOps is not null && _controlNetForwardResidencySupported != false)
        {
            try
            {
                var (downGpu, midGpu) = ForwardGpuCore(_imageOps, latent, tEmb, context, hint, latH, latW, conditioningScale);
                var downResiduals = new List<float[]>(12);
                for (int i = 0; i < downGpu.Count; i++)
                {
                    var g = downGpu[i];
                    var buf = new float[(int)g.ElementCount];
                    _imageOps.Download(g, buf);
                    _imageOps.Free(g);
                    downResiduals.Add(buf);
                }

                var midResidual = new float[(int)midGpu.ElementCount];
                _imageOps.Download(midGpu, midResidual);
                _imageOps.Free(midGpu);

                _controlNetForwardResidencySupported = true;
                return (downResiduals, midResidual);
            }
            catch (NotSupportedException)
            {
                _controlNetForwardResidencySupported = false;
            }
        }

        // 3. Input Block 0: Conv 4 -> 320 + hint
        int h = latH, w = latW;
        var cur = Conv("input_blocks.0.0", latent, 4, h, w, 320, 3);
        for (int i = 0; i < cur.Length; i++)
            cur[i] += hint[i];

        var downResidualsFallback = new List<float[]>(12);
        downResidualsFallback.Add(ZeroConv("zero_convs.0.0", cur, 320, h, w, conditioningScale));

        // Block 1
        cur = ResBlock("input_blocks.1.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.1.1", cur, context, 320, h, w);
        downResidualsFallback.Add(ZeroConv("zero_convs.1.0", cur, 320, h, w, conditioningScale));

        // Block 2
        cur = ResBlock("input_blocks.2.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.2.1", cur, context, 320, h, w);
        downResidualsFallback.Add(ZeroConv("zero_convs.2.0", cur, 320, h, w, conditioningScale));

        // Block 3: Downsample (Conv2D 320 -> 320, stride 2)
        cur = Conv("input_blocks.3.0.op", cur, 320, h, w, 320, 3, stride: 2);
        h /= 2; w /= 2;
        downResidualsFallback.Add(ZeroConv("zero_convs.3.0", cur, 320, h, w, conditioningScale));

        // Block 4
        cur = ResBlock("input_blocks.4.0", cur, tEmb, 320, 640, h, w);
        cur = SpatialTransformer("input_blocks.4.1", cur, context, 640, h, w);
        downResidualsFallback.Add(ZeroConv("zero_convs.4.0", cur, 640, h, w, conditioningScale));

        // Block 5
        cur = ResBlock("input_blocks.5.0", cur, tEmb, 640, 640, h, w);
        cur = SpatialTransformer("input_blocks.5.1", cur, context, 640, h, w);
        downResidualsFallback.Add(ZeroConv("zero_convs.5.0", cur, 640, h, w, conditioningScale));

        // Block 6: Downsample (Conv2D 640 -> 640, stride 2)
        cur = Conv("input_blocks.6.0.op", cur, 640, h, w, 640, 3, stride: 2);
        h /= 2; w /= 2;
        downResidualsFallback.Add(ZeroConv("zero_convs.6.0", cur, 640, h, w, conditioningScale));

        // Block 7
        cur = ResBlock("input_blocks.7.0", cur, tEmb, 640, 1280, h, w);
        cur = SpatialTransformer("input_blocks.7.1", cur, context, 1280, h, w);
        downResidualsFallback.Add(ZeroConv("zero_convs.7.0", cur, 1280, h, w, conditioningScale));

        // Block 8
        cur = ResBlock("input_blocks.8.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("input_blocks.8.1", cur, context, 1280, h, w);
        downResidualsFallback.Add(ZeroConv("zero_convs.8.0", cur, 1280, h, w, conditioningScale));

        // Block 9: Downsample (Conv2D 1280 -> 1280, stride 2)
        cur = Conv("input_blocks.9.0.op", cur, 1280, h, w, 1280, 3, stride: 2);
        h /= 2; w /= 2;
        downResidualsFallback.Add(ZeroConv("zero_convs.9.0", cur, 1280, h, w, conditioningScale));

        // Block 10
        cur = ResBlock("input_blocks.10.0", cur, tEmb, 1280, 1280, h, w);
        downResidualsFallback.Add(ZeroConv("zero_convs.10.0", cur, 1280, h, w, conditioningScale));

        // Block 11
        cur = ResBlock("input_blocks.11.0", cur, tEmb, 1280, 1280, h, w);
        downResidualsFallback.Add(ZeroConv("zero_convs.11.0", cur, 1280, h, w, conditioningScale));

        // 4. Middle Block
        cur = ResBlock("middle_block.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("middle_block.1", cur, context, 1280, h, w);
        cur = ResBlock("middle_block.2", cur, tEmb, 1280, 1280, h, w);

        var midResidualFallback = ZeroConv("middle_block_out.0", cur, 1280, h, w, conditioningScale);

        return (downResidualsFallback, midResidualFallback);
    }

    /// <summary>
    /// Computes ControlNet residuals on GPU and returns device resident CoreTensors directly,
    /// eliminating host-device synchronization and downloads during diffusion sampling.
    /// </summary>
    public (List<CoreTensor> downResidualsGpu, CoreTensor midResidualGpu) ForwardGpuTensors(
        float[] latent,
        float timestep,
        float[] context,
        float[] hintRgb,
        int latH,
        int latW,
        float conditioningScale = 1.0f)
    {
        if (_imageOps is null)
            throw new InvalidOperationException("ForwardGpuTensors requires an active GPU imageOps backend.");

        EnsureGpuResident();

        var hint = ComputeHint(hintRgb, latH, latW);
        var tEmb = ComputeTimeEmbedding(timestep, 320, 1280);

        return ForwardGpuCore(_imageOps, latent, tEmb, context, hint, latH, latW, conditioningScale);
    }

    private (List<CoreTensor> downResidualsGpu, CoreTensor midResidualGpu) ForwardGpuCore(
        IImageOpsBackend imageOps,
        float[] latent,
        float[] tEmb,
        float[] context,
        float[] hint,
        int latH,
        int latW,
        float conditioningScale)
    {
        EnsureGpuResident();

        int h = latH, w = latW;

        // Pre-upload all host inputs before BeginBatch() so transfer commands do not interrupt batch recording
        CoreTensor contextGpu;
        lock (_cachedContextGpu)
        {
            if (!_cachedContextGpu.TryGetValue(context, out contextGpu!))
            {
                contextGpu = imageOps.Upload(context.AsSpan(0, 77 * 768), TensorShape.D1(77 * 768));
                _cachedContextGpu[context] = contextGpu;
            }
        }
        var tEmbGpu = UploadSiluTEmb(imageOps, tEmb);
        var hintGpu = imageOps.Upload(hint.AsSpan(0, 320 * h * w), TensorShape.D1(320 * h * w));
        var xGpu = imageOps.Upload(latent.AsSpan(0, 4 * h * w), TensorShape.D1(4 * h * w));

        var downResidualsGpu = new List<CoreTensor>(12);
        bool batchSuccess = false;

        try
        {
            imageOps.BeginBatch();

            var cur = ConvGpuTensor(imageOps, "input_blocks.0.0", xGpu, 4, h, w, 320, 3);
            imageOps.Free(xGpu);
            imageOps.AddInPlace(cur, hintGpu);
            imageOps.Free(hintGpu);

            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.0.0", cur, 320, h, w, conditioningScale));

            // Block 1
            cur = ResBlockGpu(imageOps, "input_blocks.1.0", cur, tEmbGpu, 320, h, w, 320);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.1.1", cur, contextGpu, 320, h, w);
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.1.0", cur, 320, h, w, conditioningScale));

            // Block 2
            cur = ResBlockGpu(imageOps, "input_blocks.2.0", cur, tEmbGpu, 320, h, w, 320);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.2.1", cur, contextGpu, 320, h, w);
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.2.0", cur, 320, h, w, conditioningScale));

            // Block 3: Downsample (Conv2D 320 -> 320, stride 2)
            cur = ConvGpuTensor(imageOps, "input_blocks.3.0.op", cur, 320, h, w, 320, 3, padding: 1, stride: 2);
            h /= 2; w /= 2;
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.3.0", cur, 320, h, w, conditioningScale));

            // Block 4
            cur = ResBlockGpu(imageOps, "input_blocks.4.0", cur, tEmbGpu, 320, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.4.1", cur, contextGpu, 640, h, w);
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.4.0", cur, 640, h, w, conditioningScale));

            // Block 5
            cur = ResBlockGpu(imageOps, "input_blocks.5.0", cur, tEmbGpu, 640, h, w, 640);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.5.1", cur, contextGpu, 640, h, w);
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.5.0", cur, 640, h, w, conditioningScale));

            // Block 6: Downsample (Conv2D 640 -> 640, stride 2)
            cur = ConvGpuTensor(imageOps, "input_blocks.6.0.op", cur, 640, h, w, 640, 3, padding: 1, stride: 2);
            h /= 2; w /= 2;
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.6.0", cur, 640, h, w, conditioningScale));

            // Block 7
            cur = ResBlockGpu(imageOps, "input_blocks.7.0", cur, tEmbGpu, 640, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.7.1", cur, contextGpu, 1280, h, w);
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.7.0", cur, 1280, h, w, conditioningScale));

            // Block 8
            cur = ResBlockGpu(imageOps, "input_blocks.8.0", cur, tEmbGpu, 1280, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "input_blocks.8.1", cur, contextGpu, 1280, h, w);
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.8.0", cur, 1280, h, w, conditioningScale));

            // Block 9: Downsample (Conv2D 1280 -> 1280, stride 2)
            cur = ConvGpuTensor(imageOps, "input_blocks.9.0.op", cur, 1280, h, w, 1280, 3, padding: 1, stride: 2);
            h /= 2; w /= 2;
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.9.0", cur, 1280, h, w, conditioningScale));

            // Block 10
            cur = ResBlockGpu(imageOps, "input_blocks.10.0", cur, tEmbGpu, 1280, h, w, 1280);
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.10.0", cur, 1280, h, w, conditioningScale));

            // Block 11
            cur = ResBlockGpu(imageOps, "input_blocks.11.0", cur, tEmbGpu, 1280, h, w, 1280);
            downResidualsGpu.Add(ZeroConvGpu(imageOps, "zero_convs.11.0", cur, 1280, h, w, conditioningScale));

            // Middle Block
            cur = ResBlockGpu(imageOps, "middle_block.0", cur, tEmbGpu, 1280, h, w, 1280);
            cur = SpatialTransformerGpu(imageOps, "middle_block.1", cur, contextGpu, 1280, h, w);
            cur = ResBlockGpu(imageOps, "middle_block.2", cur, tEmbGpu, 1280, h, w, 1280);

            var midResidualGpu = ZeroConvGpu(imageOps, "middle_block_out.0", cur, 1280, h, w, conditioningScale);
            imageOps.Free(cur);

            imageOps.Free(tEmbGpu);

            imageOps.EndBatch();
            batchSuccess = true;

            return (downResidualsGpu, midResidualGpu);
        }
        finally
        {
            if (!batchSuccess)
            {
                try { imageOps.EndBatch(); } catch { }
                foreach (var g in downResidualsGpu) imageOps.Free(g);
                imageOps.Free(hintGpu);
                imageOps.Free(tEmbGpu);
                imageOps.Free(xGpu);
            }
        }
    }


    private CoreTensor GetGpuWeight(string name, float[] cpuWeight)
    {
        string fullName = Resolve(name);
        if (_gpuWeights!.TryGetValue(fullName, out var wGpu)) return wGpu;

        if (_backend!.BestSgemmPrecision == SgemmPrecision.Fp16)
        {
            var half = new Half[cpuWeight.Length];
            TensorPrimitives.ConvertToHalf(cpuWeight, half);
            wGpu = _backend.UploadHalf(half, TensorShape.D1(cpuWeight.Length));
        }
        else
        {
            wGpu = _backend.Upload(cpuWeight.AsSpan(), TensorShape.D1(cpuWeight.Length));
        }
        _gpuWeights[fullName] = wGpu;
        return wGpu;
    }

    private CoreTensor GetNativeConvWeights(IImageOpsBackend imageOps, string name, float[] wF)
    {
        string fullName = Resolve($"{name}.weight");
        if (_gpuWeightsNative!.TryGetValue(fullName, out var wGpu)) return wGpu;
        wGpu = imageOps.Upload(wF.AsSpan(), TensorShape.D1(wF.Length));
        _gpuWeightsNative[fullName] = wGpu;
        return wGpu;
    }

    private CoreTensor GetGpuBias(IImageOpsBackend imageOps, string name, float[] biasData)
    {
        string fullName = Resolve(name);
        if (_gpuBiasCache.TryGetValue(fullName, out var cached)) return cached;
        var t = imageOps.Upload(biasData.AsSpan(), TensorShape.D1(biasData.Length));
        _gpuBiasCache[fullName] = t;
        return t;
    }

    private CoreTensor UploadSiluTEmb(IImageOpsBackend imageOps, float[] tEmb)
    {
        var activated = new float[tEmb.Length];
        for (int i = 0; i < tEmb.Length; i++)
        {
            float v = tEmb[i];
            activated[i] = v / (1.0f + MathF.Exp(-v));
        }
        return imageOps.Upload(activated.AsSpan(), TensorShape.D2(1, activated.Length));
    }

    private CoreTensor ConvGpuTensor(IImageOpsBackend imageOps, string name, CoreTensor xGpu, int inCh, int h, int w, int outCh, int k, int padding = -1, int stride = 1)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");
        var wGpu = GetNativeConvWeights(imageOps, name, wF);
        var bGpu = GetGpuBias(imageOps, $"{name}.bias", bF ?? new float[outCh]);
        return imageOps.Conv2dImplicitGemm(xGpu, wGpu, bGpu, inCh, outCh, h, w, k, padding, stride);
    }

    private CoreTensor LinGpuTensor(string name, CoreTensor xGpu, int n, int inDim, int outDim)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");
        var wGpu = GetGpuWeight($"{name}.weight", wF);
        var cGpu = _backend!.Allocate(TensorShape.D1(n * outDim));
        _backend.Sgemm(cGpu, xGpu, wGpu, n, inDim, outDim);
        if (bF is not null && _imageOps is not null)
        {
            var bGpu = GetGpuBias(_imageOps, $"{name}.bias", bF);
            _imageOps.AddRowBroadcastInPlace(cGpu, bGpu, n, outDim);
        }
        return cGpu;
    }

    private CoreTensor GroupNormSiluGpuTensor(IImageOpsBackend imageOps, string name, CoreTensor xGpu, int channels, int hw)
    {
        var (gnGamma, gnBeta) = GetNormWeights(name, channels);
        var gGpu = GetGpuBias(imageOps, $"{name}.weight", gnGamma);
        var bGpu = GetGpuBias(imageOps, $"{name}.bias", gnBeta);
        return imageOps.GroupNormSilu(xGpu, gGpu, bGpu, channels, hw);
    }

    private CoreTensor ResBlockGpu(IImageOpsBackend imageOps, string name, CoreTensor xGpu, CoreTensor tEmbGpu, int inC, int h, int w, int outC)
    {
        var h1 = GroupNormSiluGpuTensor(imageOps, $"{name}.in_layers.0", xGpu, inC, h * w);
        var h1Conv = ConvGpuTensor(imageOps, $"{name}.in_layers.2", h1, inC, h, w, outC, 3);
        imageOps.Free(h1);

        var tProj = LinGpuTensor($"{name}.emb_layers.1", tEmbGpu, 1, 1280, outC);
        imageOps.AddChannelBroadcastInPlace(h1Conv, tProj, outC, h * w);
        imageOps.Free(tProj);

        var h2 = GroupNormSiluGpuTensor(imageOps, $"{name}.out_layers.0", h1Conv, outC, h * w);
        imageOps.Free(h1Conv);
        var h2Conv = ConvGpuTensor(imageOps, $"{name}.out_layers.3", h2, outC, h, w, outC, 3);
        imageOps.Free(h2);

        CoreTensor residualGpu;
        if (inC != outC)
        {
            string scKey = Resolve($"{name}.skip_connection");
            if (_weights.Contains($"{scKey}.weight"))
                residualGpu = ConvGpuTensor(imageOps, $"{name}.skip_connection", xGpu, inC, h, w, outC, 1, padding: 0);
            else if (_weights.Contains($"{scKey}.conv.weight"))
                residualGpu = ConvGpuTensor(imageOps, $"{name}.skip_connection.conv", xGpu, inC, h, w, outC, 1, padding: 0);
            else
                residualGpu = xGpu;
        }
        else
        {
            residualGpu = xGpu;
        }

        imageOps.AddInPlace(h2Conv, residualGpu);
        if (residualGpu != xGpu) imageOps.Free(residualGpu);
        imageOps.Free(xGpu);
        return h2Conv;
    }

    private CoreTensor SpatialTransformerGpu(IImageOpsBackend imageOps, string prefix, CoreTensor xGpu, CoreTensor contextGpu, int c, int h, int w)
    {
        int hw = h * w;
        var normW = GetGpuBias(imageOps, $"{prefix}.norm.weight", GetNormWeights($"{prefix}.norm", c).g);
        var normB = GetGpuBias(imageOps, $"{prefix}.norm.bias", GetNormWeights($"{prefix}.norm", c).b);
        var xNorm = imageOps.GroupNormGpu(xGpu, normW, normB, c, hw);

        // Perf: 1x1 conv is a pure linear projection. Permuting [C, HW] -> [HW, C] first allows
        // running proj_in via SgemmF16 (128-bit vector loads, fp16 weights) instead of the scalar Conv2dImplicitGemm.
        var xNormSeq = imageOps.PermuteChwToHwc(xNorm, c, hw);
        imageOps.Free(xNorm);

        var xSeq = LinGpuTensor($"{prefix}.proj_in", xNormSeq, hw, c, c);
        imageOps.Free(xNormSeq);

        string tb = $"{prefix}.transformer_blocks.0";

        // 1. Self-attention
        var saNormW = GetGpuBias(imageOps, $"{tb}.norm1.weight", GetLayerNormWeights($"{tb}.norm1", c).g);
        var saNormB = GetGpuBias(imageOps, $"{tb}.norm1.bias", GetLayerNormWeights($"{tb}.norm1", c).b);
        var saNorm = imageOps.LayerNormGpu(xSeq, saNormW, saNormB, hw, c);

        var saQ = LinGpuTensor($"{tb}.attn1.to_q", saNorm, hw, c, c);
        var saK = LinGpuTensor($"{tb}.attn1.to_k", saNorm, hw, c, c);
        var saV = LinGpuTensor($"{tb}.attn1.to_v", saNorm, hw, c, c);
        imageOps.Free(saNorm);

        var saAttnOut = AttentionIsland(imageOps, saQ, saK, saV, hw, hw, c, 8);
        imageOps.Free(saQ); imageOps.Free(saK); imageOps.Free(saV);

        var saProjOut = LinGpuTensor($"{tb}.attn1.to_out.0", saAttnOut, hw, c, c);
        imageOps.Free(saAttnOut);
        imageOps.AddInPlace(xSeq, saProjOut);
        imageOps.Free(saProjOut);

        // 2. Cross-attention
        var caNormW = GetGpuBias(imageOps, $"{tb}.norm2.weight", GetLayerNormWeights($"{tb}.norm2", c).g);
        var caNormB = GetGpuBias(imageOps, $"{tb}.norm2.bias", GetLayerNormWeights($"{tb}.norm2", c).b);
        var caNorm = imageOps.LayerNormGpu(xSeq, caNormW, caNormB, hw, c);

        var caQ = LinGpuTensor($"{tb}.attn2.to_q", caNorm, hw, c, c);
        imageOps.Free(caNorm);
        var caK = LinGpuTensor($"{tb}.attn2.to_k", contextGpu, 77, 768, c);
        var caV = LinGpuTensor($"{tb}.attn2.to_v", contextGpu, 77, 768, c);

        var caAttnOut = AttentionIsland(imageOps, caQ, caK, caV, hw, 77, c, 8);
        imageOps.Free(caQ); imageOps.Free(caK); imageOps.Free(caV);

        var caProjOut = LinGpuTensor($"{tb}.attn2.to_out.0", caAttnOut, hw, c, c);
        imageOps.Free(caAttnOut);
        imageOps.AddInPlace(xSeq, caProjOut);
        imageOps.Free(caProjOut);

        // 3. GEGLU FeedForward
        var ffNormW = GetGpuBias(imageOps, $"{tb}.norm3.weight", GetLayerNormWeights($"{tb}.norm3", c).g);
        var ffNormB = GetGpuBias(imageOps, $"{tb}.norm3.bias", GetLayerNormWeights($"{tb}.norm3", c).b);
        var ffNorm = imageOps.LayerNormGpu(xSeq, ffNormW, ffNormB, hw, c);

        int mlpDim = c * 4;
        var ffH = LinGpuTensor($"{tb}.ff.net.0.proj", ffNorm, hw, c, mlpDim * 2);
        imageOps.Free(ffNorm);
        var ffGated = imageOps.GeGlu(ffH, hw, mlpDim);
        imageOps.Free(ffH);

        var ffOut = LinGpuTensor($"{tb}.ff.net.2", ffGated, hw, mlpDim, c);
        imageOps.Free(ffGated);
        imageOps.AddInPlace(xSeq, ffOut);
        imageOps.Free(ffOut);

        // Perf: run proj_out directly on xSeq [HW, C] via SgemmF16 before permuting back to [C, HW]
        var projOutSeq = LinGpuTensor($"{prefix}.proj_out", xSeq, hw, c, c);
        imageOps.Free(xSeq);

        var projOut = imageOps.PermuteHwcToChw(projOutSeq, c, hw);
        imageOps.Free(projOutSeq);

        imageOps.AddInPlace(projOut, xGpu);
        return projOut;
    }

    private CoreTensor ZeroConvGpu(IImageOpsBackend imageOps, string name, CoreTensor xGpu, int channels, int h, int w, float scale)
    {
        var outGpu = ConvGpuTensor(imageOps, name, xGpu, channels, h, w, channels, 1, padding: 0);
        if (scale != 1.0f)
        {
            imageOps.ScaleInPlace(outGpu, scale);
        }
        return outGpu;
    }



    private CoreTensor AttentionIsland(IImageOpsBackend imageOps, CoreTensor qGpu, CoreTensor kGpu, CoreTensor vGpu, int qSeq, int kvSeq, int c, int nHeads)
    {
        int headDim = c / nHeads;
        if (headDim <= 256 && _residentGpuAttentionSupported != false)
        {
            try
            {
                var result = imageOps.MultiHeadAttentionTiled(qGpu, kGpu, vGpu, qSeq, kvSeq, nHeads, headDim);
                _residentGpuAttentionSupported = true;
                return result;
            }
            catch (NotSupportedException)
            {
                _residentGpuAttentionSupported = false;
            }
        }
        return CpuAttentionIsland(imageOps, qGpu, kGpu, vGpu, qSeq, kvSeq, c, nHeads, headDim);
    }

    private static CoreTensor CpuAttentionIsland(IImageOpsBackend imageOps, CoreTensor qGpu, CoreTensor kGpu, CoreTensor vGpu, int qSeq, int kvSeq, int c, int nHeads, int headDim)
    {
        var q = new float[qSeq * c];
        var k = new float[kvSeq * c];
        var v = new float[kvSeq * c];
        imageOps.Download(qGpu, q);
        imageOps.Download(kGpu, k);
        imageOps.Download(vGpu, v);

        var attnOut = DiffusionOps.MultiHeadAttention(q, k, v, qSeq, kvSeq, nHeads, headDim);
        return imageOps.Upload(attnOut.AsSpan(), TensorShape.D1(attnOut.Length));
    }

    private float[] ZeroConv(string name, float[] x, int channels, int h, int w, float scale)
    {
        var outF = Conv(name, x, channels, h, w, channels, 1, padding: 0);
        if (scale != 1.0f)
        {
            for (int i = 0; i < outF.Length; i++)
                outF[i] *= scale;
        }
        return outF;
    }

    private float[] Conv(string name, float[] x, int inC, int h, int w, int outC, int ksize, int stride = 1, int padding = -1)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");
        return DiffusionOps.Conv2D(x, wF, bF, 1, inC, h, w, outC, ksize, ksize, stride, padding);
    }

    private float[] ResBlock(string name, float[] x, float[] tEmb, int inC, int outC, int h, int w)
    {
        var (gn1Gamma, gn1Beta) = GetNormWeights($"{name}.in_layers.0", inC);
        var h1 = (float[])x.Clone();
        DiffusionOps.GroupNorm(h1, gn1Gamma, gn1Beta, 1, inC, h, w, 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = Conv($"{name}.in_layers.2", h1, inC, h, w, outC, 3);

        var tEmbAct = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(tEmbAct);
        var tProj = Linear($"{name}.emb_layers.1", tEmbAct, 1280, outC);
        for (int c = 0; c < outC; c++)
        {
            float bias = tProj[c];
            int offset = c * h * w;
            for (int i = 0; i < h * w; i++)
                h1[offset + i] += bias;
        }

        var (gn2Gamma, gn2Beta) = GetNormWeights($"{name}.out_layers.0", outC);
        var h2 = (float[])h1.Clone();
        DiffusionOps.GroupNorm(h2, gn2Gamma, gn2Beta, 1, outC, h, w, 32);
        DiffusionOps.SiluInPlace(h2);
        h2 = Conv($"{name}.out_layers.3", h2, outC, h, w, outC, 3);

        var residual = x;
        if (inC != outC)
        {
            string scKey = Resolve($"{name}.skip_connection");
            if (_weights.Contains($"{scKey}.weight"))
                residual = Conv($"{name}.skip_connection", x, inC, h, w, outC, 1, padding: 0);
            else if (_weights.Contains($"{scKey}.conv.weight"))
                residual = Conv($"{name}.skip_connection.conv", x, inC, h, w, outC, 1, padding: 0);
        }

        for (int i = 0; i < h2.Length; i++)
            h2[i] += residual[i];

        return h2;
    }

    private float[] SpatialTransformer(string name, float[] x, float[] context, int channels, int h, int w)
    {
        string p = $"{name}.transformer_blocks.0";
        var (gnGamma, gnBeta) = GetNormWeights($"{name}.norm", channels);
        var normX = (float[])x.Clone();
        DiffusionOps.GroupNorm(normX, gnGamma, gnBeta, 1, channels, h, w, 32);

        var projIn = Conv($"{name}.proj_in", normX, channels, h, w, channels, 1, padding: 0);

        int seqLen = h * w;
        // Permute [1, C, H, W] -> [H*W, C] sequence
        var xSeq = new float[seqLen * channels];
        for (int ch = 0; ch < channels; ch++)
        {
            int chOff = ch * seqLen;
            for (int s = 0; s < seqLen; s++)
                xSeq[s * channels + ch] = projIn[chOff + s];
        }

        // 1. Self-Attention
        var (n1G, n1B) = GetLayerNormWeights($"{p}.norm1", channels);
        var h1 = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(h1, n1G, n1B, channels);
        var q1 = Linear($"{p}.attn1.to_q", h1, channels, channels);
        var k1 = Linear($"{p}.attn1.to_k", h1, channels, channels);
        var v1 = Linear($"{p}.attn1.to_v", h1, channels, channels);
        var attn1Out = DiffusionOps.MultiHeadAttention(q1, k1, v1, seqLen, seqLen, 8, channels / 8);
        var saProjOut = Linear($"{p}.attn1.to_out.0", attn1Out, channels, channels);
        TensorPrimitives.Add(xSeq, saProjOut, xSeq);

        // 2. Cross-Attention
        var (n2G, n2B) = GetLayerNormWeights($"{p}.norm2", channels);
        var h2 = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(h2, n2G, n2B, channels);
        var q2 = Linear($"{p}.attn2.to_q", h2, channels, channels);
        var k2 = Linear($"{p}.attn2.to_k", context, 768, channels);
        var v2 = Linear($"{p}.attn2.to_v", context, 768, channels);
        var attn2Out = DiffusionOps.MultiHeadAttention(q2, k2, v2, seqLen, 77, 8, channels / 8);
        var caProjOut = Linear($"{p}.attn2.to_out.0", attn2Out, channels, channels);
        TensorPrimitives.Add(xSeq, caProjOut, xSeq);

        // 3. GEGLU FeedForward
        var (n3G, n3B) = GetLayerNormWeights($"{p}.norm3", channels);
        var h3 = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(h3, n3G, n3B, channels);
        var ff1 = LinearGeluGeGLU($"{p}.ff.net.0.proj", h3, channels, channels * 4);
        var ffOut = Linear($"{p}.ff.net.2", ff1, channels * 4, channels);
        TensorPrimitives.Add(xSeq, ffOut, xSeq);

        // Permute [H*W, C] back to [1, C, H, W]
        var xSpatial = new float[seqLen * channels];
        for (int ch = 0; ch < channels; ch++)
        {
            int chOff = ch * seqLen;
            for (int s = 0; s < seqLen; s++)
                xSpatial[chOff + s] = xSeq[s * channels + ch];
        }

        var projOut = Conv($"{name}.proj_out", xSpatial, channels, h, w, channels, 1, padding: 0);
        TensorPrimitives.Add(projOut, x, projOut);
        return projOut;
    }

    private static float[] AttentionHeads(float[] q, float[] k, float[] v, int qSeq, int kvSeq, int dim, int heads)
    {
        int headDim = dim / heads;
        float scale = 1.0f / MathF.Sqrt(headDim);
        var output = new float[qSeq * dim];

        for (int h = 0; h < heads; h++)
        {
            int hOff = h * headDim;
            for (int i = 0; i < qSeq; i++)
            {
                int qRow = i * dim + hOff;
                var scores = new float[kvSeq];
                float maxScore = float.NegativeInfinity;

                for (int j = 0; j < kvSeq; j++)
                {
                    int kRow = j * dim + hOff;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qRow + d] * k[kRow + d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < kvSeq; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - maxScore);
                    sumExp += scores[j];
                }
                float invSum = 1f / sumExp;
                for (int j = 0; j < kvSeq; j++) scores[j] *= invSum;

                int outRow = i * dim + hOff;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kvSeq; j++)
                        sum += scores[j] * v[j * dim + hOff + d];
                    output[outRow + d] = sum;
                }
            }
        }

        return output;
    }

    private float[] LinearGeluGeGLU(string name, float[] x, int inDim, int outDim)
    {
        var proj = Linear(name, x, inDim, outDim * 2);
        int rows = x.Length / inDim;
        var result = new float[rows * outDim];

        for (int r = 0; r < rows; r++)
        {
            int srcOff = r * (outDim * 2);
            int dstOff = r * outDim;
            for (int c = 0; c < outDim; c++)
            {
                float val = proj[srcOff + c];
                float gate = proj[srcOff + outDim + c];
                result[dstOff + c] = val * DiffusionOps.Gelu(gate);
            }
        }
        return result;
    }

    private float[] Linear(string name, float[] x, int inDim, int outDim)
    {
        var w = GetWeight($"{name}.weight");
        var b = TryGetWeight($"{name}.bias");
        int rows = x.Length / inDim;
        return DiffusionOps.Linear(x, w, b, rows, inDim, outDim);
    }

    private (float[] g, float[] b) GetNormWeights(string name, int dim)
    {
        var g = TryGetWeight($"{name}.weight") ?? new float[dim];
        if (!_weights.Contains(Resolve($"{name}.weight"))) Array.Fill(g, 1.0f);
        var b = TryGetWeight($"{name}.bias") ?? new float[dim];
        return (g, b);
    }

    private (float[] g, float[] b) GetLayerNormWeights(string name, int dim)
    {
        var g = TryGetWeight($"{name}.weight") ?? new float[dim];
        if (!_weights.Contains(Resolve($"{name}.weight"))) Array.Fill(g, 1.0f);
        var b = TryGetWeight($"{name}.bias") ?? new float[dim];
        return (g, b);
    }

    private float[] ComputeTimeEmbedding(float timestep, int dim, int outDim)
    {
        var emb = new float[dim];
        int half = dim / 2;
        float factor = 10000.0f;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(factor) * i / half);
            emb[i] = MathF.Cos(timestep * freq);
            emb[half + i] = MathF.Sin(timestep * freq);
        }

        var t0 = Linear("time_embed.0", emb, dim, outDim);
        DiffusionOps.SiluInPlace(t0);
        return Linear("time_embed.2", t0, outDim, outDim);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weights.Dispose();
            ClearContextCache();
            if (_gpuWeights is not null && _backend is not null)
            {
                foreach (var t in _gpuWeights.Values) _backend.Free(t);
                _gpuWeights.Clear();
            }
            if (_gpuWeightsNative is not null && _backend is not null)
            {
                foreach (var t in _gpuWeightsNative.Values) _backend.Free(t);
                _gpuWeightsNative.Clear();
            }
            if (_backend is not null)
            {
                foreach (var t in _gpuBiasCache.Values) _backend.Free(t);
                _gpuBiasCache.Clear();
            }
        }
    }
}
