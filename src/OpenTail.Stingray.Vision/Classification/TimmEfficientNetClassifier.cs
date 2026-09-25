using System.Globalization;
using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Vision.Classification;

/// <summary>
/// timm EfficientNet-family image classifiers (MobileNetV3 small/large, EfficientNet-B0) from a timm HF checkpoint
/// (<c>config.json</c> + <c>model.safetensors</c>), ported from the vendored timm source (<c>examples/timm</c>:
/// <c>mobilenetv3.py</c>, <c>efficientnet.py</c>, <c>_efficientnet_blocks.py</c>). The block list is timm's own arch-def
/// strings (<c>ds_r1_k3_s2_e1_c16_se0.25_nre</c>: type, repeats, kernel, first-repeat stride, expansion, channels, SE,
/// <c>nre</c> = ReLU instead of the model's default activation); channel counts, including SE widths, come from the
/// weights. BatchNorm (eps 1e-5) is folded into the preceding conv. Symmetric padding <c>((s-1) + (k-1)) / 2</c>.
/// Activations are laid out HWC; pointwise convs are packed GEMMs, depthwise convs direct loops over channels.
/// </summary>
public sealed class TimmEfficientNetClassifier : IDisposable
{
    private enum Act { None, Relu, HardSwish, Swish }

    private sealed record Spec(string Type, int Repeats, int Kernel, int Stride, bool Se, bool Relu);

    private sealed class Conv(int outCh, int inCh, int k, int stride, bool depthwise, float[] w, float[] b) : IDisposable
    {
        public readonly int OutCh = outCh, InCh = inCh, K = k, Stride = stride;
        public readonly bool Depthwise = depthwise;
        public readonly float[] W = w, B = b;               // folded; depthwise W: [k*k, ch] (tap-major for channel vectors)
        public readonly PackedLinearF32? Packed = !depthwise ? new PackedLinearF32(w, null, outCh, inCh * k * k) : null;
        public void Dispose() => Packed?.Dispose();
    }

    private sealed record Se(PackedLinearF32 Reduce, PackedLinearF32 Expand, bool HardSigmoid, Act Act);

    private sealed record Block(string Type, Conv? Pw, Conv Dw, Se? Se, Conv? Pwl, Act Act, bool Skip);

    private readonly Conv _stem;
    private readonly Act _stemAct, _headAct;
    private readonly List<Block> _blocks;
    private readonly PackedLinearF32 _convHead;
    private readonly float[]? _headBnScale, _headBnShift; // EfficientNet: BN after conv_head (before pooling)
    private readonly bool _poolBeforeHead;                  // MobileNetV3: global pool first, then conv_head (+bias)
    private readonly PackedLinearF32 _classifier;

    public int InputSize { get; }
    public float CropPct { get; }
    public float[] Mean { get; }
    public float[] Std { get; }
    public bool Bicubic { get; }
    public int NumClasses => _classifier.OutDim;

    private static readonly Dictionary<string, (string[][] Arch, Act Act, bool Mnv3)> s_variants = new()
    {
        ["mobilenetv3_small_100"] = ([
            ["ds_r1_k3_s2_e1_c16_se0.25_nre"], ["ir_r1_k3_s2_e4.5_c24_nre", "ir_r1_k3_s1_e3.67_c24_nre"],
            ["ir_r1_k5_s2_e4_c40_se0.25", "ir_r2_k5_s1_e6_c40_se0.25"], ["ir_r2_k5_s1_e3_c48_se0.25"],
            ["ir_r3_k5_s2_e6_c96_se0.25"], ["cn_r1_k1_s1_c576"]], Act.HardSwish, true),
        ["mobilenetv3_large_100"] = ([
            ["ds_r1_k3_s1_e1_c16_nre"], ["ir_r1_k3_s2_e4_c24_nre", "ir_r1_k3_s1_e3_c24_nre"], ["ir_r3_k5_s2_e3_c40_se0.25_nre"],
            ["ir_r1_k3_s2_e6_c80", "ir_r1_k3_s1_e2.5_c80", "ir_r2_k3_s1_e2.3_c80"], ["ir_r2_k3_s1_e6_c112_se0.25"],
            ["ir_r3_k5_s2_e6_c160_se0.25"], ["cn_r1_k1_s1_c960"]], Act.HardSwish, true),
        ["efficientnet_b0"] = ([
            ["ds_r1_k3_s1_e1_c16_se0.25"], ["ir_r2_k3_s2_e6_c24_se0.25"], ["ir_r2_k5_s2_e6_c40_se0.25"],
            ["ir_r3_k3_s2_e6_c80_se0.25"], ["ir_r3_k5_s1_e6_c112_se0.25"], ["ir_r4_k5_s2_e6_c192_se0.25"],
            ["ir_r1_k3_s1_e6_c320_se0.25"]], Act.Swish, false),
    };

    private TimmEfficientNetClassifier(Conv stem, Act stemAct, List<Block> blocks, PackedLinearF32 convHead, float[]? bnScale, float[]? bnShift,
        bool poolFirst, Act headAct, PackedLinearF32 classifier, int inputSize, float cropPct, float[] mean, float[] std, bool bicubic)
    {
        (_stem, _stemAct, _blocks, _convHead, _headBnScale, _headBnShift, _poolBeforeHead, _headAct, _classifier) =
            (stem, stemAct, blocks, convHead, bnScale, bnShift, poolFirst, headAct, classifier);
        (InputSize, CropPct, Mean, Std, Bicubic) = (inputSize, cropPct, mean, std, bicubic);
    }

    private static Spec Parse(string s)
    {
        var parts = s.Split('_');
        int r = 1, k = 1, st = 1;
        bool se = false, relu = false;
        foreach (var p in parts.Skip(1))
        {
            if (p == "nre") relu = true;
            else if (p.StartsWith("se", StringComparison.Ordinal)) se = true;
            else if (p[0] == 'r') r = int.Parse(p[1..], CultureInfo.InvariantCulture);
            else if (p[0] == 'k') k = int.Parse(p[1..], CultureInfo.InvariantCulture);
            else if (p[0] == 's') st = int.Parse(p[1..], CultureInfo.InvariantCulture);
        }
        return new Spec(parts[0], r, k, st, se, relu);
    }

    public static TimmEfficientNetClassifier Load(string dir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "config.json")));
        var root = doc.RootElement;
        string arch = root.GetProperty("architecture").GetString() ?? "";
        if (!s_variants.TryGetValue(arch, out var v))
            throw new NotSupportedException($"timm architecture '{arch}' not supported ({string.Join(", ", s_variants.Keys)}).");
        var cfg = root.GetProperty("pretrained_cfg");
        float[] F(string n) => cfg.GetProperty(n).EnumerateArray().Select(x => x.GetSingle()).ToArray();

        using var st = SafetensorsLoader.OpenDirectory(dir);
        Conv ReadConv(string name, string bn, int stride, bool depthwise)
        {
            int[] shape = st.GetShape(name + ".weight");
            int outCh = shape[0], inCh = shape[1], k = shape[2];
            var w = st.ReadF32(name + ".weight");
            var b = new float[outCh];
            if (st.TensorNames.Contains(name + ".bias")) b = st.ReadF32(name + ".bias");
            if (bn.Length > 0)
            {
                float[] g = st.ReadF32(bn + ".weight"), beta = st.ReadF32(bn + ".bias"), m = st.ReadF32(bn + ".running_mean"), var = st.ReadF32(bn + ".running_var");
                int per = inCh * k * k;
                for (int o = 0; o < outCh; o++)
                {
                    float sc = g[o] / MathF.Sqrt(var[o] + 1e-5f);
                    for (int i = 0; i < per; i++) w[o * per + i] *= sc;
                    b[o] = (b[o] - m[o]) * sc + beta[o];
                }
            }
            if (depthwise)
            {
                if (inCh != 1) throw new InvalidDataException($"{name}: expected a depthwise [ch, 1, k, k] weight.");
                var tap = new float[k * k * outCh];
                for (int o = 0; o < outCh; o++)
                    for (int t = 0; t < k * k; t++) tap[t * outCh + o] = w[o * k * k + t];
                w = tap;
            }
            return new Conv(outCh, depthwise ? outCh : inCh, k, stride, depthwise, w, b);
        }
        PackedLinearF32 Fc(string name)
        {
            int[] s = st.GetShape(name + ".weight");
            return new PackedLinearF32(st.ReadF32(name + ".weight"), st.ReadF32(name + ".bias"), s[0], s.Skip(1).Aggregate(1, (a, x) => a * x));
        }

        var blocks = new List<Block>();
        for (int si = 0; si < v.Arch.Length; si++)
        {
            int bi = 0;
            foreach (var spec in v.Arch[si].Select(Parse))
                for (int rep = 0; rep < spec.Repeats; rep++, bi++)
                {
                    string p = $"blocks.{si}.{bi}";
                    int stride = rep == 0 ? spec.Stride : 1;
                    Act act = spec.Relu ? Act.Relu : v.Act;
                    Se? se = spec.Se ? new Se(Fc(p + ".se.conv_reduce"), Fc(p + ".se.conv_expand"), v.Mnv3, v.Mnv3 ? Act.Relu : act) : null;
                    Block block = spec.Type switch
                    {
                        "ds" => new Block("ds", null, ReadConv(p + ".conv_dw", p + ".bn1", stride, true), se, ReadConv(p + ".conv_pw", p + ".bn2", 1, false), act, false),
                        "ir" => new Block("ir", ReadConv(p + ".conv_pw", p + ".bn1", 1, false), ReadConv(p + ".conv_dw", p + ".bn2", stride, true), se,
                            ReadConv(p + ".conv_pwl", p + ".bn3", 1, false), act, false),
                        "cn" => new Block("cn", null, ReadConv(p + ".conv", p + ".bn1", stride, false), null, null, act, false),
                        _ => throw new NotSupportedException($"block type '{spec.Type}'"),
                    };
                    int inCh = block.Type == "ir" ? block.Pw!.InCh : block.Dw.InCh;
                    int outCh = block.Type == "cn" ? block.Dw.OutCh : block.Pwl!.OutCh;
                    // has_skip = stride 1 and in == out (DS, IR); ConvBnAct (cn) has skip disabled by default.
                    blocks.Add(block with { Skip = block.Type != "cn" && stride == 1 && inCh == outCh });
                }
        }

        var stem = ReadConv("conv_stem", "bn1", 2, false);
        PackedLinearF32 head;
        float[]? bnScale = null, bnShift = null;
        if (v.Mnv3)
            head = Fc("conv_head");
        else
        {
            var hc = ReadConv("conv_head", "bn2", 1, false); // 1x1 conv with bn2 folded in: its packed weight is the head GEMM
            head = hc.Packed!;
            (bnScale, bnShift) = (Enumerable.Repeat(1f, hc.OutCh).ToArray(), hc.B);
        }
        return new TimmEfficientNetClassifier(stem, v.Act, blocks, head, bnScale, bnShift, v.Mnv3, v.Act, Fc("classifier"),
            (int)F("input_size")[1], cfg.GetProperty("crop_pct").GetSingle(), F("mean"), F("std"),
            (cfg.GetProperty("interpolation").GetString() ?? "bicubic") == "bicubic");
    }

    private static void Activate(Span<float> x, Act act)
    {
        switch (act)
        {
            case Act.Relu: TensorPrimitives.Max(x, 0f, x); break;
            case Act.HardSwish:
                for (int i = 0; i < x.Length; i++) x[i] *= Math.Clamp(x[i] + 3f, 0f, 6f) / 6f;
                break;
            case Act.Swish:
                for (int i = 0; i < x.Length; i++) x[i] /= 1f + MathF.Exp(-x[i]);
                break;
        }
    }

    private static int OutSize(int n, int k, int s) => (n + 2 * ((s - 1 + k - 1) / 2) - k) / s + 1;

    /// <summary>Dense conv over HWC via im2col ([ci, ky, kx] order, matching torch's [out, in, kh, kw] weights).</summary>
    private static float[] DenseConv(float[] x, int h, int w, Conv c, out int oh, out int ow)
    {
        int pad = (c.Stride - 1 + c.K - 1) / 2, k = c.K;
        oh = OutSize(h, k, c.Stride);
        ow = OutSize(w, k, c.Stride);
        int rows = oh * ow, cols = c.InCh * k * k;
        float[] input = x;
        if (k != 1 || c.Stride != 1)
        {
            input = new float[rows * cols];
            int ohh = oh, oww = ow;
            Parallel.For(0, ohh, oy =>
            {
                for (int ox = 0; ox < oww; ox++)
                {
                    var row = input.AsSpan((oy * oww + ox) * cols, cols);
                    for (int ci = 0; ci < c.InCh; ci++)
                        for (int ky = 0; ky < k; ky++)
                            for (int kx = 0; kx < k; kx++)
                            {
                                int iy = oy * c.Stride + ky - pad, ix = ox * c.Stride + kx - pad;
                                row[(ci * k + ky) * k + kx] = iy < 0 || iy >= h || ix < 0 || ix >= w ? 0f : x[(iy * w + ix) * c.InCh + ci];
                            }
                }
            });
        }
        var y = new float[rows * c.OutCh];
        c.Packed!.Forward(input, y, rows);
        for (int r = 0; r < rows; r++) TensorPrimitives.Add(y.AsSpan(r * c.OutCh, c.OutCh), c.B, y.AsSpan(r * c.OutCh, c.OutCh));
        return y;
    }

    private static float[] DepthwiseConv(float[] x, int h, int w, Conv c, out int oh, out int ow)
    {
        int pad = (c.Stride - 1 + c.K - 1) / 2, k = c.K, ch = c.OutCh;
        oh = OutSize(h, k, c.Stride);
        ow = OutSize(w, k, c.Stride);
        var y = new float[oh * ow * ch];
        int oww = ow;
        Parallel.For(0, oh, oy =>
        {
            for (int ox = 0; ox < oww; ox++)
            {
                var o = y.AsSpan((oy * oww + ox) * ch, ch);
                c.B.CopyTo(o);
                for (int ky = 0; ky < k; ky++)
                {
                    int iy = oy * c.Stride + ky - pad;
                    if (iy < 0 || iy >= h) continue;
                    for (int kx = 0; kx < k; kx++)
                    {
                        int ix = ox * c.Stride + kx - pad;
                        if (ix < 0 || ix >= w) continue;
                        TensorPrimitives.MultiplyAdd(x.AsSpan((iy * w + ix) * ch, ch), c.W.AsSpan((ky * k + kx) * ch, ch), o, o);
                    }
                }
            }
        });
        return y;
    }

    private static float[] MeanPool(float[] x, int pixels, int ch)
    {
        var m = new float[ch];
        for (int p = 0; p < pixels; p++) TensorPrimitives.Add(m, x.AsSpan(p * ch, ch), m);
        TensorPrimitives.Divide(m, pixels, m);
        return m;
    }

    private static void SqueezeExcite(float[] x, int pixels, int ch, Se se)
    {
        var pooled = MeanPool(x, pixels, ch);
        var r = new float[se.Reduce.OutDim];
        se.Reduce.Forward(pooled, r, 1);
        Activate(r, se.Act);
        var gate = new float[ch];
        se.Expand.Forward(r, gate, 1);
        for (int i = 0; i < ch; i++)
            gate[i] = se.HardSigmoid ? Math.Clamp(gate[i] + 3f, 0f, 6f) / 6f : 1f / (1f + MathF.Exp(-gate[i]));
        for (int p = 0; p < pixels; p++) TensorPrimitives.Multiply(x.AsSpan(p * ch, ch), gate, x.AsSpan(p * ch, ch));
    }

    /// <summary>ImageNet logits for a preprocessed planar CHW image [3, InputSize, InputSize].</summary>
    public float[] Logits(float[] chw, int height, int width)
    {
        int plane = height * width;
        var x = new float[plane * 3];
        for (int p = 0; p < plane; p++)
            for (int c = 0; c < 3; c++) x[p * 3 + c] = chw[c * plane + p];

        int h = height, w = width;
        x = DenseConv(x, h, w, _stem, out h, out w);
        Activate(x, _stemAct);
        foreach (var b in _blocks)
        {
            var shortcut = x;
            switch (b.Type)
            {
                case "ds":
                    x = DepthwiseConv(x, h, w, b.Dw, out h, out w);
                    Activate(x, b.Act);
                    if (b.Se is not null) SqueezeExcite(x, h * w, b.Dw.OutCh, b.Se);
                    x = DenseConv(x, h, w, b.Pwl!, out h, out w);  // bn2: no activation (pw_act false)
                    break;
                case "ir":
                    x = DenseConv(x, h, w, b.Pw!, out h, out w);
                    Activate(x, b.Act);
                    x = DepthwiseConv(x, h, w, b.Dw, out h, out w);
                    Activate(x, b.Act);
                    if (b.Se is not null) SqueezeExcite(x, h * w, b.Dw.OutCh, b.Se);
                    x = DenseConv(x, h, w, b.Pwl!, out h, out w); // bn3: no activation
                    break;
                default: // cn: ConvBnAct
                    x = DenseConv(x, h, w, b.Dw, out h, out w);
                    Activate(x, b.Act);
                    break;
            }
            if (b.Skip) TensorPrimitives.Add(x, shortcut, x);
        }

        int ch = x.Length / (h * w);
        float[] feat;
        if (_poolBeforeHead)
        {
            var pooled = MeanPool(x, h * w, ch);
            feat = new float[_convHead.OutDim];
            _convHead.Forward(pooled, feat, 1);
            Activate(feat, _headAct);
        }
        else
        {
            var hx = new float[h * w * _convHead.OutDim];
            _convHead.Forward(x, hx, h * w);
            for (int p = 0; p < h * w; p++) TensorPrimitives.Add(hx.AsSpan(p * _convHead.OutDim, _convHead.OutDim), _headBnShift!, hx.AsSpan(p * _convHead.OutDim, _convHead.OutDim));
            Activate(hx, _headAct);
            feat = MeanPool(hx, h * w, _convHead.OutDim);
        }
        var logits = new float[NumClasses];
        _classifier.Forward(feat, logits, 1);
        return logits;
    }

    public float[] Classify(ReadOnlySpan<byte> rgb, int width, int height) =>
        Logits(ImageClassificationPreprocessor.ResizeCenterCropNormalize(rgb, width, height, InputSize, CropPct, Mean, Std, Bicubic), InputSize, InputSize);

    public void Dispose()
    {
        _stem.Dispose();
        foreach (var b in _blocks)
        {
            b.Pw?.Dispose();
            b.Dw.Dispose();
            b.Pwl?.Dispose();
            b.Se?.Reduce.Dispose();
            b.Se?.Expand.Dispose();
        }
        _convHead.Dispose();
        _classifier.Dispose();
    }
}
