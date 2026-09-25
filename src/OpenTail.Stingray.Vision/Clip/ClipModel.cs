using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Vision.Classification;

namespace OpenTail.Stingray.Vision.Clip;

/// <summary>
/// Standalone OpenAI CLIP (HF <c>CLIPModel</c>, e.g. openai/clip-vit-base-patch32), CPU F32 from safetensors, configured
/// from <c>config.json</c>: text and image embeddings in the shared projection space plus zero-shot logits
/// (<c>exp(logit_scale) · cos</c>). Both towers are pre-LN transformers with QuickGELU (HF <c>modeling_clip.py</c>):
/// text = token + learned position embeddings, causal self-attention, final LayerNorm, pooled at the end-of-text token
/// (the highest id, i.e. HF's <c>argmax(input_ids)</c> path), <c>text_projection</c>; vision = 32×32 (patch_size)
/// patch conv + class token + learned positions, <c>pre_layrnorm</c>, encoder, <c>post_layernorm</c> on the class token,
/// <c>visual_projection</c>. Preprocessing: shorter side to image_size (bicubic), center crop, CLIP mean/std.
/// </summary>
public sealed class ClipModel : IDisposable
{
    private sealed class Layer : IDisposable
    {
        public required PackedLinearF32 Qkv, Out, Fc1, Fc2;
        public required float[] Ln1W, Ln1B, Ln2W, Ln2B;
        public void Dispose() { Qkv.Dispose(); Out.Dispose(); Fc1.Dispose(); Fc2.Dispose(); }
    }

    private sealed record Tower(Layer[] Layers, int Dim, int Heads, float Eps);

    private readonly Tower _text, _vision;
    private readonly float[] _tokEmb, _textPos, _finalLnW, _finalLnB;
    private readonly float[] _clsEmb, _visPos, _preLnW, _preLnB, _postLnW, _postLnB;
    private readonly PackedLinearF32 _patch, _textProj, _visProj;
    private readonly float _logitScale;

    public ClipBpeTokenizer Tokenizer { get; }
    public int ImageSize { get; }
    public int PatchSize { get; }
    public int ProjectionDim => _textProj.OutDim;
    public int MaxTextLength { get; }
    public float[] Mean { get; }
    public float[] Std { get; }

    private ClipModel(ClipBpeTokenizer tok, Tower text, Tower vision, float[] tokEmb, float[] textPos, float[] finalLnW, float[] finalLnB,
        float[] clsEmb, float[] visPos, float[] preLnW, float[] preLnB, float[] postLnW, float[] postLnB,
        PackedLinearF32 patch, PackedLinearF32 textProj, PackedLinearF32 visProj, float logitScale, int imageSize, int patchSize,
        float[] mean, float[] std)
    {
        (Tokenizer, _text, _vision, _tokEmb, _textPos, _finalLnW, _finalLnB) = (tok, text, vision, tokEmb, textPos, finalLnW, finalLnB);
        (_clsEmb, _visPos, _preLnW, _preLnB, _postLnW, _postLnB) = (clsEmb, visPos, preLnW, preLnB, postLnW, postLnB);
        (_patch, _textProj, _visProj, _logitScale, ImageSize, PatchSize, Mean, Std) = (patch, textProj, visProj, logitScale, imageSize, patchSize, mean, std);
        MaxTextLength = textPos.Length / text.Dim;
    }

    public static ClipModel Load(string dir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "config.json")));
        var root = doc.RootElement;
        var tc = root.GetProperty("text_config");
        var vc = root.GetProperty("vision_config");
        foreach (var c in new[] { tc, vc })
            if ((c.TryGetProperty("hidden_act", out var act) ? act.GetString() : "quick_gelu") != "quick_gelu")
                throw new NotSupportedException("CLIP: only quick_gelu towers are supported.");

        using var st = SafetensorsLoader.OpenDirectory(dir);
        Tower ReadTower(string prefix, JsonElement c)
        {
            int d = c.GetProperty("hidden_size").GetInt32(), inter = c.GetProperty("intermediate_size").GetInt32();
            int n = c.GetProperty("num_hidden_layers").GetInt32(), heads = c.GetProperty("num_attention_heads").GetInt32();
            float eps = c.TryGetProperty("layer_norm_eps", out var e) ? e.GetSingle() : 1e-5f;
            var layers = new Layer[n];
            for (int i = 0; i < n; i++)
            {
                string p = $"{prefix}.encoder.layers.{i}.";
                float[] R(string s) => st.ReadF32(p + s);
                layers[i] = new Layer
                {
                    Qkv = new PackedLinearF32([.. R("self_attn.q_proj.weight"), .. R("self_attn.k_proj.weight"), .. R("self_attn.v_proj.weight")],
                        [.. R("self_attn.q_proj.bias"), .. R("self_attn.k_proj.bias"), .. R("self_attn.v_proj.bias")], 3 * d, d),
                    Out = new PackedLinearF32(R("self_attn.out_proj.weight"), R("self_attn.out_proj.bias"), d, d),
                    Fc1 = new PackedLinearF32(R("mlp.fc1.weight"), R("mlp.fc1.bias"), inter, d),
                    Fc2 = new PackedLinearF32(R("mlp.fc2.weight"), R("mlp.fc2.bias"), d, inter),
                    Ln1W = R("layer_norm1.weight"), Ln1B = R("layer_norm1.bias"), Ln2W = R("layer_norm2.weight"), Ln2B = R("layer_norm2.bias"),
                };
            }
            return new Tower(layers, d, heads, eps);
        }

        var text = ReadTower("text_model", tc);
        var vision = ReadTower("vision_model", vc);
        int patchSize = vc.GetProperty("patch_size").GetInt32(), imageSize = vc.GetProperty("image_size").GetInt32();
        int proj = root.GetProperty("projection_dim").GetInt32();
        float[] mean = [0.48145466f, 0.4578275f, 0.40821073f], std = [0.26862954f, 0.26130258f, 0.27577711f];
        string pre = Path.Combine(dir, "preprocessor_config.json");
        if (File.Exists(pre))
        {
            using var pdoc = JsonDocument.Parse(File.ReadAllBytes(pre));
            if (pdoc.RootElement.TryGetProperty("image_mean", out var m)) mean = m.EnumerateArray().Select(x => x.GetSingle()).ToArray();
            if (pdoc.RootElement.TryGetProperty("image_std", out var s)) std = s.EnumerateArray().Select(x => x.GetSingle()).ToArray();
        }
        return new ClipModel(ClipBpeTokenizer.FromTokenizerJson(Path.Combine(dir, "tokenizer.json")), text, vision,
            st.ReadF32("text_model.embeddings.token_embedding.weight"), st.ReadF32("text_model.embeddings.position_embedding.weight"),
            st.ReadF32("text_model.final_layer_norm.weight"), st.ReadF32("text_model.final_layer_norm.bias"),
            st.ReadF32("vision_model.embeddings.class_embedding"), st.ReadF32("vision_model.embeddings.position_embedding.weight"),
            st.ReadF32("vision_model.pre_layrnorm.weight"), st.ReadF32("vision_model.pre_layrnorm.bias"),
            st.ReadF32("vision_model.post_layernorm.weight"), st.ReadF32("vision_model.post_layernorm.bias"),
            new PackedLinearF32(st.ReadF32("vision_model.embeddings.patch_embedding.weight"), null, vision.Dim, 3 * patchSize * patchSize),
            new PackedLinearF32(st.ReadF32("text_projection.weight"), null, proj, text.Dim),
            new PackedLinearF32(st.ReadF32("visual_projection.weight"), null, proj, vision.Dim),
            MathF.Exp(st.ReadF32("logit_scale")[0]), imageSize, patchSize, mean, std);
    }

    private static void LayerNorm(float[] x, int rows, int d, float[] w, float[] b, float eps, float[] y)
    {
        Parallel.For(0, rows, r =>
        {
            var xr = x.AsSpan(r * d, d);
            var yr = y.AsSpan(r * d, d);
            float mean = TensorPrimitives.Sum(xr) / d;
            TensorPrimitives.Subtract(xr, mean, yr);
            float inv = 1f / MathF.Sqrt(TensorPrimitives.SumOfSquares(yr) / d + eps);
            TensorPrimitives.Multiply(yr, inv, yr);
            TensorPrimitives.Multiply(yr, w, yr);
            TensorPrimitives.Add(yr, b, yr);
        });
    }

    /// <summary>Pre-LN encoder over [t, dim] in place; causal masks keys after the query (text tower).</summary>
    private static void Encode(Tower tw, float[] x, int t, bool causal)
    {
        int d = tw.Dim, heads = tw.Heads, dh = d / heads;
        float scale = 1f / MathF.Sqrt(dh);
        var n = new float[t * d];
        var qkv = new float[t * 3 * d];
        var ctx = new float[t * d];
        var o = new float[t * d];
        var hid = new float[t * tw.Layers[0].Fc1.OutDim];
        foreach (var l in tw.Layers)
        {
            LayerNorm(x, t, d, l.Ln1W, l.Ln1B, tw.Eps, n);
            l.Qkv.Forward(n, qkv, t);
            Array.Clear(ctx);
            Parallel.For(0, heads * t, hi =>
            {
                int h = hi / t, i = hi % t, off = h * dh, keys = causal ? i + 1 : t;
                var scores = new float[keys];
                var q = qkv.AsSpan(i * 3 * d + off, dh);
                for (int j = 0; j < keys; j++) scores[j] = TensorPrimitives.Dot(q, qkv.AsSpan(j * 3 * d + d + off, dh)) * scale;
                TensorPrimitives.Subtract(scores, TensorPrimitives.Max(scores), scores);
                TensorPrimitives.Exp(scores, scores);
                TensorPrimitives.Divide(scores, TensorPrimitives.Sum(scores), scores);
                var c = ctx.AsSpan(i * d + off, dh);
                for (int j = 0; j < keys; j++) TensorPrimitives.MultiplyAdd(qkv.AsSpan(j * 3 * d + 2 * d + off, dh), scores[j], c, c);
            });
            l.Out.Forward(ctx, o, t);
            TensorPrimitives.Add(x, o, x);

            LayerNorm(x, t, d, l.Ln2W, l.Ln2B, tw.Eps, n);
            l.Fc1.Forward(n, hid, t);
            for (int i = 0; i < hid.Length; i++) hid[i] *= 1f / (1f + MathF.Exp(-1.702f * hid[i])); // QuickGELU
            l.Fc2.Forward(hid, o, t);
            TensorPrimitives.Add(x, o, x);
        }
    }

    /// <summary>Projected (unnormalized) text embedding for token ids including start/end tokens.</summary>
    public float[] TextEmbedding(int[] ids)
    {
        int d = _text.Dim, t = ids.Length;
        if (t > MaxTextLength) throw new ArgumentException($"CLIP text longer than {MaxTextLength} tokens.");
        var x = new float[t * d];
        for (int i = 0; i < t; i++)
        {
            var row = x.AsSpan(i * d, d);
            TensorPrimitives.Add(_tokEmb.AsSpan(ids[i] * d, d), _textPos.AsSpan(i * d, d), row);
        }
        Encode(_text, x, t, causal: true);
        int eos = Array.IndexOf(ids, ids.Max()); // HF (eos_token_id == 2 legacy path): pool at argmax(input_ids)
        var pooled = new float[d];
        LayerNorm(x.AsSpan(eos * d, d).ToArray(), 1, d, _finalLnW, _finalLnB, _text.Eps, pooled);
        var e = new float[ProjectionDim];
        _textProj.Forward(pooled, e, 1);
        return e;
    }

    public float[] TextEmbedding(string text) => TextEmbedding(Tokenizer.Encode(text, MaxTextLength));

    /// <summary>Projected (unnormalized) image embedding for a preprocessed CHW image [3, ImageSize, ImageSize].</summary>
    public float[] ImageEmbedding(float[] chw)
    {
        int d = _vision.Dim, p = PatchSize, g = ImageSize / p, np = g * g, t = np + 1, plane = ImageSize * ImageSize;
        var cols = new float[np * 3 * p * p];
        for (int py = 0; py < g; py++)
            for (int px = 0; px < g; px++)
            {
                var row = cols.AsSpan((py * g + px) * 3 * p * p, 3 * p * p);
                for (int c = 0; c < 3; c++)
                    for (int ky = 0; ky < p; ky++)
                        for (int kx = 0; kx < p; kx++)
                            row[(c * p + ky) * p + kx] = chw[c * plane + (py * p + ky) * ImageSize + px * p + kx];
            }
        var x = new float[t * d];
        _clsEmb.CopyTo(x, 0);
        _patch.Forward(cols, x.AsSpan(d), np);
        TensorPrimitives.Add(x, _visPos.AsSpan(0, t * d), x);
        var y = new float[t * d];
        LayerNorm(x, t, d, _preLnW, _preLnB, _vision.Eps, y);
        Encode(_vision, y, t, causal: false);
        var cls = new float[d];
        LayerNorm(y.AsSpan(0, d).ToArray(), 1, d, _postLnW, _postLnB, _vision.Eps, cls);
        var e = new float[ProjectionDim];
        _visProj.Forward(cls, e, 1);
        return e;
    }

    public float[] Preprocess(ReadOnlySpan<byte> rgb, int width, int height) =>
        ImageClassificationPreprocessor.ResizeCenterCropNormalize(rgb, width, height, ImageSize, 1f, Mean, Std, bicubic: true);

    /// <summary>Zero-shot: <c>exp(logit_scale) · cos(image, text_i)</c> for each label text.</summary>
    public float[] ZeroShotLogits(float[] imageEmbedding, IReadOnlyList<float[]> textEmbeddings) =>
        textEmbeddings.Select(te => _logitScale * TensorPrimitives.CosineSimilarity(imageEmbedding, te)).ToArray();

    public void Dispose()
    {
        foreach (var l in _text.Layers) l.Dispose();
        foreach (var l in _vision.Layers) l.Dispose();
        _patch.Dispose();
        _textProj.Dispose();
        _visProj.Dispose();
    }
}
