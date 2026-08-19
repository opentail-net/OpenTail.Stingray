using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Unified factory and dispatcher for multimodal vision models (Qwen 2.5 VL, Gemma 4 UV, Gemma 4 ViT, Gemma 3, Llama 4).
/// Automatically determines the projector type from the GGUF file and provides an <see cref="IVisionEmbedder"/>.
/// </summary>
public static class UnifiedVisionPipeline
{
    public static IVisionEmbedder Open(string mmprojPath)
    {
        if (string.IsNullOrWhiteSpace(mmprojPath))
            throw new ArgumentException("mmprojPath must not be null or empty.", nameof(mmprojPath));

        if (!File.Exists(mmprojPath))
            throw new FileNotFoundException($"Vision model file not found: {mmprojPath}", mmprojPath);

        // Inspect metadata to determine projector type
        using var gguf = GgufModel.Open(mmprojPath);

        string? projType = null;
        if (gguf.Metadata.TryGetValue("clip.vision.projector_type", out var ptObj) && ptObj is string ptStr)
            projType = ptStr.Trim().ToLowerInvariant();
        else if (gguf.Metadata.TryGetValue("clip.projector_type", out var ptObj2) && ptObj2 is string ptStr2)
            projType = ptStr2.Trim().ToLowerInvariant();

        if (projType != null && (projType.Contains("qwen2.5vl") || projType.Contains("qwen2vl") || projType.Contains("qwen3vl") || projType.Contains("qwenvl")))
        {
            var model = QwenVlVisionModel.Open(mmprojPath);
            return new QwenVlAdapter(model);
        }

        // Check if gemma4uv / encoder-free
        if (projType == "gemma4uv" || (projType == null && gguf.Tensors.Any(t => t.Name == "v.patch_embd.weight") && !gguf.Tensors.Any(t => t.Name.StartsWith("v.blk.0.") || t.Name.StartsWith("v.blocks.0."))))
        {
            var model = VisionModel.Open(mmprojPath);
            return new GemmaUvAdapter(model);
        }

        if (projType == "gemma4v")
        {
            var model = Gemma4VVisionModel.Open(mmprojPath);
            return new Gemma4VAdapter(model);
        }

        if (projType == "gemma3" || projType == "gemma3nv")
        {
            var model = Gemma3VisionModel.Open(mmprojPath);
            return new Gemma3Adapter(model);
        }

        if (projType == "llama4")
        {
            var model = Llama4VisionModel.Open(mmprojPath);
            return new Llama4Adapter(model);
        }

        // Fallback heuristics based on tensors
        if (gguf.Tensors.Any(t => t.Name.StartsWith("v.patch_embd.0.") || t.Name.StartsWith("v.patch_embd.1.")))
        {
            var model = QwenVlVisionModel.Open(mmprojPath);
            return new QwenVlAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name == "mm.soft_emb_norm.weight" || t.Name == "v.post_ln.weight"))
        {
            var model = Gemma3VisionModel.Open(mmprojPath);
            return new Gemma3Adapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name == "v.blocks.0.attn_q.input_min"))
        {
            var model = Gemma4VVisionModel.Open(mmprojPath);
            return new Gemma4VAdapter(model);
        }

        throw new NotSupportedException(
            $"Unsupported or unrecognized vision projector type '{projType ?? "unknown"}' in {Path.GetFileName(mmprojPath)}. " +
            "Supported types: qwen2.5vl, gemma4uv, gemma4v, gemma3, llama4.");
    }

    private sealed class QwenVlAdapter : IVisionEmbedder
    {
        private readonly QwenVlVisionModel _model;
        private readonly QwenVlVisionEncoder _encoder;

        public QwenVlAdapter(QwenVlVisionModel model)
        {
            _model = model;
            _encoder = new QwenVlVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => 448;
        public int ImageHeight => 448;
        public string ImageOpenMarker => "<|vision_start|>";
        public string ImageCloseMarker => "<|vision_end|>";
        public string PlaceholderMarker => "<|image_pad|>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = QwenVlImagePreprocessor.Preprocess(rgb, width, height, _model.PatchSize, _model.SpatialMergeFactor);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class GemmaUvAdapter : IVisionEmbedder
    {
        private readonly VisionModel _model;
        private readonly GemmaUvVisionEmbedder _embedder;

        public GemmaUvAdapter(VisionModel model)
        {
            _model = model;
            _embedder = new GemmaUvVisionEmbedder(model);
        }

        public string ProjectorType => "gemma4uv";
        public int EmbeddingDim => _model.EmbeddingLength;
        public int ImageWidth => 480;
        public int ImageHeight => 480;
        public string ImageOpenMarker => "<|image|>";
        public string ImageCloseMarker => "";
        public string PlaceholderMarker => "<|image|>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = ImagePreprocessor.Preprocess(rgb.ToArray(), width, height, _model);
            return _embedder.Forward(pre.Chw, pre.Height, pre.Width, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class Gemma4VAdapter : IVisionEmbedder
    {
        private readonly Gemma4VVisionModel _model;
        private readonly Gemma4VVisionEncoder _encoder;

        public Gemma4VAdapter(Gemma4VVisionModel model)
        {
            _model = model;
            _encoder = new Gemma4VVisionEncoder(model);
        }

        public string ProjectorType => "gemma4v";
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<|image|>";
        public string ImageCloseMarker => "";
        public string PlaceholderMarker => "<|image|>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var img = Gemma4VImagePreprocessor.Preprocess(rgb.ToArray(), width, height, _model);
            tokenCount = _encoder.TokenCount;
            return _encoder.Forward(img.Chw);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class Gemma3Adapter : IVisionEmbedder
    {
        private readonly Gemma3VisionModel _model;
        private readonly Gemma3VisionEncoder _encoder;

        public Gemma3Adapter(Gemma3VisionModel model)
        {
            _model = model;
            _encoder = new Gemma3VisionEncoder(model);
        }

        public string ProjectorType => "gemma3";
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<|image|>";
        public string ImageCloseMarker => "";
        public string PlaceholderMarker => "<|image|>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var img = Gemma3ImagePreprocessor.Preprocess(rgb.ToArray(), width, height, _model);
            tokenCount = _encoder.TokenCount;
            return _encoder.Forward(img.Chw);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class Llama4Adapter : IVisionEmbedder
    {
        private readonly Llama4VisionModel _model;
        private readonly Llama4VisionEncoder _encoder;

        public Llama4Adapter(Llama4VisionModel model)
        {
            _model = model;
            _encoder = new Llama4VisionEncoder(model);
        }

        public string ProjectorType => "llama4";
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<|image_start|>";
        public string ImageCloseMarker => "<|image_end|>";
        public string PlaceholderMarker => "<|image|>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var img = Llama4ImagePreprocessor.Preprocess(rgb.ToArray(), width, height, _model);
            tokenCount = _encoder.TokenCount;
            return _encoder.Forward(img.Chw);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }
}
