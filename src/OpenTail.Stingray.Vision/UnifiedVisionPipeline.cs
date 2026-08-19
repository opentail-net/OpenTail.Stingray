using System;
using System.IO;
using System.Linq;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Vision;

/// <summary>
/// Unified factory and dispatcher for multimodal vision models.
/// Supports: Nemotron-V2-VL, Dots-OCR, DeepSeek-OCR, Kimi-VL, GLM-4V, LLaVA, InternVL,
/// Pixtral, MiniCPM-V, Qwen 2.5/3 VL, Gemma 4 UV, Gemma 4 ViT, Gemma 3, Llama 4,
/// HunyuanVL, Step3VL, YoutuVL, EXAONE 4.5, MiMo-VL.
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

        if (projType != null && (projType.Contains("nemotron") || projType.Contains("nemotron_v2_vl") || projType.Contains("nemotron-v2-vl")))
        {
            var model = NemotronVisionModel.Open(mmprojPath);
            return new NemotronAdapter(model);
        }

        if (projType != null && (projType.Contains("paddleocr") || projType.Contains("paddle_ocr")))
        {
            var model = PaddleOcrVisionModel.Open(mmprojPath);
            return new PaddleOcrAdapter(model);
        }

        if (projType != null && (projType.Contains("dotsocr") || projType.Contains("dots_ocr")))
        {
            var model = DotsOcrVisionModel.Open(mmprojPath);
            return new DotsOcrAdapter(model);
        }

        if (projType != null && (projType.Contains("cogvlm") || projType.Contains("cogagent")))
        {
            var model = CogVlmVisionModel.Open(mmprojPath);
            return new CogVlmAdapter(model);
        }

        if (projType != null && (projType.Contains("granite4") || projType.Contains("granite-vision") || projType.Contains("granite4-vision")))
        {
            var model = Granite4VisionModel.Open(mmprojPath);
            return new Granite4Adapter(model);
        }

        if (projType != null && (projType.Contains("mobilenetv5") || projType.Contains("mobilenet_v5")))
        {
            var model = MobileNetV5VisionModel.Open(mmprojPath);
            return new MobileNetV5Adapter(model);
        }

        if (projType != null && (projType.Contains("deepseekocr") || projType.Contains("deepseek-ocr") || projType.Contains("deepseek_ocr")))
        {
            var model = DeepSeekOcrVisionModel.Open(mmprojPath);
            return new DeepSeekOcrAdapter(model);
        }

        if (projType != null && (projType.Contains("kimi") || projType.Contains("kimik25") || projType.Contains("kimivl")))
        {
            var model = KimiVisionModel.Open(mmprojPath);
            return new KimiAdapter(model);
        }

        if (projType != null && (projType.Contains("glm4v") || projType.Contains("glm-4v") || projType.Contains("glm_edge")))
        {
            var model = Glm4VisionModel.Open(mmprojPath);
            return new Glm4Adapter(model);
        }

        if (projType != null && (projType == "llava" || projType.Contains("llava-onevision") || projType.Contains("llava-next") || projType == "mlp"))
        {
            var model = LlavaVisionModel.Open(mmprojPath);
            return new LlavaAdapter(model);
        }

        if (projType != null && projType.Contains("internvl"))
        {
            var model = InternVlVisionModel.Open(mmprojPath);
            return new InternVlAdapter(model);
        }

        if (projType != null && projType.Contains("pixtral"))
        {
            var model = PixtralVisionModel.Open(mmprojPath);
            return new PixtralAdapter(model);
        }

        if (projType != null && (projType.Contains("minicpm") || projType.Contains("resampler")))
        {
            var model = MiniCpmVisionModel.Open(mmprojPath);
            return new MiniCpmAdapter(model);
        }

        if (projType != null && (projType.Contains("qwen2.5vl") || projType.Contains("qwen2vl") || projType.Contains("qwen3vl") || projType.Contains("qwenvl")))
        {
            var model = QwenVlVisionModel.Open(mmprojPath);
            return new QwenVlAdapter(model);
        }

        if (projType != null && (projType.Contains("hunyuanvl") || projType.Contains("hunyuan_vl")))
        {
            var model = HunyuanVlVisionModel.Open(mmprojPath);
            return new HunyuanVlAdapter(model);
        }

        if (projType != null && (projType.Contains("step3vl") || projType.Contains("step3")))
        {
            var model = Step3VlVisionModel.Open(mmprojPath);
            return new Step3VlAdapter(model);
        }

        if (projType != null && (projType.Contains("youtuvl") || projType.Contains("youtu_vl")))
        {
            var model = YoutuVlVisionModel.Open(mmprojPath);
            return new YoutuVlAdapter(model);
        }

        if (projType != null && (projType.Contains("exaone4") || projType.Contains("exaone_4")))
        {
            var model = Exaone4VisionModel.Open(mmprojPath);
            return new Exaone4Adapter(model);
        }

        if (projType != null && (projType.Contains("mimovl") || projType.Contains("mimo_vl") || projType.Contains("mimo")))
        {
            var model = MimoVlVisionModel.Open(mmprojPath);
            return new MimoVlAdapter(model);
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

        // Structural inference from tensor topology when projector_type metadata is absent:
        if (gguf.Tensors.Any(t => t.Name.Contains("v.registers") || t.Name.Contains("mm.reg_norm.weight")))
        {
            var model = NemotronVisionModel.Open(mmprojPath);
            return new NemotronAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name.Contains("model.view_seperator") || t.Name.Contains("resample_query_1024")))
        {
            var model = DeepSeekOcrVisionModel.Open(mmprojPath);
            return new DeepSeekOcrAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name.Contains("mm.input_norm.weight") && t.Name.Contains("mm.2.weight")))
        {
            var model = DotsOcrVisionModel.Open(mmprojPath);
            return new DotsOcrAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name.Contains("mm.input_norm.weight") && t.Name.Contains("mm.1.weight")))
        {
            var model = KimiVisionModel.Open(mmprojPath);
            return new KimiAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name.Contains("mm.patch_merger.weight") && t.Name.Contains("mm.fc.weight")))
        {
            var model = Glm4VisionModel.Open(mmprojPath);
            return new Glm4Adapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name.Contains("v.class_embd") || t.Name.Contains("v.cls_embd")))
        {
            var model = InternVlVisionModel.Open(mmprojPath);
            return new InternVlAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name.Contains("v.token_embd.img_break") || t.Name.Contains("mm.patch_merger.weight")))
        {
            var model = PixtralVisionModel.Open(mmprojPath);
            return new PixtralAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name.Contains("resampler.query") || t.Name.Contains("mm.model.query")))
        {
            var model = MiniCpmVisionModel.Open(mmprojPath);
            return new MiniCpmAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name.StartsWith("v.patch_embd.0.") || t.Name.StartsWith("v.patch_embd.1.")))
        {
            var model = QwenVlVisionModel.Open(mmprojPath);
            return new QwenVlAdapter(model);
        }

        if (gguf.Tensors.Any(t => t.Name == "mm.pre_norm.weight" && t.Name == "mm.model_proj.weight"))
        {
            var model = HunyuanVlVisionModel.Open(mmprojPath);
            return new HunyuanVlAdapter(model);
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

        if (gguf.Tensors.Any(t => t.Name == "mm.0.weight" && t.Name != "mm.0.bias"))
        {
            var model = LlavaVisionModel.Open(mmprojPath);
            return new LlavaAdapter(model);
        }

        throw new NotSupportedException(
            $"Unsupported or unrecognized vision projector type '{projType ?? "unknown"}' in {Path.GetFileName(mmprojPath)}. " +
            "Supported types: nemotron_v2_vl, dotsocr, deepseekocr, kimik25, glm4v, llava, internvl, pixtral, minicpmv, " +
            "qwen2.5vl, gemma4uv, gemma4v, gemma3, llama4, hunyuanvl, step3vl, youtuvl, exaone4, mimovl.");
    }

    private sealed class NemotronAdapter : IVisionEmbedder
    {
        private readonly NemotronVisionModel _model;
        private readonly NemotronVisionEncoder _encoder;

        public NemotronAdapter(NemotronVisionModel model)
        {
            _model = model;
            _encoder = new NemotronVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image_pad>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = NemotronImagePreprocessor.Preprocess(rgb, width, height, _model.ImageSize, _model.PatchSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class DotsOcrAdapter : IVisionEmbedder
    {
        private readonly DotsOcrVisionModel _model;
        private readonly DotsOcrVisionEncoder _encoder;

        public DotsOcrAdapter(DotsOcrVisionModel model)
        {
            _model = model;
            _encoder = new DotsOcrVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image_pad>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = DotsOcrImagePreprocessor.Preprocess(rgb, width, height, _model.PatchSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class DeepSeekOcrAdapter : IVisionEmbedder
    {
        private readonly DeepSeekOcrVisionModel _model;
        private readonly DeepSeekOcrVisionEncoder _encoder;

        public DeepSeekOcrAdapter(DeepSeekOcrVisionModel model)
        {
            _model = model;
            _encoder = new DeepSeekOcrVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image_pad>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = DeepSeekOcrImagePreprocessor.Preprocess(rgb, width, height, _model.ImageSize, _model.PatchSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class KimiAdapter : IVisionEmbedder
    {
        private readonly KimiVisionModel _model;
        private readonly KimiVisionEncoder _encoder;

        public KimiAdapter(KimiVisionModel model)
        {
            _model = model;
            _encoder = new KimiVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<|vision_start|>";
        public string ImageCloseMarker => "<|vision_end|>";
        public string PlaceholderMarker => "<|image_pad|>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = KimiImagePreprocessor.Preprocess(rgb, width, height, _model.PatchSize, _model.MergeFactor);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class Glm4Adapter : IVisionEmbedder
    {
        private readonly Glm4VisionModel _model;
        private readonly Glm4VisionEncoder _encoder;

        public Glm4Adapter(Glm4VisionModel model)
        {
            _model = model;
            _encoder = new Glm4VisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<|begin_of_image|>";
        public string ImageCloseMarker => "<|end_of_image|>";
        public string PlaceholderMarker => "<|image|>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = Glm4ImagePreprocessor.Preprocess(rgb, width, height, _model.PatchSize, _model.MergeFactor);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class LlavaAdapter : IVisionEmbedder
    {
        private readonly LlavaVisionModel _model;
        private readonly LlavaVisionEncoder _encoder;

        public LlavaAdapter(LlavaVisionModel model)
        {
            _model = model;
            _encoder = new LlavaVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = LlavaImagePreprocessor.Preprocess(rgb, width, height, _model.ImageSize, _model.PatchSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class InternVlAdapter : IVisionEmbedder
    {
        private readonly InternVlVisionModel _model;
        private readonly InternVlVisionEncoder _encoder;

        public InternVlAdapter(InternVlVisionModel model)
        {
            _model = model;
            _encoder = new InternVlVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<img>";
        public string ImageCloseMarker => "</img>";
        public string PlaceholderMarker => "<IMG_CONTEXT>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = InternVlImagePreprocessor.Preprocess(rgb, width, height, _model.ImageSize, _model.PatchSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class PixtralAdapter : IVisionEmbedder
    {
        private readonly PixtralVisionModel _model;
        private readonly PixtralVisionEncoder _encoder;

        public PixtralAdapter(PixtralVisionModel model)
        {
            _model = model;
            _encoder = new PixtralVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "[IMG]";
        public string ImageCloseMarker => "[/IMG]";
        public string PlaceholderMarker => "[IMG_PAD]";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = PixtralImagePreprocessor.Preprocess(rgb, width, height, _model.PatchSize, _model.ImageSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class MiniCpmAdapter : IVisionEmbedder
    {
        private readonly MiniCpmVisionModel _model;
        private readonly MiniCpmVisionEncoder _encoder;

        public MiniCpmAdapter(MiniCpmVisionModel model)
        {
            _model = model;
            _encoder = new MiniCpmVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<unk>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var slices = MiniCpmImagePreprocessor.Preprocess(rgb, width, height, _model.ImageSize);
            return _encoder.Forward(slices, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
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

    private sealed class HunyuanVlAdapter : IVisionEmbedder
    {
        private readonly HunyuanVlVisionModel _model;
        private readonly HunyuanVlVisionEncoder _encoder;

        public HunyuanVlAdapter(HunyuanVlVisionModel model)
        {
            _model = model;
            _encoder = new HunyuanVlVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image_pad>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = HunyuanVlImagePreprocessor.Preprocess(rgb, width, height, _model.ImageSize, _model.PatchSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class Step3VlAdapter : IVisionEmbedder
    {
        private readonly Step3VlVisionModel _model;
        private readonly Step3VlVisionEncoder _encoder;

        public Step3VlAdapter(Step3VlVisionModel model)
        {
            _model = model;
            _encoder = new Step3VlVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image_pad>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = Step3VlImagePreprocessor.Preprocess(rgb, width, height, _model.ImageSize, _model.PatchSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class YoutuVlAdapter : IVisionEmbedder
    {
        private readonly YoutuVlVisionModel _model;
        private readonly YoutuVlVisionEncoder _encoder;

        public YoutuVlAdapter(YoutuVlVisionModel model)
        {
            _model = model;
            _encoder = new YoutuVlVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image_pad>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = YoutuVlImagePreprocessor.Preprocess(rgb, width, height, _model.PatchSize, _model.SpatialMergeFactor, _model.ImageSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class Exaone4Adapter : IVisionEmbedder
    {
        private readonly Exaone4VisionModel _model;
        private readonly Exaone4VisionEncoder _encoder;

        public Exaone4Adapter(Exaone4VisionModel model)
        {
            _model = model;
            _encoder = new Exaone4VisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = Exaone4ImagePreprocessor.Preprocess(rgb, width, height);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
        }

        public float[] EmbedImageFile(string filePath, out int tokenCount)
        {
            var rgb = ImageIO.LoadRgb(filePath, out int w, out int h);
            return EmbedImage(rgb, w, h, out tokenCount);
        }

        public void Dispose() => _model.Dispose();
    }

    private sealed class MimoVlAdapter : IVisionEmbedder
    {
        private readonly MimoVlVisionModel _model;
        private readonly MimoVlVisionEncoder _encoder;

        public MimoVlAdapter(MimoVlVisionModel model)
        {
            _model = model;
            _encoder = new MimoVlVisionEncoder(model);
        }

        public string ProjectorType => _model.ProjectorType;
        public int EmbeddingDim => _model.ProjectionDim;
        public int ImageWidth => _model.ImageSize;
        public int ImageHeight => _model.ImageSize;
        public string ImageOpenMarker => "<image>";
        public string ImageCloseMarker => "</image>";
        public string PlaceholderMarker => "<image_pad>";

        public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, out int tokenCount)
        {
            var pre = MimoVlImagePreprocessor.Preprocess(rgb, width, height, _model.PatchSize, _model.NMerge, _model.ImageSize);
            return _encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out tokenCount);
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
