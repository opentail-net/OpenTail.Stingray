# Guide: ask about a picture, or read a document

**You get:** a model that looks at an image and answers a question about it: describe a photo,
read a screenshot, pull the text out of a scanned page.

**You need:** two files, not one: the model (`.gguf`) and its **image projector** (`mmproj-*.gguf`).
They must belong together. Download both from the same Hugging Face repo.

## Run it

```bash
stingray -m models/_models/Qwen3VL-2B-Instruct-Q8_0.gguf \
         --mmproj models/_models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
         --image photo.png -p "Describe this picture."
```

- `--image` takes a PNG. Repeat it for several images and mark where each goes with an `<image>`
  marker in the prompt, or leave the markers out and the images are put first.
- `--mmproj` is required whenever `--image` is used.

## Which model for which job

| You want to | Try | Notes |
|---|---|---|
| General questions about photos | Gemma 3 4B, Qwen3-VL 2B, InternVL3 2B | Qwen3-VL 2B decodes at about 20 tok/s on the reference CPU |
| Read text in a document or screenshot (OCR) | dots.ocr 1.5B, PaddleOCR-VL 1.6, DeepSeek-OCR2 | Ask "Read the text in this image." |
| Detailed description, bigger model | Qwen2.5-VL 7B, LLaVA-1.5 7B | Slower: about 7 tok/s |
| Small footprint | Granite Vision 3.2 2B, Youtu-VL 4B | |

Each model's exact command, memory and speed is in [RUNNING.md](../RUNNING.md) (the vision table);
how well each was checked is in [STATUS.md](../STATUS.md) (the "Vision:" rows).

## Know the limits

- An image costs time. Every image is turned into hundreds of tokens (for example 576 for LLaVA-1.5
  and about 730 for Granite Vision) that the model must read first.
- **Qwen-VL-style models and Granite 4.0 Vision run on the CPU,** and Qwen3-VL also on a fully
  offloaded Vulkan run. On other GPU setups the CLI refuses the image instead of giving a wrong
  answer.
- **The HTTP server's image support is narrower than the CLI's.** The CLI is the reliable route
  today.
- MiMo-VL is rated partial in STATUS.md, and Llama 4 vision is not supported.

## From C#

Vision is in `OpenTail.Stingray.Vision` (`UnifiedVisionPipeline`). The CLI source
(`src/OpenTail.Stingray.Cli`) is the working reference for wiring a model, a projector and an image
together.
