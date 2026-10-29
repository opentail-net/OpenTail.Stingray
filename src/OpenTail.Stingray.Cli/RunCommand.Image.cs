using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Cli;

public sealed partial class RunCommand
{
    /// <summary>User-facing prompt marker for an image position (mapped to the model's
    /// <c>&lt;|image|&gt;</c> placeholder before templating). One per <c>--image</c>, left-to-right.</summary>
    private const string ImageMarker = "<image>";

    /// <summary>
    /// Single-prompt image→text for Gemma 4 (issue #250), one or more images. Each image is
    /// preprocessed and run through the encoder-free projector to soft tokens, then spliced
    /// into the decoder via <see cref="ForwardPass.ForwardEmbedding"/>, wrapped in the runtime
    /// markers (<c>&lt;|image&gt;</c> … soft tokens … <c>&lt;image|&gt;</c>).
    ///
    /// Placement: each <c>&lt;image&gt;</c> marker in the prompt is mapped to the model's
    /// <c>&lt;|image|&gt;</c> placeholder (id 258880), the prompt is rendered through the model's
    /// own chat template (so BOS / <c>&lt;|turn&gt;</c> / thinking handling matches the text path —
    /// Gemma 4 uses <c>&lt;|turn&gt;role\n…&lt;turn|&gt;</c>, NOT Gemma 3's <c>&lt;start_of_turn&gt;</c>),
    /// then each placeholder token in the token stream is expanded with its image's soft tokens
    /// in order. With no markers, the images are prepended to the user turn. The
    /// embedding-injection seam (<see cref="IForwardPass.ForwardEmbedding"/>) is implemented by
    /// <see cref="ForwardPass"/> (CPU), <see cref="CudaForwardPass"/> (full CUDA offload), and
    /// <see cref="CudaHybridForwardPass"/> (CUDA partial offload, issue #252).
    /// </summary>
    private static int RunImagePrompt(Settings s,
        IForwardPass fwd, GgufTokenizer tok, ModelHyperparams hp,
        SamplingParams sp, Random rng, int maxContextLength)
    {
        if (!fwd.SupportsEmbeddingInput)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] the selected backend does not support image embedding input. " +
                "Image input runs on CPU ([yellow]-g 0[/]), full CUDA offload ([yellow]-g -1[/]), or CUDA " +
                "partial-offload ([yellow]-g N[/]); the Vulkan partial-offload hybrid is not supported yet.");
            return 1;
        }

        // M-RoPE image positions (Qwen2-VL family) are applied by the CPU ForwardPass and the full Vulkan
        // GpuForwardPass only; anything else would rotate image tokens with 1D positions and answer wrongly.
        if (hp.RopeSections is { Count: > 0 } && fwd is not (ForwardPass or GpuForwardPass))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] image input for this model (M-RoPE) runs on CPU ([yellow]-g 0[/]) " +
                "or full Vulkan offload ([yellow]-g -1 --backend vulkan[/]) only.");
            return 1;
        }

        var imagePaths = s.ImagePaths!;
        int nImages = imagePaths.Length;

        IVisionEmbedder vision;
        try
        {
            vision = UnifiedVisionPipeline.Open(s.MmprojPath!);
        }
        catch (NotSupportedException ex)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
        using var __ = vision; // matches this method's existing risk tolerance: other early
                                // returns below (e.g. the EmbedImageFile catch) also don't dispose
                                // vision explicitly; process exit reclaims it for a CLI invocation.
        // Real bug (found 2026-09-11, benchmarking MiMo-VL-7B-sft/Step3-VL-10B): this used to read
        // hp.EmbeddingDim (the TEXT backbone's hidden size) as the per-token stride into the vision
        // projector's soft-token buffer. That's only correct when the projector's real output width
        // happens to equal the text hidden size — true for most working checkpoints, but not
        // guaranteed, and false for at least these two, where the projector genuinely outputs a
        // narrower vector. Using the wrong (larger) stride then slices `soft.AsSpan(t * embd, embd)`
        // past the end of the real, shorter buffer once `t` is large enough — an
        // ArgumentOutOfRangeException from the Span slice itself, not a meaningful error. Use the
        // vision embedder's own reported width instead, and fail with a clear, actionable message
        // up front if it can't actually feed the text backbone (ForwardEmbedding requires an exact
        // length match against hp.EmbeddingDim; a size mismatch here is a real architecture-adapter
        // bug, not something a caller can work around).
        int embd = vision.EmbeddingDim;
        if (embd != hp.EmbeddingDim && embd != hp.EmbeddingDim * (1 + hp.NumDeepstack))
        {
            AnsiConsole.ErrorLine(
                $"[red]Error:[/] vision projector ({vision.ProjectorType}) outputs {embd}-dim " +
                $"embeddings but the text backbone expects {hp.EmbeddingDim}-dim input — this " +
                $"checkpoint's vision adapter has a real dimension mismatch and cannot be used " +
                $"for image input as converted.");
            return 1;
        }

        // Reconcile the number of <image> markers in the prompt with the number of --image
        // files. No markers → prepend one placeholder per image (in --image order). Otherwise
        // the counts must match so the i-th marker pairs with the i-th --image.
        //
        // The text substituted in must be the model's own placeholder marker (vision.
        // PlaceholderMarker), not a hardcoded literal: this method originally only supported
        // Gemma 4, whose marker happens to literally be "<|image|>", but every other architecture
        // uses its own text (DotsOcr/Granite4 = "<image_pad>", InternVL = "<IMG_CONTEXT>", etc.).
        // Hardcoding "<|image|>" for all of them meant the substituted text tokenized back to
        // ordinary characters instead of the model's real placeholder token id, so the later
        // placeholder-count check always found 0 -- see docs/done/vl-migration-plan-2026-08-20.md.
        int markerCount = CountOccurrences(s.Prompt!, ImageMarker);
        string userMsg;
        if (markerCount == 0)
        {
            userMsg = string.Concat(Enumerable.Repeat(vision.PlaceholderMarker, nImages)) + s.Prompt;
        }
        else if (markerCount == nImages)
        {
            userMsg = s.Prompt!.Replace(ImageMarker, vision.PlaceholderMarker);
        }
        else
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] prompt has {markerCount} '{ImageMarker}' marker(s) but " +
                $"{nImages} --image file(s) were given; the counts must match (or omit markers to prepend the images).");
            return 1;
        }

        // Project every image to its soft-token block up front, in --image order.
        var blocks = new (float[] Soft, int NTok)[nImages];
        var grids = new (int W, int H)[nImages];
        var segments = new IReadOnlyList<VisionSegment>?[nImages];
        int totalSoft = 0;
        for (int i = 0; i < nImages; i++)
        {
            float[] soft;
            int nTok;
            try
            {
                soft = vision.EmbedImageFile(imagePaths[i], out nTok);
            }
            catch (Exception ex) when (ex is IOException or NotSupportedException or InvalidDataException
                                          or UnauthorizedAccessException or System.Security.SecurityException)
            {
                AnsiConsole.ErrorLine($"[red]Error reading image[/] {Markup.Escape(imagePaths[i])}: {Markup.Escape(ex.Message)}");
                return 1;
            }
            blocks[i] = (soft, nTok);
            grids[i] = vision.LastTokenGrid;
            segments[i] = vision.LastSegments;
            totalSoft += nTok;
            AnsiConsole.MarkupLine($"[dim]Image {i + 1}/{nImages}: {vision.ProjectorType} -> {nTok} soft tokens ({embd}-dim)[/]");
        }

        int imgOpen = !string.IsNullOrEmpty(vision.ImageOpenMarker) && tok.SpecialTokens.TryGetValue(vision.ImageOpenMarker, out var o) ? o : -1;
        int imgClose = !string.IsNullOrEmpty(vision.ImageCloseMarker) && tok.SpecialTokens.TryGetValue(vision.ImageCloseMarker, out var c) ? c : -1;
        // A close marker that is plain text rather than a special token (DeepSeek-OCR's newline, mtmd img_end)
        // is tokenized and fed as ordinary tokens after the image.
        int[] imgCloseText = imgClose < 0 && !string.IsNullOrEmpty(vision.ImageCloseMarker)
            ? tok.Encode(vision.ImageCloseMarker).Where(t => !tok.SpecialTokens.Values.Contains(t)).ToArray()
            : [];
        // Determine whether the model's placeholder is a real special token or plain text.
        // Classic LLaVA-1.5 (llava-v1.5-*): its GGUF is a plain LLaMA-2 32K vocab — "<image>" is
        // NOT a special token, so the previous fallback chain (<|image|> -> <image_soft_token> ->
        // 258880) silently produced a placeholder ID that never appeared in the tokenised output,
        // and the count check always failed with "found 0". Do not fall through to 258880 for a
        // model whose own marker is plain text — that is a silent-failure trap (258880 is a
        // hardcoded magic number that may appear in some other model's vocab but carries no meaning
        // here). Instead, use the direct-splice path below (found 2026-09-27).
        bool markerIsSpecialToken = tok.SpecialTokens.ContainsKey(vision.PlaceholderMarker);
        int placeholder;
        if (markerIsSpecialToken)
        {
            placeholder = tok.SpecialTokens[vision.PlaceholderMarker];
        }
        else if (tok.SpecialTokens.TryGetValue("<|image|>", out var ph2))
        {
            // Model has a different real special token; keep original fallback for models that
            // reported a wrong PlaceholderMarker but genuinely have <|image|> as a special token.
            placeholder = ph2;
            markerIsSpecialToken = true;
        }
        else if (tok.SpecialTokens.TryGetValue("<image_soft_token>", out var ph3))
        {
            placeholder = ph3;
            markerIsSpecialToken = true;
        }
        else
        {
            // PlaceholderMarker is plain text (not a special token). Use the direct-splice path.
            // Do NOT fall back to 258880 here — that sentinel means nothing in this vocabulary
            // and would produce silent zero-count failures for any model on this path.
            placeholder = -1;  // sentinel: used only by the direct-splice builder below
        }

        // No injected default system prompt for image prompts: llama-server (the parity reference for every
        // vision model) renders the model's own template as-is, and Qwen3-based VLMs (Step3-VL) are not the
        // text-only Qwen3 chat models that default was measured on.
        var prompt = FormatPrompt(userMsg, s.SystemPrompt, enableThinking: !s_noThinking, injectDefaultSystem: false);
        List<int> allTokens;
        if (markerIsSpecialToken)
        {
            // Normal path: the placeholder is a real special token that survives tokenisation;
            // encode the whole prompt and count occurrences.
            allTokens = EnsureBos(tok.Encode(prompt), tok).ToList();
            int placeholdersFound = allTokens.Count(t => t == placeholder);
            if (placeholdersFound != nImages)
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] expected {nImages} image placeholder token(s) ({vision.PlaceholderMarker}, {placeholder}) " +
                    $"after templating but found {placeholdersFound}; this model may not support image input.");
                return 1;
            }
        }
        else
        {
            // Direct-splice path (classic LLaVA-1.5 and any model whose PlaceholderMarker is plain
            // text). llama.cpp's own mtmd splits the prompt text at the marker string, tokenises
            // each piece separately (BOS on the first piece only), then splices the image embeddings
            // directly between the pieces — no placeholder token ID is needed. We do the same:
            // split on the literal marker, validate the count, and build allTokens with sentinel
            // (-1) at each image injection point. The playback loop below already handles
            // id == placeholder (which is -1 here) identically to the special-token path.
            string[] textParts = prompt.Split(vision.PlaceholderMarker, StringSplitOptions.None);
            int spliceCount = textParts.Length - 1;
            if (spliceCount != nImages)
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] prompt has {spliceCount} '{vision.PlaceholderMarker}' occurrence(s) " +
                    $"after templating but {nImages} --image file(s) were given; the counts must match.");
                return 1;
            }
            allTokens = [];
            for (int pi = 0; pi < textParts.Length; pi++)
            {
                // Tokenise each text segment. BOS is prepended only to the first segment, matching
                // llama.cpp's common_tokenize(add_special: true) on the first part and (add_special:
                // false) on the remainder.
                var segTokens = tok.Encode(textParts[pi]);
                if (pi == 0) segTokens = EnsureBos(segTokens, tok);
                allTokens.AddRange(segTokens);
                if (pi < spliceCount)
                    allTokens.Add(-1);  // sentinel: expands to the image block in the playback loop
            }
        }

        // Each placeholder expands to [open] + its soft tokens + [close], so the prefill is
        // longer than the token list — an image is easily hundreds of positions. Check the
        // expanded length before the first Forward: ForwardPass sizes its attention-score and
        // RoPE scratch from the context bound but not its KV cache, so running past it writes
        // out of bounds instead of failing.
        int markerTokens = (imgOpen >= 0 ? 1 : 0) + (imgClose >= 0 ? 1 : 0);
        markerTokens += imgCloseText.Length;
        foreach (var seg in segments)
            if (seg is not null) totalSoft += seg.Count(x => x.Token is not null) - markerTokens;
        int plannedPrefill = allTokens.Count + (nImages * markerTokens) + totalSoft - nImages;
        if (plannedPrefill >= maxContextLength)
        {
            AnsiConsole.ErrorLine(
                $"[red]Error:[/] prompt plus images expand to {plannedPrefill} tokens ({totalSoft} image) " +
                $"but the active context is {maxContextLength}; use fewer/smaller images or raise " +
                $"[yellow]--ctx-size[/] so at least one token can be generated.");
            return 1;
        }
        sp = ClampToRemainingContext(sp, plannedPrefill, maxContextLength);

        var sw = Stopwatch.StartNew();
        int pos = 0;
        int imgIdx = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in allTokens)
        {
            if (id == placeholder)
            {
                var grid = grids[imgIdx];
                var segs = segments[imgIdx];
                var (soft, nTok) = blocks[imgIdx++];
                if (segs is not null)
                {
                    foreach (var seg in segs)
                    {
                        if (seg.Token is not null)
                        {
                            if (!tok.SpecialTokens.TryGetValue(seg.Token, out int segTok))
                                throw new InvalidOperationException($"Image segment marker '{seg.Token}' is not a special token of this model.");
                            logits = fwd.Forward(segTok, pos++);
                        }
                        else
                        {
                            for (int t = seg.Start; t < seg.Start + seg.Count; t++)
                                logits = fwd.ForwardEmbedding(soft.AsSpan(t * embd, embd), pos++);
                        }
                    }
                    continue;
                }
                if (imgOpen >= 0) logits = fwd.Forward(imgOpen, pos++);
                if (grid.W * grid.H == nTok)
                {
                    if (fwd is ForwardPass cpuMrope && cpuMrope.UsesMRope) cpuMrope.AddMRopeImage(pos, grid.W, grid.H);
                    else if (fwd is GpuForwardPass gpuMrope && gpuMrope.UsesMRope) gpuMrope.AddMRopeImage(pos, grid.W, grid.H);
                }
                if (fwd is ForwardPass cpuBatch)
                {
                    // One batched prefill for the whole image instead of nTok single-token passes.
                    logits = cpuBatch.PrefillEmbeddings(soft.AsSpan(0, nTok * embd), nTok, pos);
                    pos += nTok;
                }
                else
                {
                    for (int t = 0; t < nTok; t++)
                        logits = fwd.ForwardEmbedding(soft.AsSpan(t * embd, embd), pos++);
                }
                if (imgClose >= 0) logits = fwd.Forward(imgClose, pos++);
                foreach (int ct in imgCloseText) logits = fwd.Forward(ct, pos++);
            }
            else
            {
                logits = fwd.Forward(id, pos++);
            }
        }
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        bool graniteDiagnostic = vision.ProjectorType.Contains("granite4", StringComparison.OrdinalIgnoreCase)
            && Environment.GetEnvironmentVariable("STINGRAY_GRANITE4_DIAG") == "1";
        if (graniteDiagnostic)
        {
            Span<int> topIds = stackalloc int[10];
            Span<float> topValues = stackalloc float[10];
            int topCount = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                float value = logits[i];
                if (topCount == topIds.Length && value <= topValues[topCount - 1]) continue;
                int insert = Math.Min(topCount, topIds.Length - 1);
                while (insert > 0 && value > topValues[insert - 1])
                {
                    topValues[insert] = topValues[insert - 1];
                    topIds[insert] = topIds[insert - 1];
                    insert--;
                }
                topValues[insert] = value;
                topIds[insert] = i;
                if (topCount < topIds.Length) topCount++;
            }
            var top = new System.Text.StringBuilder(256);
            for (int i = 0; i < topCount; i++)
            {
                if (i > 0) top.Append(" | ");
                top.Append(topIds[i]).Append('(').Append($"{topValues[i]:F4}")
                    .Append(",'").Append(tok.Decode([topIds[i]])).Append('\'');
            }
            float eogLogit = float.NegativeInfinity;
            foreach (int id in tok.EogTokenIds)
                if (id >= 0 && id < logits.Length && logits[id] > eogLogit)
                    eogLogit = logits[id];
            var periodTokens = tok.Encode(".");
            float periodLogit = periodTokens.Count == 0 ? float.NaN : logits[periodTokens[0]];
            float margin = topCount > 1 ? topValues[0] - topValues[1] : float.NaN;
            Console.Error.WriteLine($"[GRANITE4-DIAG] prefill={Environment.GetEnvironmentVariable("STINGRAY_GRANITE4_PREFILL") ?? "current"} " +
                $"deepstack={hp.NumDeepstack} mapping=[{string.Join(',', hp.DeepstackMapping ?? [])}] " +
                $"top1-top2={margin:F6} eog-max={eogLogit:F6} period={periodLogit:F6} top10={top}");
        }

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        sw.Restart();
        var (generated, totalDecoded) =
            DecodeLoop(fwd.Forward, logits, pos, tok, sp, rng, s.VerbosePrompt, s.HideThinking, s.MaxThinkingTokens,
                repeatLastN: s.RepeatLastN, graniteDiagnostic: graniteDiagnostic);
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {pos} tokens ({totalSoft} image + {pos - totalSoft} text), " +
            $"{pos / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            "[/]");
        return 0;
    }
}
