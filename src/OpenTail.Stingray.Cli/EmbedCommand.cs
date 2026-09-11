using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Cli;

public sealed class EmbedCommand : Command<EmbedCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-p|--prompt <TEXT>")]
        [Description("Input text prompt to embed into a dense semantic vector.")]
        public string? Prompt { get; init; }

        [CommandOption("-f|--file <PATH>")]
        [Description("Optional path to text file containing input text or lines to embed.")]
        public string? FilePath { get; init; }

        [CommandOption("-m|--model <MODEL>")]
        [Description("Embedding model name or GGUF path. Default: text-embedding-3-small.")]
        public string Model { get; init; } = "text-embedding-3-small";

        [CommandOption("-d|--dimensions <N>")]
        [Description("Matryoshka representation dimension reduction (e.g. 512, 768, 1536).")]
        public int? Dimensions { get; init; }

        [CommandOption("--pooling <TYPE>")]
        [Description("Sequence pooling strategy: mean (default), cls, or last.")]
        public string Pooling { get; init; } = "mean";

        [CommandOption("--no-norm")]
        [Description("Disable unit L2 vector normalization.")]
        public bool NoNorm { get; init; }

        [CommandOption("-o|--output <PATH>")]
        [Description("Optional output file path to write embedding vectors as JSON.")]
        public string? OutputPath { get; init; }
    }

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        List<string> texts = [];

        if (!string.IsNullOrWhiteSpace(s.Prompt))
        {
            texts.Add(s.Prompt);
        }

        if (!string.IsNullOrWhiteSpace(s.FilePath) && File.Exists(s.FilePath))
        {
            var lines = File.ReadAllLines(s.FilePath).Where(l => !string.IsNullOrWhiteSpace(l));
            texts.AddRange(lines);
        }

        if (texts.Count == 0)
        {
            Console.Error.WriteLine("Error: Input prompt or file is required. Use -p \"text\" or -f <file.txt>.");
            return 1;
        }

        PoolingType pooling = s.Pooling.ToLowerInvariant() switch
        {
            "cls" or "first" => PoolingType.Cls,
            "last" or "lasttoken" => PoolingType.LastToken,
            _ => PoolingType.Mean
        };

        if (s.Model.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) && File.Exists(s.Model))
        {
            using var onnxSession = OnnxModelSession.TryLoad(s.Model);
            if (onnxSession == null)
            {
                Console.Error.WriteLine("Error: Could not load ONNX model file. Ensure onnxruntime.dll is available.");
                return 1;
            }

            var wordPieceTokenizer = TryLoadWordPieceTokenizer(s.Model);

            Console.WriteLine($"Dense Text Embedding Generation ({Path.GetFileName(s.Model)})");
            Console.WriteLine($"Input Count:  {texts.Count}");
            Console.WriteLine($"Pooling Mode: {pooling}");
            Console.WriteLine($"L2 Normalize: {!s.NoNorm}");
            Console.WriteLine($"Tokenizer:    {(wordPieceTokenizer is not null ? "real WordPiece (vocab.txt)" : "char-per-token placeholder (no vocab.txt found -- see EmbedCommand.TryLoadWordPieceTokenizer)")}");
            Console.WriteLine();

            var swOnnx = Stopwatch.StartNew();
            var vectors = new List<float[]>(texts.Count);
            int totalTokens = 0;

            foreach (var text in texts)
            {
                long[] inputIds;
                if (wordPieceTokenizer is not null)
                {
                    // Real WordPiece tokenization (BasicTokenizer + WordpieceTokenizer, a faithful
                    // port of HuggingFace transformers' real BertTokenizer algorithm) against this
                    // checkpoint's own real vocab.txt -- see BertWordPieceTokenizer.cs.
                    inputIds = wordPieceTokenizer.Encode(text);
                }
                else
                {
                    // Fallback for checkpoints with no locally-available vocab.txt: NOT real
                    // WordPiece/BPE tokenization -- maps each raw character to its char code as a
                    // placeholder "token id". The model will run and produce a real vector, but it
                    // will not be a semantically meaningful embedding of the input text.
                    inputIds = text.Select(c => (long)c).ToArray();
                    if (inputIds.Length == 0) inputIds = [0];
                }
                // Most BERT-family encoders cap position embeddings at 512; truncate defensively
                // instead of letting ONNX Runtime fail with an opaque broadcast error deep in the
                // graph (real WordPiece tokenization rarely exceeds this for reasonably-sized
                // inputs, but the char-per-token fallback inflates length ~4x and hits it often).
                const int maxPositions = 512;
                if (inputIds.Length > maxPositions)
                {
                    Console.Error.WriteLine($"Warning: input truncated from {inputIds.Length} to {maxPositions} tokens.");
                    inputIds = inputIds[..maxPositions];
                }
                totalTokens += inputIds.Length;
                long[] attentionMask = new long[inputIds.Length];
                Array.Fill(attentionMask, 1L);
                long[] tokenTypeIds = new long[inputIds.Length]; // all-zero: single-segment input

                Dictionary<string, float[]> outputs;
                try
                {
                    outputs = onnxSession.Run(
                        ("input_ids", inputIds, [1, inputIds.Length]),
                        ("attention_mask", attentionMask, [1, inputIds.Length]),
                        ("token_type_ids", tokenTypeIds, [1, inputIds.Length])
                    );
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error: ONNX inference failed for input \"{text}\": {ex.Message}");
                    return 1;
                }

                if (outputs.Values.FirstOrDefault() is { } outTensor && outTensor.Length > 0)
                {
                    int outDim = outTensor.Length / inputIds.Length;
                    float[] pooled = outDim > 0
                        ? EmbeddingNormalizer.ApplyPooling(outTensor, inputIds.Length, outDim, pooling)
                        : outTensor;

                    if (s.Dimensions.HasValue && s.Dimensions.Value > 0 && s.Dimensions.Value < pooled.Length)
                        pooled = EmbeddingNormalizer.TruncateAndNormalize(pooled, s.Dimensions.Value);
                    else if (!s.NoNorm)
                        EmbeddingNormalizer.NormalizeL2(pooled);

                    vectors.Add(pooled);
                }
            }
            swOnnx.Stop();

            for (int i = 0; i < vectors.Count; i++)
            {
                var vec = vectors[i];
                Console.WriteLine($"Vector [{i}] (dim={vec.Length}): [{vec[0]:F4}, {vec[1]:F4}, {vec[2]:F4}, ... {vec[^1]:F4}]");
            }

            Console.WriteLine();
            Console.WriteLine($"Processed {texts.Count} text(s) in {swOnnx.ElapsedMilliseconds}ms ({totalTokens} tokens)");

            if (!string.IsNullOrEmpty(s.OutputPath))
            {
                File.WriteAllText(s.OutputPath, SerializeVectors(vectors));
                Console.WriteLine($"Saved embeddings to: {Path.GetFullPath(s.OutputPath)}");
            }
            return 0;
        }

        using var engine = new EmbeddingEngine(
            modelName: s.Model,
            embeddingDimensions: s.Dimensions ?? 1536,
            defaultPooling: pooling);

        Console.WriteLine($"Dense Text Embedding Generation ({engine.ModelName})");
        Console.WriteLine($"Input Count:  {texts.Count}");
        Console.WriteLine($"Pooling Mode: {pooling}");
        Console.WriteLine($"Dimensions:   {s.Dimensions ?? engine.EmbeddingDimensions}");
        Console.WriteLine($"L2 Normalize: {!s.NoNorm}");
        Console.WriteLine();

        var sw = Stopwatch.StartNew();

        var req = new EmbeddingRequest
        {
            Inputs = texts,
            Model = s.Model,
            Dimensions = s.Dimensions,
            Normalize = !s.NoNorm,
            Pooling = pooling
        };

        var result = engine.Embed(req);
        sw.Stop();

        for (int i = 0; i < result.Data.Count; i++)
        {
            var item = result.Data[i];
            Console.WriteLine($"Vector [{item.Index}] (dim={item.Vector.Length}): [{item.Vector[0]:F4}, {item.Vector[1]:F4}, {item.Vector[2]:F4}, ... {item.Vector[^1]:F4}]");
        }

        Console.WriteLine();
        Console.WriteLine($"Processed {texts.Count} text(s) in {sw.ElapsedMilliseconds}ms ({result.TotalTokens} tokens)");

        if (!string.IsNullOrEmpty(s.OutputPath))
        {
            var vectors = result.Data.Select(d => d.Vector).ToList();
            File.WriteAllText(s.OutputPath, SerializeVectors(vectors));
            Console.WriteLine($"Saved embeddings to: {Path.GetFullPath(s.OutputPath)}");
        }

        return 0;
    }

    // Looks for a real vocab.txt next to the ONNX checkpoint. Tries, in order: (1) the exact
    // sibling name "<model-basename>-vocab.txt" (the convention used for the vocab files fetched
    // for all-MiniLM-L6-v2/BGE small/base/large -- see docs/00-current-work.md's 2026-09-11 entry
    // for provenance); (2) the same name with a trailing "_quantized"/"_qint8"/"_int8" suffix
    // stripped first, so a quantized checkpoint's own basename still resolves to its base model's
    // vocab; (3) a plain "vocab.txt" in the same directory (the standard HF convention, for any
    // future checkpoint that ships one directly alongside the .onnx file). Returns null (not an
    // error) when none exist -- callers fall back to the char-per-token placeholder rather than
    // failing outright, since not every ONNX checkpoint on this machine has a downloaded vocab yet.
    private static BertWordPieceTokenizer? TryLoadWordPieceTokenizer(string onnxModelPath)
    {
        string? dir = Path.GetDirectoryName(onnxModelPath);
        if (string.IsNullOrEmpty(dir)) dir = ".";
        string baseName = Path.GetFileNameWithoutExtension(onnxModelPath);

        var candidates = new List<string> { Path.Combine(dir, baseName + "-vocab.txt") };

        string[] stripSuffixes = ["_quantized", "_qint8", "_int8", "-quantized"];
        foreach (var suffix in stripSuffixes)
        {
            if (baseName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            {
                string stripped = baseName[..^suffix.Length];
                candidates.Add(Path.Combine(dir, stripped + "-vocab.txt"));
            }
        }

        candidates.Add(Path.Combine(dir, "vocab.txt"));

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                try
                {
                    return BertWordPieceTokenizer.LoadVocabFile(candidate);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Warning: found vocab file {candidate} but failed to load it: {ex.Message}. Falling back to char-per-token placeholder.");
                    return null;
                }
            }
        }
        return null;
    }

    // Hand-rolled JSON serialization for a simple List<float[]> shape, avoiding reflection-based
    // JsonSerializer.Serialize (disabled for this NativeAOT-trimmed app -- see CLAUDE.md rule 4).
    private static string SerializeVectors(List<float[]> vectors)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append('[');
        for (int i = 0; i < vectors.Count; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append('[');
            var vec = vectors[i];
            for (int j = 0; j < vec.Length; j++)
            {
                if (j > 0) sb.Append(',');
                sb.Append(vec[j].ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            }
            sb.Append(']');
        }
        sb.Append(']');
        return sb.ToString();
    }
}
