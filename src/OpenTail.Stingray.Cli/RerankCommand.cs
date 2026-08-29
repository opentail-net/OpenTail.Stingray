using OpenTail.Stingray.Core.Embeddings;

namespace OpenTail.Stingray.Cli;

public sealed class RerankCommand : Command<RerankCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-q|--query <TEXT>")]
        [Description("Search query to rank candidate documents against.")]
        public string? Query { get; init; }

        [CommandOption("-d|--doc|--document <TEXT>")]
        [Description("Candidate document string (can be specified multiple times).")]
        public string? Document { get; init; }

        [CommandOption("-f|--file <PATH>")]
        [Description("Optional file containing candidate documents (one per line).")]
        public string? FilePath { get; init; }

        [CommandOption("-m|--model <MODEL>")]
        [Description("Reranker model name or GGUF path. Default: bge-reranker-large.")]
        public string Model { get; init; } = "bge-reranker-large";

        [CommandOption("-k|--top-n <N>")]
        [Description("Number of top most relevant documents to return.")]
        public int? TopN { get; init; }

        [CommandOption("-o|--output <PATH>")]
        [Description("Optional output file path to write reranked results as JSON.")]
        public string? OutputPath { get; init; }
    }

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        if (string.IsNullOrWhiteSpace(s.Query))
        {
            Console.Error.WriteLine("Error: Query is required. Use -q or --query \"search query\".");
            return 1;
        }

        List<string> docs = [];

        if (!string.IsNullOrWhiteSpace(s.Document))
        {
            docs.Add(s.Document);
        }

        if (!string.IsNullOrWhiteSpace(s.FilePath) && File.Exists(s.FilePath))
        {
            var lines = File.ReadAllLines(s.FilePath).Where(l => !string.IsNullOrWhiteSpace(l));
            docs.AddRange(lines);
        }

        if (docs.Count == 0)
        {
            Console.Error.WriteLine("Error: At least one candidate document is required. Use -d \"doc\" or -f <file.txt>.");
            return 1;
        }

        using var engine = new EmbeddingEngine(modelName: s.Model);

        Console.WriteLine($"Cross-Encoder Document Reranking ({engine.ModelName})");
        Console.WriteLine($"Query:     \"{s.Query}\"");
        Console.WriteLine($"Documents: {docs.Count}");
        Console.WriteLine($"Top N:     {s.TopN ?? docs.Count}");
        Console.WriteLine();

        var sw = Stopwatch.StartNew();

        var req = new RerankRequest
        {
            Query = s.Query,
            Documents = docs,
            TopN = s.TopN,
            Model = s.Model,
            ReturnDocuments = true
        };

        var result = engine.Rerank(req);
        sw.Stop();

        Console.WriteLine("--- Ranked Results ---");
        for (int rank = 0; rank < result.Results.Count; rank++)
        {
            var r = result.Results[rank];
            Console.WriteLine($"  #{rank + 1} [Score: {r.RelevanceScore:F4}] (Original Index {r.Index}): {r.Document}");
        }
        Console.WriteLine("----------------------");
        Console.WriteLine();
        Console.WriteLine($"Reranked {docs.Count} documents in {sw.ElapsedMilliseconds}ms ({result.TotalTokens} tokens)");

        if (!string.IsNullOrEmpty(s.OutputPath))
        {
            File.WriteAllText(s.OutputPath, JsonSerializer.Serialize(result.Results));
            Console.WriteLine($"Saved results to: {Path.GetFullPath(s.OutputPath)}");
        }

        return 0;
    }
}
