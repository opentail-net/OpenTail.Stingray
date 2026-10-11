// Structured extraction: turn free text into a typed JSON record that is guaranteed to match a schema.
//   dotnet run --project samples/OpenTail.Stingray.Sample.Extract -- [-m model.gguf] [--text "..."]
// With no --text it reads the text from standard input; with neither it extracts from a built-in invoice email.
using System.Text;
using System.Text.Json;
using OpenTail.Stingray;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Core.Grammar;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Executors;

string? modelPath = Environment.GetEnvironmentVariable("STINGRAY_MODEL");
string? text = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "-m" or "--model" when i + 1 < args.Length: modelPath = args[++i]; break;
        case "--text" when i + 1 < args.Length: text = args[++i]; break;
        default:
            Console.Error.WriteLine("usage: opentail-extract-sample [-m <model.gguf>] [--text \"...\"]");
            return 2;
    }
}

if (modelPath is null && !ModelHome.TryResolveModelPath("qwen2.5-0.5b", out modelPath))
{
    Console.Error.WriteLine("error: pass -m <model.gguf>, set STINGRAY_MODEL, or run 'stingray setup chat'.");
    return 2;
}

if (text is null && Console.IsInputRedirected)
{
    string piped = await Console.In.ReadToEndAsync();
    if (!string.IsNullOrWhiteSpace(piped)) text = piped;   // empty or closed stdin (CI, some shells) falls back to the built-in email
}
text ??= """
    Hi team, please find invoice INV-2026-0417 from Northwind Traders for 1,284.50 EUR.
    Payment is due on 2026-11-15. Thanks, Dana
    """;

// The schema says what shape the answer must have. The constraint makes every token the model emits keep the output valid for it;
// orderedProperties makes the model write the fields in the order declared here, which gives small models a much better chance of filling them in.
const string schema = """
    {
      "type": "object",
      "properties": {
        "vendor":         { "type": "string" },
        "invoice_number": { "type": "string" },
        "total":          { "type": "number" },
        "currency":       { "type": "string" },
        "due_date":       { "type": "string" }
      },
      "required": ["vendor", "invoice_number", "total", "currency", "due_date"]
    }
    """;

using var model = Model.Load(modelPath!);
using var context = model.CreateContext(new ContextParams { ContextSize = 2048 });
var session = new ChatSession(new InteractiveExecutor(context));
session.AddSystemMessage("You extract data from text. Reply with JSON only.");

var constraint = JsonConstraint.FromSchemaJson(new GrammarVocabulary(context.Tokenizer), schema, orderedProperties: true);
var json = new StringBuilder();
await foreach (var chunk in session.ChatChunksAsync(
    $"Extract the invoice details from this message:\n\n{text}",
    new InferenceParams { MaxTokens = 200, Temperature = 0f, EnableThinking = false, Constraint = constraint }))
{
    if (chunk.Kind == GenerateChunkKind.Text) json.Append(chunk.Text);
}

// The constraint guarantees the SHAPE. It does not guarantee the values are true, so check them against the source text.
using var doc = JsonDocument.Parse(json.ToString());
Console.WriteLine(json);
string invoice = doc.RootElement.GetProperty("invoice_number").GetString() ?? "";
bool grounded = invoice.Length > 0 && text.Contains(invoice, StringComparison.Ordinal);
Console.Error.WriteLine(grounded
    ? "check: invoice_number appears in the source text"
    : "check: invoice_number is empty or does NOT appear in the source text, do not trust this record");
return grounded ? 0 : 1;
