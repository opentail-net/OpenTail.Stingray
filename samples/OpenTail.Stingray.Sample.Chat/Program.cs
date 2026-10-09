// OpenTail.Stingray library sample: minimal streaming chat against a GGUF model.
//
//   dotnet run --project samples/OpenTail.Stingray.Sample.Chat -c Release -- \
//       -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
//
// Stdin is read one line at a time (or piped — EOF ends the session); generated
// tokens are streamed to stdout as soon as the engine produces them.

using OpenTail.Stingray;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Executors;

string? modelPath = null;
string? systemPrompt = null;
float temperature = 0.7f;
uint contextSize = 2048;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "-m" or "--model" when i + 1 < args.Length:
            modelPath = args[++i]; break;
        case "-s" or "--system" when i + 1 < args.Length:
            systemPrompt = args[++i]; break;
        case "-c" or "--ctx-size" when i + 1 < args.Length:
            contextSize = uint.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--temp" when i + 1 < args.Length:
            temperature = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "-h" or "--help":
            Console.Error.WriteLine("usage: opentail-llm-sample-chat -m <model.gguf> [--system <prompt>] [--temp 0.7] [--ctx-size 2048]");
            return 0;
    }
}

modelPath ??= Environment.GetEnvironmentVariable("STINGRAY_MODEL");
if (modelPath is null || !File.Exists(modelPath))
{
    Console.Error.WriteLine("error: pass -m <model.gguf> or set STINGRAY_MODEL.");
    return 1;
}

// Ctrl+C => cancel generation and unblock the read loop. We handle the first
// press cooperatively; a second press lets the runtime terminate the process.
using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    if (cts.IsCancellationRequested) return;
    e.Cancel = true;
    cts.Cancel();
};

Console.Error.Write($"loading {modelPath} ... ");
using var model = Model.Load(new ModelParams(modelPath)
{
    Backend = "cpu",
    GpuLayerCount = 0
});

using var context = model.CreateContext(new ContextParams
{
    ContextSize = contextSize
});

var executor = new InteractiveExecutor(context);
var session = new ChatSession(executor);

if (systemPrompt is not null)
{
    session.AddSystemMessage(systemPrompt);
}

Console.Error.WriteLine($"{model.Architecture} (ctx {context.ContextSize})");
Console.Error.WriteLine("ready. type a message, Ctrl+D / Ctrl+Z to exit.\n");

var inferenceParams = new InferenceParams
{
    Temperature = temperature,
    MaxTokens = 512
};

while (!cts.IsCancellationRequested)
{
    Console.Write("> ");
    string? line;
    try { line = await Console.In.ReadLineAsync(cts.Token); }
    catch (OperationCanceledException) { break; }
    if (line is null) break;             // stdin closed (Ctrl+D / Ctrl+Z / piped EOF)
    if (line.Length == 0) continue;

    try
    {
        await foreach (var chunk in session.ChatChunksAsync(line, inferenceParams, cts.Token))
        {
            if (chunk.Kind == GenerateChunkKind.Text)
            {
                Console.Out.Write(chunk.Text);
                Console.Out.Flush();
            }
        }
    }
    catch (OperationCanceledException)
    {
        Console.Error.WriteLine("\n[cancelled]");
        break;
    }
    Console.WriteLine();
}

return 0;
