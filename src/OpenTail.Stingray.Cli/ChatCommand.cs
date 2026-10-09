using System.ComponentModel;
using OpenTail.Stingray;
using OpenTail.Stingray.Executors;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Cli.Terminal;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// <c>stingray chat [message]</c>: quick interactive or single-turn chat using the default or specified model.
/// </summary>
public sealed class ChatCommand : Command<ChatCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-p|--prompt <MESSAGE>", Positional = true)]
        [Description("Optional single-turn message. If omitted, starts an interactive multi-turn chat.")]
        public string? Message { get; init; }

        [CommandOption("-m|--model <ID>")]
        [Description("Model catalog id (default: qwen2.5-0.5b).")]
        public string? Model { get; init; }

        [CommandOption("--model-file <PATH>")]
        [Description("Direct path to a model file (.gguf) to load.")]
        public string? ModelFile { get; init; }

        [CommandOption("-t|--temp|--temperature <FLOAT>")]
        [Description("Sampling temperature. Default: 0.7.")]
        [DefaultValue(0.7f)]
        public float Temperature { get; init; } = 0.7f;

        [CommandOption("-c|--ctx-size|--context-size <INT>")]
        [Description("Context size in tokens. Default: 2048.")]
        [DefaultValue(2048)]
        public int ContextSize { get; init; } = 2048;

        [CommandOption("-n|--max-tokens <INT>")]
        [Description("Maximum tokens to generate per turn. Default: 512.")]
        [DefaultValue(512)]
        public int MaxTokens { get; init; } = 512;

        [CommandOption("-s|--system <TEXT>")]
        [Description("System message instructions for the conversation.")]
        public string? SystemPrompt { get; init; }

        [CommandOption("--thinking")]
        [Description("Display model thinking/reasoning output chunks.")]
        public bool ShowThinking { get; init; }

        [CommandOption("-g|--backend <BACKEND>")]
        [Description("Compute backend: auto (default), cpu, or vulkan.")]
        public string Backend { get; init; } = "auto";

        [CommandOption("--gpu-layers <N>")]
        [Description("Number of layers to offload to GPU (-1 for auto/all). Default: -1.")]
        [DefaultValue(-1)]
        public int GpuLayers { get; init; } = -1;
    }

    protected override int Execute(Settings s, CancellationToken cancellation)
    {
        if (!CatalogTaskResolver.TryResolveOrOffer("chat", s.Model, s.ModelFile, out var resolved, out string? error, cancellation))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(error ?? "Failed to resolve chat model.")}");
            return ExitCodes.Usage;
        }

        var modelParams = new ModelParams(resolved!.ModelPath)
        {
            Backend = s.Backend,
            GpuLayerCount = s.GpuLayers >= 0 ? s.GpuLayers : (s.Backend.Equals("cpu", StringComparison.OrdinalIgnoreCase) ? 0 : 999)
        };

        using var model = Stingray.Model.Load(modelParams);
        using var context = model.CreateContext(new ContextParams { ContextSize = (uint)s.ContextSize });
        var executor = new InteractiveExecutor(context);
        var session = new ChatSession(executor);

        if (!string.IsNullOrWhiteSpace(s.SystemPrompt))
        {
            session.AddSystemMessage(s.SystemPrompt);
        }

        if (!string.IsNullOrWhiteSpace(s.Message))
        {
            return RunSingleTurn(session, s, s.Message, cancellation);
        }

        return RunInteractiveLoop(session, resolved, s, cancellation);
    }

    private static int RunSingleTurn(ChatSession session, Settings s, string message, CancellationToken cancellation)
    {
        var inferenceParams = new InferenceParams
        {
            Temperature = s.Temperature,
            MaxTokens = s.MaxTokens,
            EnableThinking = s.ShowThinking
        };

        try
        {
            Task.Run(async () =>
            {
                await foreach (var chunk in session.ChatChunksAsync(message, inferenceParams, cancellation))
                {
                    if (chunk.Kind == GenerateChunkKind.Thinking)
                    {
                        if (s.ShowThinking)
                        {
                            Console.ForegroundColor = ConsoleColor.DarkGray;
                            Console.Write(chunk.Text);
                            Console.ResetColor();
                        }
                    }
                    else if (chunk.Kind == GenerateChunkKind.Text)
                    {
                        Console.Write(chunk.Text);
                    }
                }
            }, cancellation).GetAwaiter().GetResult();

            Console.WriteLine();
            return ExitCodes.Success;
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("\n[interrupted]");
            return ExitCodes.Interrupted;
        }
        catch (Exception ex)
        {
            AnsiConsole.ErrorLine($"\n[red]Error:[/] {Markup.Escape(ex.Message)}");
            return ExitCodes.Failure;
        }
    }

    private static int RunInteractiveLoop(ChatSession session, ResolvedModelTask resolved, Settings s, CancellationToken cancellation)
    {
        string modelName = resolved.Entry?.Id ?? Path.GetFileName(resolved.ModelPath);
        AnsiConsole.MarkupLine($"[bold]Stingray Chat[/] ([yellow]{Markup.Escape(modelName)}[/])");
        AnsiConsole.MarkupLine("[dim]Type '/exit' or press Ctrl+C to quit.[/]");
        Console.WriteLine();

        while (!cancellation.IsCancellationRequested)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("> ");
            Console.ResetColor();

            string? input = Console.ReadLine();
            if (input == null) break;
            input = input.Trim();
            if (input.Length == 0) continue;
            if (input.Equals("/exit", StringComparison.OrdinalIgnoreCase) || input.Equals("/quit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            var inferenceParams = new InferenceParams
            {
                Temperature = s.Temperature,
                MaxTokens = s.MaxTokens,
                EnableThinking = s.ShowThinking
            };

            try
            {
                Task.Run(async () =>
                {
                    await foreach (var chunk in session.ChatChunksAsync(input, inferenceParams, cancellation))
                    {
                        if (chunk.Kind == GenerateChunkKind.Thinking)
                        {
                            if (s.ShowThinking)
                            {
                                Console.ForegroundColor = ConsoleColor.DarkGray;
                                Console.Write(chunk.Text);
                                Console.ResetColor();
                            }
                        }
                        else if (chunk.Kind == GenerateChunkKind.Text)
                        {
                            Console.Write(chunk.Text);
                        }
                    }
                }, cancellation).GetAwaiter().GetResult();

                Console.WriteLine();
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("\n[interrupted]");
                break;
            }
            catch (Exception ex)
            {
                AnsiConsole.ErrorLine($"\n[red]Error:[/] {Markup.Escape(ex.Message)}");
            }
        }

        return ExitCodes.Success;
    }
}
