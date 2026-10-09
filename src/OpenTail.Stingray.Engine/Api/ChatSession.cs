#nullable enable

using System.Runtime.CompilerServices;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Executors;

namespace OpenTail.Stingray;

/// <summary>
/// Multi-turn conversational session managing conversation history and prompt execution.
/// Mirrors LLamaSharp's <c>ChatSession</c>.
/// Supports both simple string streaming (<see cref="ChatAsync(string, IInferenceParams?, CancellationToken)"/>)
/// and typed chunk streaming (<see cref="ChatChunksAsync(string, IInferenceParams?, CancellationToken)"/>)
/// preserving model thinking/reasoning and generation metrics.
/// </summary>
public class ChatSession
{
    private readonly IExecutor _executor;
    private readonly ChatHistory _history;

    /// <summary>Executor used for driving token generation.</summary>
    public IExecutor Executor => _executor;

    /// <summary>Current conversation history for this session.</summary>
    public ChatHistory History => _history;

    /// <summary>
    /// Optional custom formatter for turning <see cref="ChatHistory"/> into a model prompt.
    /// If null, the model's native Jinja template or ChatML fallback is used.
    /// </summary>
    public Func<ChatHistory, string>? PromptFormatter { get; set; }

    /// <summary>
    /// Initializes a new chat session over the specified executor.
    /// </summary>
    /// <param name="executor">Executor driving generation.</param>
    /// <param name="history">Optional existing chat history; creates a fresh history if null.</param>
    public ChatSession(IExecutor executor, ChatHistory? history = null)
    {
        ArgumentNullException.ThrowIfNull(executor);
        _executor = executor;
        _history = history ?? new ChatHistory();
    }

    /// <summary>Appends a message to the session's conversation history.</summary>
    public void AddMessage(AuthorRole role, string content) => _history.AddMessage(role, content);

    /// <summary>Appends a system message to the session's conversation history.</summary>
    public void AddSystemMessage(string content) => _history.AddSystemMessage(content);

    /// <summary>Appends a user message to the session's conversation history.</summary>
    public void AddUserMessage(string content) => _history.AddUserMessage(content);

    /// <summary>Appends an assistant message to the session's conversation history.</summary>
    public void AddAssistantMessage(string content) => _history.AddAssistantMessage(content);

    /// <summary>Appends a tool message to the session's conversation history.</summary>
    public void AddToolMessage(string content) => _history.AddToolMessage(content);

    /// <summary>
    /// Formats the conversation history into a prompt string for generation.
    /// </summary>
    public virtual string FormatPrompt(ChatHistory history) => FormatPrompt(history, addGenerationPrompt: true);

    /// <summary>
    /// Formats the conversation history into a prompt string for generation, optionally specifying whether to add generation prompt.
    /// </summary>
    public virtual string FormatPrompt(ChatHistory history, bool addGenerationPrompt)
    {
        ArgumentNullException.ThrowIfNull(history);
        if (history.Count == 0) return string.Empty;

        if (PromptFormatter != null)
        {
            return PromptFormatter(history);
        }

        if (_executor.Context.Tokenizer is GgufTokenizer ggufTokenizer && ggufTokenizer.ChatTemplate is { } jinja)
        {
            var messages = new List<object?>(history.Count);
            foreach (var msg in history)
            {
                string role = msg.Role switch
                {
                    AuthorRole.System => "system",
                    AuthorRole.User => "user",
                    AuthorRole.Assistant => "assistant",
                    AuthorRole.Tool => "tool",
                    _ => "user"
                };
                messages.Add(new Dictionary<string, object?>
                {
                    ["role"] = role,
                    ["content"] = msg.Content
                });
            }

            try
            {
                return jinja.Render(new Dictionary<string, object?>
                {
                    ["messages"] = messages,
                    ["add_generation_prompt"] = addGenerationPrompt
                });
            }
            catch (ChatTemplateException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new ChatTemplateException($"Jinja chat template rendering failed: {ex.Message}");
            }
        }

        string arch = _executor.Context.Model?.Architecture ?? "llama";
        var protocol = ChatProtocolRegistry.For(arch);
        if (protocol.Render != null)
        {
            return protocol.Render(new ChatRenderRequest(history, addGenerationPrompt));
        }

        var sb = new StringBuilder();
        foreach (var msg in history)
        {
            string role = msg.Role switch
            {
                AuthorRole.System => "system",
                AuthorRole.User => "user",
                AuthorRole.Assistant => "assistant",
                AuthorRole.Tool => "tool",
                _ => "user"
            };
            sb.Append($"<|im_start|>{role}\n{msg.Content}<|im_end|>\n");
        }
        if (addGenerationPrompt)
        {
            sb.Append("<|im_start|>assistant\n");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Sends a user message, executes generation, and yields simple text response strings.
    /// Filters internal thinking/reasoning chunks from the output stream.
    /// </summary>
    public async IAsyncEnumerable<string> ChatAsync(
        string userMessage,
        IInferenceParams? parameters = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var chunk in ChatChunksAsync(userMessage, parameters, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            if (chunk.Kind == GenerateChunkKind.Text)
            {
                yield return chunk.Text;
            }
        }
    }

    /// <summary>
    /// Sends a user message, executes generation, and yields typed <see cref="GenerateChunk"/> objects.
    /// Preserves thinking tokens (<see cref="GenerateChunkKind.Thinking"/>), final text, and usage metrics.
    /// </summary>
    public async IAsyncEnumerable<GenerateChunk> ChatChunksAsync(
        string userMessage,
        IInferenceParams? parameters = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(userMessage);
        int messageIndex = _history.Count;
        _history.AddUserMessage(userMessage);
        bool completed = false;
        try
        {
            string prompt = FormatPrompt(_history, addGenerationPrompt: true);
            string canonicalPrefix = FormatPrompt(_history, addGenerationPrompt: false);

            IInferenceParams? effectiveParams = parameters switch
            {
                InferenceParams ip => ip with { CanonicalHistoryPrefix = canonicalPrefix },
                null => new InferenceParams { CanonicalHistoryPrefix = canonicalPrefix },
                _ => parameters
            };

            var replyBuilder = new StringBuilder();

            await foreach (var chunk in _executor.InferChunksAsync(prompt, effectiveParams, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
            {
                if (chunk.Kind == GenerateChunkKind.Text)
                {
                    replyBuilder.Append(chunk.Text);
                }
                yield return chunk;
            }

            string fullReply = replyBuilder.ToString();
            if (!string.IsNullOrEmpty(fullReply))
            {
                _history.AddAssistantMessage(fullReply);
            }
            completed = true;
        }
        finally
        {
            if (!completed && _history.Count > messageIndex)
            {
                _history.RemoveAt(messageIndex);
            }
        }
    }

    /// <summary>
    /// Sends a message, executes generation, and yields simple text response strings.
    /// </summary>
    public async IAsyncEnumerable<string> ChatAsync(
        ChatMessage message,
        IInferenceParams? parameters = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var chunk in ChatChunksAsync(message, parameters, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            if (chunk.Kind == GenerateChunkKind.Text)
            {
                yield return chunk.Text;
            }
        }
    }

    /// <summary>
    /// Sends a message, executes generation, and yields typed <see cref="GenerateChunk"/> objects.
    /// </summary>
    public async IAsyncEnumerable<GenerateChunk> ChatChunksAsync(
        ChatMessage message,
        IInferenceParams? parameters = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(message);
        int messageIndex = _history.Count;
        _history.Add(message);
        bool completed = false;
        try
        {
            string prompt = FormatPrompt(_history, addGenerationPrompt: true);
            string canonicalPrefix = FormatPrompt(_history, addGenerationPrompt: false);

            IInferenceParams? effectiveParams = parameters switch
            {
                InferenceParams ip => ip with { CanonicalHistoryPrefix = canonicalPrefix },
                null => new InferenceParams { CanonicalHistoryPrefix = canonicalPrefix },
                _ => parameters
            };

            var replyBuilder = new StringBuilder();

            await foreach (var chunk in _executor.InferChunksAsync(prompt, effectiveParams, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
            {
                if (chunk.Kind == GenerateChunkKind.Text)
                {
                    replyBuilder.Append(chunk.Text);
                }
                yield return chunk;
            }

            string fullReply = replyBuilder.ToString();
            if (!string.IsNullOrEmpty(fullReply))
            {
                _history.AddAssistantMessage(fullReply);
            }
            completed = true;
        }
        finally
        {
            if (!completed && _history.Count > messageIndex)
            {
                _history.RemoveAt(messageIndex);
            }
        }
    }

    /// <summary>
    /// Generates assistant continuation for the existing history, yielding simple text strings.
    /// </summary>
    public async IAsyncEnumerable<string> ChatAsync(
        IInferenceParams? parameters = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var chunk in ChatChunksAsync(parameters, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            if (chunk.Kind == GenerateChunkKind.Text)
            {
                yield return chunk.Text;
            }
        }
    }

    /// <summary>
    /// Generates assistant continuation for the existing history, yielding typed <see cref="GenerateChunk"/> objects.
    /// </summary>
    public async IAsyncEnumerable<GenerateChunk> ChatChunksAsync(
        IInferenceParams? parameters = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        string prompt = FormatPrompt(_history);
        var replyBuilder = new StringBuilder();

        await foreach (var chunk in _executor.InferChunksAsync(prompt, parameters, cancellationToken).WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            if (chunk.Kind == GenerateChunkKind.Text)
            {
                replyBuilder.Append(chunk.Text);
            }
            yield return chunk;
        }

        string fullReply = replyBuilder.ToString();
        if (!string.IsNullOrEmpty(fullReply))
        {
            _history.AddAssistantMessage(fullReply);
        }
    }
}
