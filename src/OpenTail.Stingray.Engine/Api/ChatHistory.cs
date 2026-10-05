#nullable enable

using System.Collections;

namespace OpenTail.Stingray;

/// <summary>
/// Author role in a multi-turn conversation turn.
/// Mirrors LLamaSharp's <c>AuthorRole</c>.
/// </summary>
public enum AuthorRole
{
    /// <summary>System instruction or preamble guiding the model's behavior.</summary>
    System,

    /// <summary>End-user input turn.</summary>
    User,

    /// <summary>Model assistant response turn.</summary>
    Assistant,

    /// <summary>External tool execution result.</summary>
    Tool
}

/// <summary>
/// Single message in a multi-turn conversation history.
/// </summary>
/// <param name="Role">Author role of the message.</param>
/// <param name="Content">Text content of the message.</param>
public record ChatMessage(AuthorRole Role, string Content)
{
    /// <summary>Creates a system message.</summary>
    public static ChatMessage System(string content) => new(AuthorRole.System, content);

    /// <summary>Creates a user message.</summary>
    public static ChatMessage User(string content) => new(AuthorRole.User, content);

    /// <summary>Creates an assistant message.</summary>
    public static ChatMessage Assistant(string content) => new(AuthorRole.Assistant, content);

    /// <summary>Creates a tool message.</summary>
    public static ChatMessage Tool(string content) => new(AuthorRole.Tool, content);
}

/// <summary>
/// Ordered collection of conversational messages.
/// Mirrors LLamaSharp's <c>ChatHistory</c>.
/// </summary>
public class ChatHistory : IReadOnlyList<ChatMessage>
{
    private readonly List<ChatMessage> _messages;

    /// <summary>Initializes an empty chat history.</summary>
    public ChatHistory()
    {
        _messages = new List<ChatMessage>();
    }

    /// <summary>Initializes a chat history populated with the specified messages.</summary>
    public ChatHistory(IEnumerable<ChatMessage> messages)
    {
        ArgumentNullException.ThrowIfNull(messages);
        _messages = new List<ChatMessage>(messages);
    }

    /// <inheritdoc/>
    public int Count => _messages.Count;

    /// <inheritdoc/>
    public ChatMessage this[int index] => _messages[index];

    /// <summary>Appends a message to the history.</summary>
    public void Add(ChatMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        _messages.Add(message);
    }

    /// <summary>Appends a message with the specified role and content.</summary>
    public void AddMessage(AuthorRole role, string content)
    {
        ArgumentNullException.ThrowIfNull(content);
        Add(new ChatMessage(role, content));
    }

    /// <summary>Appends a system message to the history.</summary>
    public void AddSystemMessage(string content) => AddMessage(AuthorRole.System, content);

    /// <summary>Appends a user message to the history.</summary>
    public void AddUserMessage(string content) => AddMessage(AuthorRole.User, content);

    /// <summary>Appends an assistant message to the history.</summary>
    public void AddAssistantMessage(string content) => AddMessage(AuthorRole.Assistant, content);

    /// <summary>Appends a tool message to the history.</summary>
    public void AddToolMessage(string content) => AddMessage(AuthorRole.Tool, content);

    /// <summary>Clears all messages from the history.</summary>
    public void Clear() => _messages.Clear();

    /// <inheritdoc/>
    public IEnumerator<ChatMessage> GetEnumerator() => _messages.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
