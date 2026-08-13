using OpenTail.Stingray.Core.Tools;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Built-in reference skill for deterministic testing of native skill composition,
/// prompt instruction injection, and tool definition registration.
/// </summary>
public static class EchoSkill
{
    /// <summary>
    /// The default Echo instruction ("Always prefix your response with 'ECHO:'").
    /// </summary>
    public static readonly IInstruction EchoInstruction = new Instruction("Always prefix your response with 'ECHO:'", "echo-instruction");

    /// <summary>
    /// The default Echo tool definition (<c>echo_value(value)</c>).
    /// </summary>
    public static readonly ITool EchoTool = new Tool("echo_value", "Echoes a value back to the caller.");

    /// <summary>
    /// Creates an instance of the <see cref="EchoSkill"/>.
    /// </summary>
    public static ISkill Create() => new Skill(
        Name: "echo",
        Description: "Deterministic reference skill for testing skill composition.",
        Instructions: new[] { EchoInstruction },
        Tools: new[] { EchoTool },
        Resources: new[] { new Resource("echo-ref", "text/plain", "Echo reference documentation") }
    );
}
