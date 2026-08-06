using System.Diagnostics.CodeAnalysis;

namespace OpenTail.Stingray.Cli.CommandLine;

/// <summary>
/// Declares the command-line aliases that bind to a settings property.
/// </summary>
/// <remarks>
/// The template is pipe-separated aliases with an optional value placeholder:
/// <c>"-m|--model"</c>, <c>"--tools &lt;PATH&gt;"</c>, <c>"--ngl|--n-gpu-layers|-g"</c>.
///
/// <para>
/// Unlike Spectre.Console.Cli, a single-dash alias may be longer than one character, so
/// llama.cpp-compatible forms such as <c>-cmoe</c> and <c>-jf</c> are expressible.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
public sealed class CommandOptionAttribute(string template) : Attribute
{
    /// <summary>The raw template, e.g. <c>"-m|--model &lt;PATH&gt;"</c>.</summary>
    public string Template { get; } = template;
}

/// <summary>Base class for a command's parsed options.</summary>
public abstract class CommandSettings
{
    /// <summary>
    /// Validate cross-option constraints after binding. Return an error message to abort,
    /// or null when the settings are coherent.
    /// </summary>
    public virtual string? Validate() => null;
}

/// <summary>Non-generic handle the app uses to dispatch without knowing the settings type.</summary>
internal interface ICommand
{
    /// <summary>Bind <paramref name="args"/> and run. Returns the process exit code.</summary>
    int Run(string[] args, CancellationToken cancellation);

    /// <summary>Options this command accepts, for help rendering.</summary>
    IReadOnlyList<OptionModel> Options { get; }
}

/// <summary>
/// A command with strongly-typed settings. Implementations declare their options as
/// properties on <typeparamref name="TSettings"/> annotated with
/// <see cref="CommandOptionAttribute"/>.
/// </summary>
public abstract class Command<
    [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties)] TSettings>
    : ICommand
    where TSettings : CommandSettings, new()
{
    /// <summary>Run the command against its bound settings.</summary>
    protected abstract int Execute(TSettings settings, CancellationToken cancellation);

    IReadOnlyList<OptionModel> ICommand.Options => OptionModel.Describe<TSettings>();

    int ICommand.Run(string[] args, CancellationToken cancellation)
    {
        var settings = new TSettings();

        if (!OptionBinder.TryBind(settings, OptionModel.Describe<TSettings>(), args, out string? error))
        {
            Terminal.AnsiConsole.MarkupLine($"[red]Error:[/] {Terminal.Markup.Escape(error!)}");
            return 1;
        }

        if (settings.Validate() is { } validationError)
        {
            Terminal.AnsiConsole.MarkupLine($"[red]Error:[/] {Terminal.Markup.Escape(validationError)}");
            return 1;
        }

        return Execute(settings, cancellation);
    }
}
