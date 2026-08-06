using System.Diagnostics.CodeAnalysis;

namespace OpenTail.Stingray.Cli.CommandLine;

/// <summary>A named sub-command registration.</summary>
internal sealed class CommandEntry
{
    internal required string Name { get; init; }
    internal required ICommand Instance { get; init; }
    internal string Description { get; set; } = "";
}

/// <summary>Fluent handle returned by <see cref="Configurator.AddCommand{TCommand}"/>.</summary>
internal sealed class CommandConfigurator
{
    private readonly CommandEntry _entry;

    internal CommandConfigurator(CommandEntry entry) => _entry = entry;

    /// <summary>Set the one-line description shown in the command list.</summary>
    public CommandConfigurator WithDescription(string description)
    {
        _entry.Description = description;
        return this;
    }
}

/// <summary>Configures the application's name, version, and sub-commands.</summary>
internal sealed class Configurator
{
    internal string ApplicationName { get; private set; } = "app";
    internal string ApplicationVersion { get; private set; } = "0.0.0";
    internal List<CommandEntry> Commands { get; } = [];

    /// <summary>Set the executable name shown in usage output.</summary>
    public void SetApplicationName(string name) => ApplicationName = name;

    /// <summary>Set the version reported by <c>--version</c>.</summary>
    public void SetApplicationVersion(string version) => ApplicationVersion = version;

    /// <summary>Register a named sub-command.</summary>
    public CommandConfigurator AddCommand<TCommand>(string name)
        where TCommand : ICommand, new()
    {
        var entry = new CommandEntry { Name = name, Instance = new TCommand() };
        Commands.Add(entry);
        return new CommandConfigurator(entry);
    }
}

/// <summary>
/// Command-line host: routes argv to a sub-command or the default command, and owns
/// <c>--help</c> / <c>--version</c>.
/// </summary>
/// <typeparam name="TDefault">Command that runs when no sub-command name is given.</typeparam>
internal sealed class CommandApp<
    [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties)] TDefault>
    where TDefault : ICommand, new()
{
    private readonly Configurator _config = new();
    private readonly TDefault _default = new();

    /// <summary>Apply configuration (name, version, sub-commands).</summary>
    public void Configure(Action<Configurator> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        configure(_config);
    }

    /// <summary>Parse <paramref name="args"/> and run. Returns the process exit code.</summary>
    public int Run(string[] args)
    {
        // Ctrl+C cancels the running command rather than killing the process mid-write.
        using var cts = new CancellationTokenSource();
        ConsoleCancelEventHandler onCancel = (_, e) => { e.Cancel = true; cts.Cancel(); };
        Console.CancelKeyPress += onCancel;

        try
        {
            return Dispatch(args, cts.Token);
        }
        catch (OperationCanceledException)
        {
            return 130;     // 128 + SIGINT, the conventional shell exit code.
        }
        finally
        {
            Console.CancelKeyPress -= onCancel;
        }
    }

    private int Dispatch(string[] args, CancellationToken cancellation)
    {
        // A sub-command name, when present, is always the first token.
        CommandEntry? sub = args.Length > 0
            ? _config.Commands.FirstOrDefault(c => c.Name == args[0])
            : null;

        string[] rest = sub is null ? args : args[1..];

        if (rest.Any(a => a is "--help" or "-h"))
        {
            if (sub is null) HelpRenderer.WriteRootHelp(_config, _default);
            else             HelpRenderer.WriteCommandHelp(_config, sub);
            return 0;
        }

        if (sub is null && rest.Any(a => a is "--version" or "-v"))
        {
            Terminal.AnsiConsole.WriteLine(_config.ApplicationVersion);
            return 0;
        }

        ICommand target = sub?.Instance ?? _default;
        return target.Run(rest, cancellation);
    }
}
