// The fluent method StatusBuilder.Spinner(...) shadows the Spinner type inside that class,
// so refer to the type through an alias.
using SpinnerAnimation = OpenTail.Stingray.Cli.Terminal.Spinner;

namespace OpenTail.Stingray.Cli.Terminal;

/// <summary>
/// Context handed to a status callback so long-running work can update its label.
/// </summary>
public sealed class StatusContext
{
    private readonly StatusBuilder _owner;

    internal StatusContext(StatusBuilder owner) => _owner = owner;

    /// <summary>Update the status label shown next to the spinner.</summary>
    public void Status(string label) => _owner.UpdateLabel(label);
}

/// <summary>
/// Runs an action while showing a spinner and label on stderr.
///
/// <para>
/// The animation goes to <b>stderr</b> and is suppressed entirely when stderr is redirected,
/// so it never contaminates piped stdout. When animation is off the label is still printed
/// once, so non-interactive logs keep the progress narrative.
/// </para>
/// </summary>
public sealed class StatusBuilder
{
    private SpinnerAnimation _spinner = SpinnerAnimation.Known.Dots;
    private volatile string _label = "";
    private int _maxRendered;

    /// <summary>Select the spinner animation.</summary>
    public StatusBuilder Spinner(SpinnerAnimation spinner) { _spinner = spinner; return this; }

    /// <summary>Set the spinner style. Accepted for call-site compatibility.</summary>
    public StatusBuilder SpinnerStyle(Style _) => this;

    internal void UpdateLabel(string label) => _label = label;

    /// <summary>Run <paramref name="work"/> while displaying <paramref name="label"/>.</summary>
    public void Start(string label, Action<StatusContext> work)
    {
        ArgumentNullException.ThrowIfNull(work);

        _label = label;
        var ctx = new StatusContext(this);

        // No animation when stderr is not a terminal: emit the label once and just run.
        if (Console.IsErrorRedirected || !AnsiConsole.ColorEnabled)
        {
            Console.Error.WriteLine(label);
            work(ctx);
            return;
        }

        using var done = new ManualResetEventSlim(false);
        var animation = Task.Run(() => Animate(done));

        try
        {
            work(ctx);
        }
        finally
        {
            done.Set();
            animation.Wait();
            ClearLine();
        }
    }

    private void Animate(ManualResetEventSlim done)
    {
        int frame = 0;
        var frames = _spinner.Frames;

        while (!done.IsSet)
        {
            string line = $"{frames[frame % frames.Length]} {_label}";
            frame++;

            // Pad to erase the tail of a previously longer label.
            _maxRendered = Math.Max(_maxRendered, line.Length);
            Console.Error.Write("\r" + line.PadRight(_maxRendered));

            done.Wait(80);
        }
    }

    private void ClearLine()
    {
        if (_maxRendered > 0)
            Console.Error.Write("\r" + new string(' ', _maxRendered) + "\r");
    }
}
