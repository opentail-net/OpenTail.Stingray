using OpenTail.TUI.Core;
using OpenTail.TUI.Runtime;
using OpenTail.TUI.Widgets;

namespace OpenTail.Stingray.Cli.Tui;

/// <summary>Immutable snapshot of an in-flight generation, handed to the render loop.</summary>
/// <param name="Text">Assistant text decoded so far.</param>
/// <param name="Tokens">Visible tokens emitted so far.</param>
internal sealed record GenerationProgress(string Text, int Tokens)
{
    internal static readonly GenerationProgress Empty = new("", 0);
}

/// <summary>Generation finished; carries the final text and timing for the transcript.</summary>
internal sealed record GenerationDoneMsg(string Text, int Tokens, double Seconds) : IMsg;

/// <summary>Generation failed; surfaced in the transcript rather than crashing the UI.</summary>
internal sealed record GenerationFailedMsg(string Message) : IMsg;

/// <summary>
/// Interactive chat screen. Generation runs as a background <see cref="Cmd"/> on the thread
/// pool and streams partial text through a <see cref="Published{T}"/> slot, which
/// <see cref="View"/> reads each frame — so the 30fps render loop never blocks on inference and
/// no lock is needed around the growing text.
/// </summary>
internal sealed class ChatModel : IModel
{
    private readonly ChatEngine _engine;
    private readonly Published<GenerationProgress> _progress = new(GenerationProgress.Empty);

    private readonly List<TranscriptEntry> _entries;
    private TextInputState _input = TextInputState.Empty;
    private ViewportState _scroll = ViewportState.Top;
    private bool _generating;
    private int _spinnerFrame;
    private string _status;

    internal ChatModel(ChatEngine engine, string modelName)
    {
        _engine = engine;
        _status = $"{modelName} — Enter to send, Ctrl+C to quit";
        _entries =
        [
            new TranscriptEntry(TranscriptRole.System,
                "Interactive chat. Type a message and press Enter. Ctrl+C quits."),
        ];
    }

    private ChatModel(ChatModel prior,
                      List<TranscriptEntry>? entries = null,
                      TextInputState? input = null,
                      ViewportState? scroll = null,
                      bool? generating = null,
                      int? spinnerFrame = null,
                      string? status = null)
    {
        _engine        = prior._engine;
        _progress      = prior._progress;
        _entries       = entries      ?? prior._entries;
        _input         = input        ?? prior._input;
        _scroll        = scroll       ?? prior._scroll;
        _generating    = generating   ?? prior._generating;
        _spinnerFrame  = spinnerFrame ?? prior._spinnerFrame;
        _status        = status       ?? prior._status;
    }

    public Cmd? Init() => null;

    public (IModel Model, Cmd? Cmd) Update(IMsg msg)
    {
        switch (msg)
        {
            case GenerationDoneMsg done:
            {
                var entries = new List<TranscriptEntry>(_entries)
                {
                    new(TranscriptRole.Assistant, done.Text.TrimEnd()),
                };
                _progress.Publish(GenerationProgress.Empty);

                double rate = done.Seconds > 0 ? done.Tokens / done.Seconds : 0;
                return (new ChatModel(this,
                            entries: entries,
                            generating: false,
                            status: $"{done.Tokens} tokens · {rate:F1} t/s"),
                        null);
            }

            case GenerationFailedMsg failed:
            {
                var entries = new List<TranscriptEntry>(_entries)
                {
                    new(TranscriptRole.Error, failed.Message),
                };
                _progress.Publish(GenerationProgress.Empty);
                return (new ChatModel(this, entries: entries, generating: false, status: "generation failed"), null);
            }

            // While generating, keep repainting so streamed tokens appear and the spinner turns.
            case TickMsg when _generating:
                return (new ChatModel(this, spinnerFrame: _spinnerFrame + 1), ScheduleTick());

            case TickMsg:
                return (this, null);

            case KeyMsg key:
                return HandleKey(key);

            default:
                return (this, null);
        }
    }

    private (IModel, Cmd?) HandleKey(KeyMsg key)
    {
        // Ctrl+C always quits, including mid-generation.
        if (key.Control && key.Key == ConsoleKey.C)
            return (this, Cmds.Quit());

        // Input is read-only while the model is producing tokens; scrolling still works.
        if (_generating)
            return (new ChatModel(this, scroll: _scroll.HandleKey(key, int.MaxValue, 1)), null);

        if (key.Key == ConsoleKey.Enter)
        {
            string prompt = _input.Value.Trim();
            if (prompt.Length == 0) return (this, null);

            if (prompt is "/exit" or "/quit")
                return (this, Cmds.Quit());

            var entries = new List<TranscriptEntry>(_entries)
            {
                new(TranscriptRole.User, prompt),
            };

            var next = new ChatModel(this,
                entries: entries,
                input: TextInputState.Empty,
                generating: true,
                status: "generating…");

            _progress.Publish(GenerationProgress.Empty);
            return (next, Cmds.Batch(Generate(prompt), ScheduleTick()));
        }

        // PageUp/PageDown scroll the transcript; everything else edits the input line.
        if (key.Key is ConsoleKey.PageUp or ConsoleKey.PageDown)
            return (new ChatModel(this, scroll: _scroll.HandleKey(key, int.MaxValue, 1)), null);

        return (new ChatModel(this, input: _input.HandleKey(key)), null);
    }

    private static Cmd ScheduleTick() =>
        Cmds.Tick(TimeSpan.FromMilliseconds(66), t => new TickMsg(t));

    /// <summary>
    /// Run one generation on the thread pool, publishing partial text as it decodes.
    /// Inference is synchronous and CPU-bound, so it goes through <see cref="Task.Run"/> to
    /// keep the message loop responsive.
    /// </summary>
    private Cmd Generate(string prompt) => async ct =>
    {
        try
        {
            return await Task.Run(() =>
            {
                var text = new System.Text.StringBuilder();
                var started = System.Diagnostics.Stopwatch.StartNew();

                int tokens = _engine.Generate(prompt, chunk =>
                {
                    text.Append(chunk);
                    // Immutable snapshot per publish: the reader may still hold the previous one.
                    _progress.Publish(new GenerationProgress(text.ToString(), 0));
                }, ct);

                return (IMsg)new GenerationDoneMsg(
                    text.ToString(), tokens, started.Elapsed.TotalSeconds);
            }, ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return null;
        }
        catch (Exception ex)
        {
            return new GenerationFailedMsg(ex.Message);
        }
    };

    public void View(Rect area, CellBuffer buffer)
    {
        if (area.Width <= 0 || area.Height <= 0) return;

        var rows = Layout.Vertical()
            .Constraints(new Constraint.Fill(), new Constraint.Length(1), new Constraint.Length(1))
            .Split(area);

        RenderTranscript(rows[0], buffer);
        RenderStatus(rows[1], buffer);
        RenderInput(rows[2], buffer);
    }

    private void RenderTranscript(Rect area, CellBuffer buffer)
    {
        var entries = _entries;

        // Splice the in-flight response in as a provisional assistant turn so streamed text
        // renders through the same transcript styling as a finished one.
        var live = _progress.Current;
        if (_generating && live.Text.Length > 0)
        {
            entries = new List<TranscriptEntry>(_entries)
            {
                new(TranscriptRole.Assistant, live.Text),
            };
        }

        // Pin to the live edge while generating so new tokens stay visible.
        var scroll = _generating ? ViewportState.Top : _scroll;
        int contentHeight = TranscriptView.MeasureHeight(entries, area.Width);
        if (_generating && contentHeight > area.Height)
            scroll = new ViewportState(contentHeight - area.Height);

        new TranscriptView(entries, scroll, Theme.Dark).Render(area, buffer);
    }

    private void RenderStatus(Rect area, CellBuffer buffer)
    {
        var style = Style.Default.WithForeground(Color.DarkGray);

        if (_generating)
        {
            new Spinner(_spinnerFrame, Style.Default.WithForeground(Color.Cyan))
                .Render(new Rect(area.X, area.Y, 1, 1), buffer);

            var rest = new Rect(area.X + 2, area.Y, Math.Max(0, area.Width - 2), 1);
            new Text(_status, style).Render(rest, buffer);
            return;
        }

        new Text(_status, style).Render(area, buffer);
    }

    private void RenderInput(Rect area, CellBuffer buffer)
    {
        var promptStyle = Style.Default.WithForeground(Color.Green).WithBold();
        new Text("> ", promptStyle).Render(new Rect(area.X, area.Y, 2, 1), buffer);

        var field = new Rect(area.X + 2, area.Y, Math.Max(0, area.Width - 2), 1);

        if (_generating)
        {
            new Text("…", Style.Default.WithForeground(Color.DarkGray)).Render(field, buffer);
            return;
        }

        new TextInput(_input, Style.Default, Style.Default.WithReverse())
            .Render(field, buffer);
    }
}
