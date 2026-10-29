using System.Globalization;

namespace OpenTail.Stingray.Cli;

public sealed partial class ImageCommand
{
    // ── sd-cli fallback (for research comparison) ─────────────────────────

    private static int RunSdCpp(Settings s)
    {
        string? sdCli = FindSdCli(s.SdCliPath);
        if (sdCli is null)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] sd-cli not found. Set [yellow]STINGRAY_SDCPP[/] or place binary in [cyan]tools/sd-cli.exe[/].");
            AnsiConsole.MarkupLine("Download: [link]https://github.com/leejet/stable-diffusion.cpp/releases[/]");
            return 1;
        }

        var modelPath = ResolveModelPath(s.ModelPath);
        if (modelPath is null || string.IsNullOrWhiteSpace(s.Prompt))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] Model (-m) and prompt (-p) are required.");
            return 1;
        }

        string output = s.OutputPath ?? "output.png";
        int steps     = s.Steps > 0 ? s.Steps : IsDistilled(modelPath) ? 4 : 20;
        float cfg     = s.CfgScale >= 0f ? s.CfgScale : IsFlowMatching(modelPath) ? 1.0f : 3.5f;

        var args = BuildSdCppArgs(modelPath, output, steps, cfg, s);

        AnsiConsole.MarkupLine($"[dim](sd-cli mode)[/] {Markup.Escape(sdCli)}");

        var psi = new ProcessStartInfo(sdCli, args) { UseShellExecute = false,
            RedirectStandardOutput = true, RedirectStandardError = true };
        using var proc = Process.Start(psi);
        if (proc is null) { AnsiConsole.ErrorLine("[red]Error:[/] Failed to launch sd-cli."); return 1; }

        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        AnsiConsole.Status().Spinner(Spinner.Known.Dots).SpinnerStyle(Style.Parse("blue"))
            .Start($"Generating {s.Width}×{s.Height} image…", _ => proc.WaitForExit());

        if (proc.ExitCode != 0)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] sd-cli exited with code {proc.ExitCode}.");
            return proc.ExitCode;
        }
        AnsiConsole.MarkupLine($"[green]✓[/] Image saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/]");
        return 0;
    }

    private static string BuildSdCppArgs(string modelPath, string output, int steps, float cfg, Settings s)
    {
        var parts = new List<string> { "--diffusion-model", Q(modelPath) };
        if (s.VaePath is not null)            { parts.Add("--vae");   parts.Add(Q(s.VaePath)); }
        if (s.T5XXLPath is not null)          { parts.Add("--t5xxl"); parts.Add(Q(s.T5XXLPath)); }
        if (s.ClipLPath is not null)          { parts.Add("--clip_l"); parts.Add(Q(s.ClipLPath)); }
        if (s.QwenEncoderPath is not null)    { parts.Add("--llm");   parts.Add(Q(s.QwenEncoderPath)); }
        if (s.TextEncoderPath is not null)    { parts.Add("--llm");   parts.Add(Q(s.TextEncoderPath)); }
        parts.Add("-p"); parts.Add(Q(s.Prompt!));
        if (!string.IsNullOrWhiteSpace(s.NegativePrompt)) { parts.Add("-n"); parts.Add(Q(s.NegativePrompt)); }
        parts.Add("-W"); parts.Add(s.Width.ToString());
        parts.Add("-H"); parts.Add(s.Height.ToString());
        parts.Add("--steps"); parts.Add(steps.ToString());
        parts.Add("--cfg-scale"); parts.Add(cfg.ToString("F1", CultureInfo.InvariantCulture));
        if (s.Seed != -1) { parts.Add("-s"); parts.Add(s.Seed.ToString()); }
        parts.Add("-o"); parts.Add(Q(output));
        return string.Join(" ", parts);
    }

    private static string? FindSdCli(string? explicit_)
    {
        if (explicit_ is not null) return File.Exists(explicit_) ? explicit_ : null;
        var env = Environment.GetEnvironmentVariable("STINGRAY_SDCPP");
        if (env is not null && File.Exists(env)) return env;
        string[] names = OperatingSystem.IsWindows() ? ["sd-cli.exe", "sd.exe"] : ["sd-cli", "sd"];
        foreach (var b in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
            foreach (var sub in new[] { "tools", "tools/sd", "." })
                foreach (var n in names) { var p = Path.Combine(b, sub, n); if (File.Exists(p)) return p; }
        var pathVar = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var dir in pathVar.Split(Path.PathSeparator))
            foreach (var n in names) { var p = Path.Combine(dir, n); if (File.Exists(p)) return p; }
        return null;
    }

    private static string Q(string s) =>
        s.Contains(' ') || s.Contains('"') ? $"\"{s.Replace("\"", "\\\"")}\"" : s;
}
