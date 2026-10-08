using System.ComponentModel;
using System.Diagnostics;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Captures a golden reference (docs/2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md) from the vendored llama.cpp:
/// the prompt ids come from <c>llama-tokenize</c> (an oracle independent of our tokenizer, which is then cross-checked against it), the
/// continuation from a short-lived <c>llama-server</c> given those ids and greedy sampling. The result is a small JSON file holding
/// identity and evidence only (bare file name, size, SHA-256, build, settings, token ids), never a path to this machine.
/// No Python: the whole flow is C#, as per the project rule.
/// </summary>
public sealed class CaptureGoldenCommand : Command<CaptureGoldenCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model <PATH>")]
        [Description("GGUF to capture a reference for")]
        public string ModelPath { get; init; } = "";

        [CommandOption("--prompt <TEXT>")]
        [Description("Raw prompt text, tokenized by llama-tokenize (the model's own BOS behaviour applies)")]
        public string Prompt { get; init; } = "The capital of France is";

        [CommandOption("--prompt-ids <IDS>")]
        [Description("Comma-separated prompt token ids, used as-is instead of tokenizing --prompt")]
        public string? PromptIds { get; init; }

        [CommandOption("-n|--tokens <N>")]
        [Description("Tokens to generate")]
        public int Tokens { get; init; } = 24;

        [CommandOption("--mode <MODE>")]
        [Description("How the golden is checked later: free (greedy run compared until it diverges) or teacherForced (reference tokens fed, every position compared)")]
        public string Mode { get; init; } = "free";

        [CommandOption("--case <NAME>")]
        [Description("Name of this case inside the golden (an existing case of the same name is replaced)")]
        public string CaseName { get; init; } = "default";

        [CommandOption("-o|--out <PATH>")]
        [Description("Golden file to write (default: ./<architecture>.golden.json); an existing file for the same model gets the case added")]
        public string? Out { get; init; }

        [CommandOption("--source <REPO>")]
        [Description("Where the checkpoint came from (Hugging Face repo id), recorded in the golden")]
        public string? Source { get; init; }

        [CommandOption("--notes <TEXT>")]
        [Description("Provenance notes recorded in the golden (no paths: the file is checked for machine-specific text)")]
        public string? Notes { get; init; }

        [CommandOption("--server <EXE>")]
        [Description("llama-server executable (default: tools/llama.cpp/llama-server.exe found upward from here)")]
        public string? ServerExe { get; init; }

        [CommandOption("--tokenizer <EXE>")]
        [Description("llama-tokenize executable (default: tools/llama.cpp/llama-tokenize.exe found upward from here)")]
        public string? TokenizerExe { get; init; }

        [CommandOption("--threads <N>")]
        public int Threads { get; init; } = 4;

        [CommandOption("--ctx-size <N>")]
        public int CtxSize { get; init; } = 1024;

        [CommandOption("--server-timeout <SECONDS>")]
        [Description("How long to wait for llama-server to load the model")]
        public int ServerTimeoutSeconds { get; init; } = 900;

        [CommandOption("--expect <LIST>")]
        [Description("Hyperparameter guards to pin in the golden, e.g. ropeDim=16,numExperts=8,hasFfnBias=true (ModelHyperparams property=value, comma separated)")]
        public string? Expect { get; init; }

        [CommandOption("--min-confident <N>")]
        [Description("Require at least N matches at positions where the reference itself was confident (evidence requirement for teacher-forced cases)")]
        public int MinConfident { get; init; }

        [CommandOption("--no-hash")]
        [Description("Skip the SHA-256 of the model (the golden then cannot pin the file; slow for large checkpoints)")]
        public bool NoHash { get; init; }
    }

    private static string? FindTool(string? explicitPath, string exeName)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath)) return File.Exists(explicitPath) ? Path.GetFullPath(explicitPath) : null;
        foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            string? dir = start;
            for (int i = 0; i < 10 && dir is not null; i++)
            {
                string candidate = Path.Combine(dir, "tools", "llama.cpp", exeName);
                if (File.Exists(candidate)) return candidate;
                dir = Path.GetDirectoryName(dir);
            }
        }
        return null;
    }

    private static async Task<(int exit, string stdout, string stderr)> RunToolAsync(string exe, IEnumerable<string> args, TimeSpan timeout)
    {
        var psi = new ProcessStartInfo(exe) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true };
        foreach (var a in args) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi) ?? throw new InvalidOperationException($"could not start {exe}");
        var outTask = p.StandardOutput.ReadToEndAsync();
        var errTask = p.StandardError.ReadToEndAsync();
        using var cts = new CancellationTokenSource(timeout);
        try { await p.WaitForExitAsync(cts.Token); }
        catch (OperationCanceledException) { TryKill(p); throw new TimeoutException($"{Path.GetFileName(exe)} did not finish within {timeout.TotalSeconds:F0}s"); }
        return (p.ExitCode, await outTask, await errTask);
    }

    private static void TryKill(Process p)
    {
        try { if (!p.HasExited) p.Kill(entireProcessTree: true); } catch { /* already gone */ }
    }

    private static int FreePort()
    {
        var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        try { return ((IPEndPoint)l.LocalEndpoint).Port; } finally { l.Stop(); }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation) =>
        RunAsync(settings, cancellation).GetAwaiter().GetResult();

    private static async Task<int> RunAsync(Settings s, CancellationToken cancellation)
    {
        if (!ModelPathResolver.TryRequireModelFile(s.ModelPath, out int modelFileExit)) return modelFileExit;
        if (s.Tokens < 1) { AnsiConsole.ErrorLine("[red]Error:[/] --tokens must be at least 1."); return 1; }
        bool teacher = string.Equals(s.Mode, "teacherForced", StringComparison.OrdinalIgnoreCase) || string.Equals(s.Mode, "teacher", StringComparison.OrdinalIgnoreCase);
        if (!teacher && !string.Equals(s.Mode, "free", StringComparison.OrdinalIgnoreCase))
        { AnsiConsole.ErrorLine("[red]Error:[/] --mode must be free or teacherForced."); return 1; }

        string? server = FindTool(s.ServerExe, "llama-server.exe");
        string? tokenizerExe = s.PromptIds is null ? FindTool(s.TokenizerExe, "llama-tokenize.exe") : null;
        if (server is null)
        { AnsiConsole.ErrorLine("[red]Error:[/] llama-server.exe not found. It ships in tools/llama.cpp; pass --server <exe> to point at another build."); return 1; }
        if (s.PromptIds is null && tokenizerExe is null)
        { AnsiConsole.ErrorLine("[red]Error:[/] llama-tokenize.exe not found in tools/llama.cpp. Pass --tokenizer <exe>, or --prompt-ids to skip tokenizing."); return 1; }

        string modelPath = Path.GetFullPath(s.ModelPath);
        string fileName = Path.GetFileName(modelPath);
        using var model = OpenTail.Stingray.Core.GgufModel.Open(modelPath);
        string arch = model.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a) ?? "" : "";
        AnsiConsole.MarkupLine($"[bold]Model:[/] {Markup.Escape(fileName)}  [bold]architecture:[/] {Markup.Escape(arch)}");

        // One heavy process at a time on the machine (same gate as the heavy test suites): capture loads the model into a second process.
        using var gate = new Mutex(false, @"Global\OpenTailStingray.HeavyTests");
        bool gotGate = false;
        try { gotGate = gate.WaitOne(TimeSpan.FromSeconds(1)); }
        catch (AbandonedMutexException) { gotGate = true; }
        if (!gotGate)
        {
            AnsiConsole.MarkupLine("[yellow]Another heavy test/capture process is running; waiting for it (up to 10 minutes)...[/]");
            try { gotGate = gate.WaitOne(TimeSpan.FromMinutes(10)); }
            catch (AbandonedMutexException) { gotGate = true; }
            if (!gotGate) { AnsiConsole.ErrorLine("[red]Error:[/] timed out waiting for the machine-wide heavy-process gate."); return 1; }
        }

        Process? srv = null;
        ConsoleCancelEventHandler onCancel = (_, e) => { if (srv is not null) TryKill(srv); };
        Console.CancelKeyPress += onCancel;
        try
        {
            // 1. Fingerprint
            long size = new FileInfo(modelPath).Length;
            string? sha = null;
            if (!s.NoHash)
            {
                AnsiConsole.MarkupLine($"Hashing {size / 1048576.0:F0} MiB (SHA-256)...");
                var fp = ModelFingerprinter.Compute(modelPath);
                sha = fp.Sha256;
                if (fp.FromCache) AnsiConsole.MarkupLine("  [dim](from the sidecar cache; size and modification time unchanged)[/]");
                AnsiConsole.MarkupLine($"  sha256 {sha}");
            }
            else AnsiConsole.MarkupLine("[yellow]--no-hash:[/] the golden will not pin the checkpoint.");

            // 2. Prompt ids: the oracle, or as given.
            int[] promptIds;
            string oracle;
            if (s.PromptIds is { Length: > 0 })
            {
                promptIds = s.PromptIds.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
                oracle = "given";
            }
            else
            {
                var (exit, stdout, stderr) = await RunToolAsync(tokenizerExe!, ["-m", modelPath, "--ids", "--log-disable", "-p", s.Prompt], TimeSpan.FromMinutes(5));
                if (exit != 0) { AnsiConsole.ErrorLine($"[red]llama-tokenize failed (exit {exit}):[/] {Markup.Escape(stderr.Trim())}"); return 1; }
                promptIds = GoldenCaptureParsing.ParseTokenizeIds(stdout);
                oracle = "llama-tokenize";
                // Cross-check our own tokenizer against the oracle (a separate question from the forward pass).
                try
                {
                    var ours = OpenTail.Stingray.Core.GgufTokenizer.FromGgufModel(model);
                    var ids = ours.Encode(s.Prompt).ToList();
                    if (ours.AddBosToken && ours.BosTokenId >= 0 && (ids.Count == 0 || ids[0] != ours.BosTokenId)) ids.Insert(0, ours.BosTokenId);
                    AnsiConsole.MarkupLine(ids.SequenceEqual(promptIds)
                        ? "[green]Our tokenizer agrees with llama-tokenize on the prompt.[/]"
                        : $"[yellow]Our tokenizer DISAGREES with llama-tokenize[/] (ours [{string.Join(",", ids)}] vs oracle [{string.Join(",", promptIds)}]); the golden uses the oracle ids.");
                }
                catch (Exception ex) { AnsiConsole.MarkupLine($"[dim]Tokenizer cross-check skipped: {Markup.Escape(ex.Message)}[/]"); }
            }
            if (promptIds.Length == 0) { AnsiConsole.ErrorLine("[red]Error:[/] the prompt produced no tokens."); return 1; }
            AnsiConsole.MarkupLine($"Prompt ids ({oracle}): [{string.Join(",", promptIds)}]");

            // 3. Build string
            string? build = null;
            try { var v = await RunToolAsync(server, ["--version"], TimeSpan.FromSeconds(30)); build = GoldenCaptureParsing.ParseVersion(v.stdout + "\n" + v.stderr); }
            catch (Exception ex) when (ex is TimeoutException or System.ComponentModel.Win32Exception) { /* provenance only */ }

            // 4. Start llama-server and wait for it.
            int port = FreePort();
            var psi = new ProcessStartInfo(server) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true };
            foreach (var arg in new[] { "-m", modelPath, "--port", port.ToString(), "-ngl", "0", "-c", s.CtxSize.ToString(), "-t", s.Threads.ToString(), "--host", "127.0.0.1" })
                psi.ArgumentList.Add(arg);
            srv = Process.Start(psi) ?? throw new InvalidOperationException("could not start llama-server");
            var serverLog = new StringBuilder();
            srv.OutputDataReceived += (_, e) => { if (e.Data is not null && serverLog.Length < 8000) serverLog.AppendLine(e.Data); };
            srv.ErrorDataReceived += (_, e) => { if (e.Data is not null && serverLog.Length < 8000) serverLog.AppendLine(e.Data); };
            srv.BeginOutputReadLine();
            srv.BeginErrorReadLine();
            AnsiConsole.MarkupLine($"Started llama-server on 127.0.0.1:{port}; loading the model...");

            using var http = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromMinutes(30) };
            var deadline = DateTime.UtcNow.AddSeconds(s.ServerTimeoutSeconds);
            bool healthy = false;
            while (DateTime.UtcNow < deadline && !cancellation.IsCancellationRequested)
            {
                if (srv.HasExited) { AnsiConsole.ErrorLine($"[red]llama-server exited early (code {srv.ExitCode}).[/]\n{Markup.Escape(serverLog.ToString())}"); return 1; }
                try
                {
                    using var r = await http.GetAsync("/health", cancellation);
                    if (r.IsSuccessStatusCode && (await r.Content.ReadAsStringAsync(cancellation)).Contains("\"ok\"")) { healthy = true; break; }
                }
                catch (HttpRequestException) { /* still starting */ }
                await Task.Delay(1000, cancellation);
            }
            if (!healthy) { AnsiConsole.ErrorLine($"[red]llama-server did not become healthy within {s.ServerTimeoutSeconds}s.[/]\n{Markup.Escape(serverLog.ToString())}"); return 1; }

            // 5. The reference completion.
            using var body = new StringContent(GoldenCaptureParsing.BuildCompletionRequest(promptIds, s.Tokens), Encoding.UTF8, "application/json");
            using var resp = await http.PostAsync("/completion", body, cancellation);
            string json = await resp.Content.ReadAsStringAsync(cancellation);
            if (!resp.IsSuccessStatusCode) { AnsiConsole.ErrorLine($"[red]/completion failed ({(int)resp.StatusCode}):[/] {Markup.Escape(json)}"); return 1; }
            var completion = GoldenCaptureParsing.ParseCompletion(json);
            if (completion.TokensEvaluated >= 0 && completion.TokensEvaluated != promptIds.Length)
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] the server evaluated {completion.TokensEvaluated} prompt tokens but {promptIds.Length} ids were sent (it may have added or dropped a token).");
            if (completion.Tokens.Length < s.Tokens)
                AnsiConsole.MarkupLine($"[yellow]Note:[/] the reference stopped after {completion.Tokens.Length} of {s.Tokens} tokens (end of text?).");

            TryKill(srv);

            // 6. Assemble and write. Machine-specific text is refused by GoldenFile.Save.
            string outPath = s.Out ?? $"{arch}.golden.json";
            Dictionary<string, string>? expectations = s.Expect is { Length: > 0 }
                ? s.Expect.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(p => p.Split('=', 2)).ToDictionary(p => p[0], p => p.Length > 1 ? p[1] : "true", StringComparer.OrdinalIgnoreCase)
                : null;
            var newCase = new GoldenCase
            {
                Name = s.CaseName,
                PromptText = s.PromptIds is null ? s.Prompt : null,
                PromptTokens = promptIds,
                TokenizerOracle = oracle,
                NPredict = s.Tokens,
                Tokens = completion.Tokens,
                Margins = completion.Margins,
                MinConfident = s.MinConfident,
                Text = completion.Content,
                Mode = teacher ? "teacherForced" : "free",
            };
            GoldenFile golden;
            if (File.Exists(outPath))
            {
                var existing = GoldenFile.Load(outPath);
                if (!string.Equals(existing.Model.FileName, fileName, StringComparison.OrdinalIgnoreCase)
                    || (existing.Model.Sha256 is not null && sha is not null && existing.Model.Sha256 != sha))
                { AnsiConsole.ErrorLine($"[red]Error:[/] '{Markup.Escape(outPath)}' is a golden for a different model file; refusing to mix them. Pick another --out."); return 1; }
                golden = existing with { Cases = [.. existing.Cases.Where(c => c.Name != s.CaseName), newCase], ExpectedHyperparameters = expectations ?? existing.ExpectedHyperparameters, Notes = s.Notes ?? existing.Notes };
            }
            else
            {
                golden = new GoldenFile
                {
                    Architecture = arch,
                    Notes = s.Notes,
                    Model = new GoldenModel { FileName = fileName, SizeBytes = size, Sha256 = sha, Source = s.Source },
                    Reference = new GoldenReference
                    {
                        Engine = "llama-server",
                        Build = build,
                        Settings = $"-ngl 0 -c {s.CtxSize} -t {s.Threads}",
                        Sampling = new GoldenSampling { Temperature = 0, TopK = 1, Seed = 0, RepeatPenalty = 1.0, CachePrompt = false },
                        CapturedUtc = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                    },
                    Cases = [newCase],
                    ExpectedHyperparameters = expectations,
                };
            }
            golden.Save(outPath);

            AnsiConsole.MarkupLine($"[green bold]Captured[/] {completion.Tokens.Length} tokens -> {Markup.Escape(outPath)}");
            AnsiConsole.MarkupLine($"  text: {Markup.Escape(completion.Content.Replace("\n", "\\n"))}");
            AnsiConsole.MarkupLine("  Check it with: [yellow]stingray admit-arch -m <model> --golden " + Markup.Escape(outPath) + "[/]");
            return 0;
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("machine-specific"))
        {
            AnsiConsole.ErrorLine($"[red]Refusing to write:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
        finally
        {
            Console.CancelKeyPress -= onCancel;
            if (srv is not null) { TryKill(srv); srv.Dispose(); }
            try { gate.ReleaseMutex(); } catch (ApplicationException) { /* not owned */ }
        }
    }
}
