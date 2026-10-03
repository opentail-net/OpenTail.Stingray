namespace OpenTail.Stingray.Core.Catalog;

/// <summary>One file of a catalog entry: where it lives on Hugging Face and what it must hash to.</summary>
/// <param name="Repo">Hugging Face repo id, e.g. <c>Qwen/Qwen2.5-0.5B-Instruct-GGUF</c>.</param>
/// <param name="Revision">Full 40-character commit the file is pinned to, never a branch name: a floating <c>main</c> can change under a pinned SHA-256.</param>
/// <param name="RepoPath">Path of the file inside the repo (may contain folders).</param>
/// <param name="Sha256">Lower-case hex SHA-256 of the file, as the HF tree API reports it for LFS files.</param>
/// <param name="Size">Exact size in bytes.</param>
public sealed record CatalogFile(string Repo, string Revision, string RepoPath, string Sha256, long Size)
{
    /// <summary>File name the file is stored under in the model home (the last path segment).</summary>
    public string FileName => RepoPath[(RepoPath.LastIndexOf('/') + 1)..];

    /// <summary>Server root; only tests point it elsewhere.</summary>
    internal string BaseUrl { get; init; } = "https://huggingface.co";

    /// <summary>Direct download URL, at the pinned <see cref="Revision"/>.</summary>
    public string Url => $"{BaseUrl}/{Repo}/resolve/{Revision}/{RepoPath}?download=true";
}

/// <summary>
/// One ready-to-use bundle: every file a task command needs, with the evidence that it works.
/// </summary>
/// <param name="Id">Short stable id, e.g. <c>qwen2.5-0.5b</c>.</param>
/// <param name="Task">The task it serves (<see cref="ModelCatalog.Tasks"/>).</param>
/// <param name="Why">One user-level line: why pick this one.</param>
/// <param name="Files">All files of the bundle; the first one is what task commands are pointed at.</param>
/// <param name="Licence">Licence summary, with any caveat a user must read before use.</param>
/// <param name="LicenceNeedsConsent">True when the licence is not plainly permissive: setup asks even with <c>--yes</c> unless <c>--accept-licence</c> is given.</param>
/// <param name="Hardware">RAM / GPU needs.</param>
/// <param name="Speed">A measured speed note, with the machine it was measured on.</param>
/// <param name="Evidence">Where the proof that these exact files work lives (test, sample, status row).</param>
/// <param name="RunTemplate">Command that uses the bundle; <c>{0}</c> is replaced by the main file's local path.</param>
public sealed record CatalogEntry(
    string Id,
    string Task,
    string Why,
    IReadOnlyList<CatalogFile> Files,
    string Licence,
    bool LicenceNeedsConsent,
    string Hardware,
    string Speed,
    string Evidence,
    string RunTemplate)
{
    /// <summary>Total download size in bytes.</summary>
    public long TotalSize => Files.Sum(f => f.Size);

    /// <summary>The file task commands are pointed at.</summary>
    public CatalogFile MainFile => Files[0];

    /// <summary>The run command for a bundle installed under <paramref name="home"/>.</summary>
    public string RunCommand(ModelHome home) => string.Format(System.Globalization.CultureInfo.InvariantCulture, RunTemplate, Quote(home.PathOf(MainFile)));

    private static string Quote(string path) => path.Contains(' ') ? $"\"{path}\"" : path;
}

/// <summary>
/// The recommended model per task: a small, curated list, not the coverage list (that is
/// <c>docs/STATUS.md</c>; anything else still works through <c>pull -r</c> / <c>-m</c>).
///
/// <para>Rules for adding an entry (<c>docs/3-product-and-runtime/103-front-door-design.md</c>):
/// a public source with the SHA-256 Hugging Face reports, a green status row, and a test or
/// compiled sample that ran on those exact files. One default per task (the first entry for that
/// task), at most three alternatives, each justified by a user-level reason.</para>
///
/// <para>A static C# table rather than an embedded JSON file: it needs no serializer, so it stays
/// NativeAOT/trim safe, and a typo is a compile error.</para>
/// </summary>
public static class ModelCatalog
{
    /// <summary>Tasks in display order.</summary>
    public static readonly IReadOnlyList<string> Tasks = ["chat", "speak", "transcribe"];

    /// <summary>All entries. The first entry for each task is that task's default.</summary>
    public static readonly IReadOnlyList<CatalogEntry> Entries =
    [
        // sha256 values: HF tree API lfs.oid (2026-09-28), matching the files the README quick start and
        // samples/QuickStart ran on. Revisions: the repo head on 2026-10-03; for each file the X-Linked-ETag (LFS
        // sha256) and Content-Length served at that commit were re-checked against the values here.
        new(
            Id: "qwen2.5-0.5b",
            Task: "chat",
            Why: "Small and quick; a first chat model that runs on any CPU.",
            Files:
            [
                new("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "9217f5db79a29953eb74d5343926648285ec7e67", "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                    "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db", 491_400_032),
            ],
            Licence: "Apache-2.0",
            LicenceNeedsConsent: false,
            Hardware: "about 1 GB RAM, CPU only",
            Speed: "about 23 tokens/s on a Ryzen 7 5700G CPU (2026-09-28)",
            Evidence: "README quick start, samples/QuickStart",
            RunTemplate: "stingray -m {0} -p \"Hello\""),

        new(
            Id: "piper-lessac",
            Task: "speak",
            Why: "Fast, clear US English voice.",
            Files:
            [
                new("rhasspy/piper-voices", "c10ece1aade47bb51c153c893d14e5bf8e5b7117", "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                    "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f", 63_201_294),
                // Not an LFS file; hashed from the downloaded file (2026-09-28).
                new("rhasspy/piper-voices", "c10ece1aade47bb51c153c893d14e5bf8e5b7117", "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
                    "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0", 4_885),
            ],
            Licence: "MIT (code); the voice was trained on the Lessac Blizzard 2013 data, whose own licence applies: "
                + "https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html",
            LicenceNeedsConsent: true,
            Hardware: "about 200 MB RAM, CPU only",
            Speed: "2.9 s of audio in 1.1-1.35 s on a Ryzen 7 5700G CPU (2026-09-28)",
            Evidence: "README 'Speak and listen'",
            RunTemplate: "stingray tts -e piper -m {0} -t \"Hello!\" -o hello.wav"),

        new(
            Id: "whisper-base",
            Task: "transcribe",
            Why: "Accurate multilingual transcription at a small size.",
            Files:
            [
                new("ggerganov/whisper.cpp", "5359861c739e955e79d9a303bcbc70fb988958b1", "ggml-base.bin",
                    "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe", 147_951_465),
            ],
            Licence: "MIT",
            LicenceNeedsConsent: false,
            Hardware: "about 500 MB RAM, CPU only",
            Speed: "a 2.9 s clip in 1.0-1.5 s on a Ryzen 7 5700G CPU (2026-09-28)",
            Evidence: "README 'Speak and listen' (word-exact round trip of the Piper sample)",
            RunTemplate: "stingray stt -m base --model-file {0} -i hello.wav"),
    ];

    /// <summary>Finds an entry by id, or the default entry for a task name. Case-insensitive.</summary>
    public static CatalogEntry? Find(string idOrTask)
    {
        foreach (var e in Entries)
            if (e.Id.Equals(idOrTask, StringComparison.OrdinalIgnoreCase)) return e;
        return DefaultFor(idOrTask);
    }

    /// <summary>The default entry for a task, or null for an unknown task.</summary>
    public static CatalogEntry? DefaultFor(string task)
    {
        foreach (var e in Entries)
            if (e.Task.Equals(task, StringComparison.OrdinalIgnoreCase)) return e;
        return null;
    }

    /// <summary>All entries for a task, default first.</summary>
    public static IEnumerable<CatalogEntry> ForTask(string task) =>
        Entries.Where(e => e.Task.Equals(task, StringComparison.OrdinalIgnoreCase));
}
