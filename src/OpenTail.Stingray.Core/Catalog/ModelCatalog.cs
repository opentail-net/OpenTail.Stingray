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
/// <param name="FamilyId">Entries sharing a family id are the same model line at different sizes or quantisations; a fallback never leaves the family.</param>
/// <param name="Qualifications">What was actually checked, structured so code can read it (the free-text <paramref name="Evidence"/> stays for humans).</param>
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
    string RunTemplate,
    string? FamilyId = null,
    IReadOnlyList<CatalogQualification>? Qualifications = null)
{
    /// <summary>What was qualified for this bundle (never null). Empty means unqualified: shown as such, never as known-good.</summary>
    public IReadOnlyList<CatalogQualification> Checked => Qualifications ?? [];

    /// <summary>The qualification for a backend and capability (e.g. <c>cpu</c>, <c>chat</c>), or null when none was recorded.</summary>
    public CatalogQualification? QualificationFor(string backend, string capability) =>
        Checked.FirstOrDefault(q => q.Backend.Equals(backend, StringComparison.OrdinalIgnoreCase) && q.Capability.Equals(capability, StringComparison.OrdinalIgnoreCase));

    /// <summary>Total download size in bytes.</summary>
    public long TotalSize => Files.Sum(f => f.Size);

    /// <summary>The file task commands are pointed at.</summary>
    public CatalogFile MainFile => Files[0];

    /// <summary>The run command for a bundle installed under <paramref name="home"/>.</summary>
    public string RunCommand(ModelHome home) => RunTemplate.Contains("{0}")
        ? string.Format(System.Globalization.CultureInfo.InvariantCulture, RunTemplate, Quote(home.PathOf(MainFile)))
        : RunTemplate;

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
            Speed: "about 62-75 tokens/s decode (128 tokens, 5 runs) on a Ryzen 7 5700G CPU (2026-10-10)",
            Evidence: "README quick start, samples/QuickStart; llama.cpp golden Exact 32/32 (2026-10-09)",
            RunTemplate: "stingray chat",
            FamilyId: "qwen2.5-instruct",
            Qualifications: [Cpu("qwen2.5-0.5b", "Exact, 32/32 tokens")]),

        new(
            Id: "qwen2.5-1.5b",
            Task: "chat",
            Why: "Better reasoning and instruction following while staying under 2 GB download.",
            Files:
            [
                new("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "91cad51170dc346986eccefdc2dd33a9da36ead9", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                    "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e", 1_117_320_736),
            ],
            Licence: "Apache-2.0",
            LicenceNeedsConsent: false,
            Hardware: "about 2.5 GiB RAM, CPU only",
            Speed: "about 33 tokens/s decode on a Ryzen 7 5700G CPU (one golden run, 2026-10-09)",
            Evidence: "llama.cpp golden NearTie 31/32 (2026-10-09); scout pretest stages 0-3 passed 2026-10-09",
            RunTemplate: "stingray chat",
            FamilyId: "qwen2.5-instruct",
            Qualifications: [Cpu("qwen2.5-1.5b", "NearTie, 31/32 tokens")]),

        new(
            Id: "qwen2.5-7b",
            Task: "chat",
            Why: "Full 7B capability for complex reasoning and coding; sharded into two files.",
            Files:
            [
                new("Qwen/Qwen2.5-7B-Instruct-GGUF", "bb5d59e06d9551d752d08b292a50eb208b07ab1f", "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                    "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db", 3_993_201_344),
                new("Qwen/Qwen2.5-7B-Instruct-GGUF", "bb5d59e06d9551d752d08b292a50eb208b07ab1f", "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
                    "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a", 689_872_288),
            ],
            Licence: "Apache-2.0",
            LicenceNeedsConsent: false,
            Hardware: "about 9 GiB RAM, CPU only",
            Speed: "about 8 tokens/s decode on a Ryzen 7 5700G CPU (one golden run, 2026-10-09)",
            Evidence: "llama.cpp golden NearTie 31/32 (2026-10-09); scout pretest stages 0-3 passed 2026-10-09",
            RunTemplate: "stingray chat",
            FamilyId: "qwen2.5-instruct",
            Qualifications: [Cpu("qwen2.5-7b", "NearTie, 31/32 tokens")]),

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
            RunTemplate: "stingray speak \"Hello from Stingray.\""),

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
            RunTemplate: "stingray transcribe <audio.wav>"),
    ];

    private static CatalogQualification Cpu(string golden, string result) => new("cpu", "chat",
        $"tests/OpenTail.Stingray.Tests.ForwardPass/Goldens/{golden}.golden.json", result, "2026-10-09", "llama-server 10306 (6b5c2efb4)");

    /// <summary>
    /// Smaller entries of the same model family that were qualified for <paramref name="backend"/> and this entry's task, largest first.
    /// A fallback never leaves the family and never offers an unqualified look-alike.
    /// </summary>
    public static IEnumerable<CatalogEntry> SmallerQualifiedAlternatives(CatalogEntry entry, string backend = "cpu") =>
        entry.FamilyId is null
            ? []
            : Entries
                .Where(e => !ReferenceEquals(e, entry) && e.Task == entry.Task && e.FamilyId == entry.FamilyId
                    && e.TotalSize < entry.TotalSize && e.QualificationFor(backend, e.Task) is not null)
                .OrderByDescending(e => e.TotalSize);

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

/// <summary>
/// One scoped claim: on <paramref name="Backend"/>, for <paramref name="Capability"/>, these exact files reproduced a reference.
/// It says nothing about other backends, capabilities, contexts or prompts.
/// </summary>
/// <param name="Backend">Where it ran: <c>cpu</c>.</param>
/// <param name="Capability">What was exercised: <c>chat</c>.</param>
/// <param name="Golden">Repo-relative path of the recorded llama.cpp golden; a test fails when it is missing or pins different files.</param>
/// <param name="Result">The verdict in words, e.g. <c>Exact, 32/32 tokens</c>.</param>
/// <param name="Date">ISO date of the run.</param>
/// <param name="Engine">Reference engine and build the golden was captured with.</param>
public sealed record CatalogQualification(string Backend, string Capability, string Golden, string Result, string Date, string Engine);
