using System.Text;
using OpenTail.Stingray.Cli;

// Force UTF-8 for stdin/stdout. On Windows the console defaults to the OEM
// code page, which mangles multi-byte UTF-8 output (CJK, emoji, smart quotes)
// into '?' or replacement glyphs.
Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding = Encoding.UTF8;

// Warn about STINGRAY_* variables the engine never reads. They are consumed ad hoc at ~141
// call sites, so a misspelling is indistinguishable from "unset" â the run silently ignores the
// user's configuration. Warn rather than fail: an unknown name may legitimately belong to a
// different OpenTail version, and refusing to start over it would be worse than the typo.
foreach (string unknown in KnownEnvironmentVariables.FindUnknown())
{
    string? suggestion = KnownEnvironmentVariables.SuggestClosest(unknown);
    Console.Error.WriteLine(suggestion is null
        ? $"warning: {unknown} is set but is not read by this build â it will have no effect."
        : $"warning: {unknown} is set but is not read by this build â did you mean {suggestion}?");
}

var app = new CommandApp<RunCommand>();
app.Configure(config =>
{
    config.SetApplicationName("opentail-llm-cli");
    // Report the MinVer-derived version rather than a literal. The hardcoded "0.1.0" that was
    // here never changed with the build, so `--version` was actively misleading â and build
    // identity is the first thing any bug report or support bundle needs to be right.
    // Generated from the SDK's InformationalVersion at build time, avoiding reflection metadata
    // that NativeAOT may trim from the published executable.
    config.SetApplicationVersion(StingrayBuildVersion.Value);
    config.AddCommand<ListMetadataCommand>("list-metadata")
        .WithDescription("Print all GGUF metadata key/value pairs from a model file");
    config.AddCommand<ListEnvCommand>("list-env")
        .WithDescription("Print the STINGRAY_* environment settings active in this process, flagging any the engine does not read");
    config.AddCommand<ShowTemplateCommand>("show-template")
        .WithDescription("Render a model's chat template against a sample conversation (or --raw for the source)");
    config.AddCommand<ListModelsCommand>("list-models")
        .WithDescription("List GGUF model files on disk, optionally opening each index (--deep)");
    config.AddCommand<ListTensorsCommand>("list-tensors")
        .WithDescription("Print the tensor index (name, dtype, shape, bytes) from a model file");
    config.AddCommand<StaticPlanCommand>("plan")
        .WithDescription("Read-only GGUF compatibility, hardware availability, and placement plan report");
    config.AddCommand<InspectCommand>("inspect")
        .WithDescription("Read-only GGUF identity, compatibility, and capability report without placement planning");
    config.AddCommand<CapabilitiesCommand>("capabilities")
        .WithDescription("Show which model-package profiles, dtypes and backends are supported");
    config.AddCommand<DoctorCommand>("doctor")
        .WithDescription("Check local runtime, backends, and optionally a GGUF model without inference");
    config.AddCommand<StatusCommand>("status")
        .WithDescription("View live runtime status, throughput, KV occupancy, and queue metrics from a running server");
    config.AddCommand<InspectKvCommand>("inspect-kv")
        .WithDescription("Inspect KV cache capacity, page distribution, forking and CoW statistics");
    config.AddCommand<PerplexityCommand>("perplexity")
        .WithDescription("Teacher-forced perplexity over a text file (llama.cpp llama-perplexity analogue; CPU only). Reports mean NLL, perplexity, and position-bucket NLLs â the TurboQuant/KVarN accuracy gate (issue #180).");
    config.AddCommand<ImageCommand>("image")
        .WithDescription("Generate an image from a text prompt using a native FLUX or Z-Image-Turbo diffusion pipeline (VAE + CLIP-L + T5-XXL + DiT GGUF). See 'opentail-llm-cli image --help' for required model paths.");
    config.AddCommand<TtsCommand>("tts")
        .WithDescription("Synthesize high-quality speech audio from text using native Kokoro-82M TTS.");
    config.AddCommand<SttCommand>("stt")
        .WithDescription("Transcribe or translate speech audio into text and timestamps using native OpenAI Whisper.");
    config.AddCommand<EmbedCommand>("embed")
        .WithDescription("Generate dense semantic vector embeddings for text with pooling and Matryoshka support.");
    config.AddCommand<RerankCommand>("rerank")
        .WithDescription("Score and rerank candidate documents by relevance against a search query.");
});

return app.Run(args);
