
namespace OpenTail.Stingray.Bench;

/// <summary>
/// Perf-loop iteration 9 (docs/perf-loop-progress.md): <see cref="SmolLM2CpuBenchmarks.
/// PrefillBatched"/> always prefills the same tiny fixed "Hi" chat-template prompt (its
/// <c>TokenCount</c> [Params] sweep only feeds <c>DecodeTokens</c>), and its default
/// <c>WarmupCount(1)</c> is short of the ~9 calls <c>cpu-prefill-repack-gemm-plan.md</c> §29
/// already established `MatMulBatched` needs to reach steady state. This exists to get a real,
/// properly warmed-up, realistic-token-count prefill number, using the SAME filler-text prompt
/// construction as <c>scripts/bench-vs-llamacpp.ps1</c> (~4.7 chars/token) so it's directly
/// comparable to the fresh <c>llama-bench</c> pp64 result already measured this session
/// (204.99 ± 7.12 t/s, this exact box, this exact model, 12 threads).
/// </summary>
[MemoryDiagnoser]
[WarmupCount(20)] // well past the documented ~9-call steady-state threshold
[IterationCount(6)] // matches this session's own n=6 minimum-sample rule
public class PrefillWarmupBenchmark
{
    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    // 64 matches the llama-bench pp64 run already done this session; 256/903 match this
    // project's own established prompt fixtures from cpu-prefill-plan.md/bench-vs-llamacpp.ps1.
    [Params(64, 256, 903)]
    public int TokenCount { get; set; }

    private const string Filler =
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn while several " +
        "birds circle overhead searching for food among the tall reeds and scattered stones. ";

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("SmolLM2-1.7B-Instruct-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp);

        // ~4.7 chars/token, empirically -- same constant bench-vs-llamacpp.ps1 uses, so both
        // tools land close to the requested token count with the same filler text.
        int chars = (int)Math.Ceiling(TokenCount * 4.7);
        var text = new System.Text.StringBuilder();
        while (text.Length < chars) text.Append(Filler);
        var promptText = text.ToString(0, chars);

        var tok = GgufTokenizer.FromGgufModel(_model);
        _promptTokens = tok.Encode(promptText);
    }

    [IterationSetup]
    public void IterSetup() => _fwd.Cache.Reset();

    [Benchmark(Description = "SmolLM2 Prefill (warmed, real prompt)")]
    public int PrefillBatched()
    {
        var logits = _fwd.Prefill(_promptTokens);
        return Sampler.Greedy(logits);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _model.Dispose();
    }
}
