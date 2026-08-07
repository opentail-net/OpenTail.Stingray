using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Cli.CommandLine;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Verifies every llama.cpp-compatibility alias in <c>docs/llamacpp-onramp-plan.md</c>
/// binds to the correct property, and every refused flag produces a message rather than
/// silently no-op-ing. Aliases are derived from llama.cpp's <c>common/arg.cpp</c>, not recall —
/// invented aliases that don't exist upstream are not tested here.
///
/// Tests are grouped by plan tier:
/// <list type="bullet">
///   <item>Tier 0 — pure aliases (template-string additions only, no new behaviour).</item>
///   <item>Tier 1 — cheap implementations where the capability already exists.</item>
///   <item>Refusals — recognised, explicitly rejected with a user-facing message.</item>
///   <item>Inert — accepted, warned about, but execution continues (-fa, --no-warmup).</item>
/// </list>
/// </summary>
public sealed class LlamaCompatAliasTests
{
    /// <summary>
    /// Binds against the REAL <see cref="RunCommand.Settings"/>, not a copy.
    /// </summary>
    /// <remarks>
    /// This previously bound a hand-written <c>TestSettings</c> replica of the option templates. That
    /// made the suite self-satisfying: it proved <see cref="OptionBinder"/> could parse the alias
    /// strings written in the test file, and nothing whatsoever about the templates the shipped CLI
    /// actually carries. A typo in <c>RunCommand.Settings</c> — exactly the failure these tests exist
    /// to catch, because it surfaces to users only as "unexpected argument" — would have left every
    /// test green. Bind the real type; there is no reason not to, since it is <c>public sealed</c>.
    /// </remarks>
    private static (RunCommand.Settings Settings, string? BindError) Bind(params string[] args)
    {
        var s = new RunCommand.Settings();
        bool ok = OptionBinder.TryBind(s, OptionModel.Describe<RunCommand.Settings>(), args, out string? err);
        return (s, ok ? null : err);
    }

    private static string? BindAndValidate(params string[] args)
    {
        var (s, bindErr) = Bind(args);
        return bindErr ?? s.Validate();
    }

    // ── Tier 0: -ngl ──────────────────────────────────────────────────────────

    [Theory]
    [InlineData("--ngl")]
    [InlineData("--n-gpu-layers")]
    [InlineData("--gpu-layers")]
    [InlineData("-g")]
    [InlineData("-ngl")]
    public void NglAliases_AllBindToNGpuLayers(string alias)
    {
        var (s, err) = Bind(alias, "20");
        Assert.Null(err);
        Assert.Equal(20, s.NGpuLayers);
    }

    // ── Tier 0: context size ───────────────────────────────────────────────────

    [Theory]
    [InlineData("-c")]
    [InlineData("--ctx-size")]
    [InlineData("-ctx")]
    [InlineData("--n-ctx")]
    public void CtxSizeAliases_AllBind(string alias)
    {
        var (s, err) = Bind(alias, "4096");
        Assert.Null(err);
        Assert.Equal(4096, s.CtxSize);
    }

    // ── Tier 0: n-predict ─────────────────────────────────────────────────────

    [Theory]
    [InlineData("-n")]
    [InlineData("--n-predict")]
    [InlineData("-npredict")]
    public void NPredictAliases_AllBind(string alias)
    {
        var (s, err) = Bind(alias, "256");
        Assert.Null(err);
        Assert.Equal(256, s.NPredict);
    }

    [Fact]
    public void NPredict_MinusOne_IsExplicitlyRefused()
    {
        string? err = BindAndValidate("-n", "-1");
        Assert.NotNull(err);
        Assert.Contains("until EOS", err, StringComparison.Ordinal);
    }

    // ── Tier 0: repeat-penalty ────────────────────────────────────────────────

    [Theory]
    [InlineData("--repeat-penalty")]
    [InlineData("--rep-penalty")]
    [InlineData("--repeat_penalty")]
    public void RepeatPenaltyAliases_AllBind(string alias)
    {
        var (s, err) = Bind(alias, "1.2");
        Assert.Null(err);
        Assert.Equal(1.2f, s.RepPenalty, precision: 4);
    }

    // ── Tier 0: -md / --draft ─────────────────────────────────────────────────

    [Theory]
    [InlineData("--model-draft")]
    [InlineData("--draft-model")]
    [InlineData("-md")]
    public void ModelDraftAliases_AllBind(string alias)
    {
        var (s, err) = Bind(alias, "draft.gguf");
        Assert.Null(err);
        Assert.Equal("draft.gguf", s.DraftModelPath);
    }

    [Theory]
    [InlineData("--spec-lookahead")]
    [InlineData("--draft-tokens")]
    [InlineData("--draft")]
    public void DraftLookaheadAliases_AllBind(string alias)
    {
        var (s, err) = Bind(alias, "8");
        Assert.Null(err);
        Assert.Equal(8, s.SpecLookahead);
    }

    // ── Tier 0: KV type ───────────────────────────────────────────────────────

    [Theory]
    [InlineData("--kv-type")]
    [InlineData("--cache-type-k")]
    [InlineData("-ctk")]
    public void KvTypeKAliases_AllBind(string alias)
    {
        var (s, err) = Bind(alias, "bf16");
        Assert.Null(err);
        Assert.Equal("bf16", s.KvTypeK);
    }

    [Theory]
    [InlineData("--cache-type-v")]
    [InlineData("-ctv")]
    public void KvTypeVAliases_AllBind(string alias)
    {
        var (s, err) = Bind(alias, "bf16");
        Assert.Null(err);
        Assert.Equal("bf16", s.KvTypeV);
    }

    [Fact]
    public void KvType_MatchingKAndV_PassesValidation()
    {
        string? err = BindAndValidate("-ctk", "bf16", "-ctv", "bf16");
        Assert.Null(err);
    }

    [Fact]
    public void KvType_DisagreeingKAndV_IsRefused()
    {
        string? err = BindAndValidate("-ctk", "bf16", "-ctv", "fp32");
        Assert.NotNull(err);
        Assert.Contains("-ctk", err, StringComparison.Ordinal);
        Assert.Contains("-ctv", err, StringComparison.Ordinal);
    }

    // ── Tier 1: -t / --threads ────────────────────────────────────────────────

    [Theory]
    [InlineData("-t")]
    [InlineData("--threads")]
    public void ThreadsAliases_AllBind(string alias)
    {
        var (s, err) = Bind(alias, "8");
        Assert.Null(err);
        Assert.Equal(8, s.Threads);
    }

    [Fact]
    public void Threads_Default_IsZero()
    {
        var (s, _) = Bind();
        Assert.Equal(0, s.Threads);
    }

    // ── Tier 1: --repeat-last-n ───────────────────────────────────────────────

    [Fact]
    public void RepeatLastN_BindsValue()
    {
        var (s, err) = Bind("--repeat-last-n", "256");
        Assert.Null(err);
        Assert.Equal(256, s.RepeatLastN);
    }

    [Fact]
    public void RepeatLastN_Default_Is64()
    {
        var (s, _) = Bind();
        Assert.Equal(64, s.RepeatLastN);
    }

    [Fact]
    public void RepeatLastN_Zero_Disables()
    {
        var (s, err) = Bind("--repeat-last-n", "0");
        Assert.Null(err);
        Assert.Equal(0, s.RepeatLastN);
    }

    [Fact]
    public void RepeatLastN_MinusOne_MeansFullContext()
    {
        var (s, err) = Bind("--repeat-last-n", "-1");
        Assert.Null(err);
        Assert.Equal(-1, s.RepeatLastN);
    }

    // ── Tier 1: -e / --escape ─────────────────────────────────────────────────

    [Theory]
    [InlineData("-e")]
    [InlineData("--escape")]
    public void EscapeAliases_SetFlagTrue(string alias)
    {
        var (s, err) = Bind(alias);
        Assert.Null(err);
        Assert.True(s.Escape);
    }

    // ── Tier 1: --logit-bias ──────────────────────────────────────────────────

    [Fact]
    public void LogitBias_SingleEntry_Binds()
    {
        var (s, err) = Bind("--logit-bias", "1234+1.5");
        Assert.Null(err);
        Assert.NotNull(s.LogitBias);
        Assert.Equal(["1234+1.5"], s.LogitBias);
    }

    [Fact]
    public void LogitBias_MultipleEntries_AccumulatesAll()
    {
        var (s, err) = Bind("--logit-bias", "1+10", "--logit-bias", "2-100");
        Assert.Null(err);
        Assert.NotNull(s.LogitBias);
        Assert.Equal(["1+10", "2-100"], s.LogitBias);
    }

    // ── Tier 1: --chat-template ───────────────────────────────────────────────

    [Fact]
    public void ChatTemplate_RawJinja_Binds()
    {
        const string src = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
        var (s, err) = Bind("--chat-template", src);
        Assert.Null(err);
        Assert.Equal(src, s.ChatTemplateOverride);
    }

    // ── Inert flags: bind OK, Validate() returns null (warned in Execute) ─────

    [Theory]
    [InlineData("-fa")]
    [InlineData("--flash-attn")]
    public void FlashAttn_BindsAndPassesValidation(string flag)
    {
        // -fa is inert (attention already fused). Validate() must NOT refuse it —
        // the Execute() path emits a warning and continues so common llama.cpp
        // command lines work without editing.
        string? err = BindAndValidate(flag);
        Assert.Null(err);
    }

    [Fact]
    public void NoWarmup_BindsAndPassesValidation()
    {
        string? err = BindAndValidate("--no-warmup");
        Assert.Null(err);
    }

    // ── Refusals: Validate() returns a non-null message ───────────────────────

    [Theory]
    [InlineData("-ts",            "0.0,0.0")]
    [InlineData("--tensor-split", "0.0,0.0")]
    public void TensorSplit_Refused(string flag, string value)
    {
        string? err = BindAndValidate(flag, value);
        Assert.NotNull(err);
        Assert.Contains("-ts", err, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("-sm",          "none")]
    [InlineData("--split-mode", "row")]
    public void SplitMode_Refused(string flag, string value)
    {
        string? err = BindAndValidate(flag, value);
        Assert.NotNull(err);
        Assert.Contains("-sm", err, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("-mg",        "0")]
    [InlineData("--main-gpu", "1")]
    public void MainGpu_Refused(string flag, string value)
    {
        string? err = BindAndValidate(flag, value);
        Assert.NotNull(err);
        Assert.Contains("--device", err, StringComparison.Ordinal);
    }

    [Fact] public void Mlock_Refused()   { Assert.NotNull(BindAndValidate("--mlock")); }
    [Fact] public void NoMmap_Refused()  { Assert.NotNull(BindAndValidate("--no-mmap")); }

    [Fact]
    public void Numa_Refused()
    {
        string? err = BindAndValidate("--numa", "distribute");
        Assert.NotNull(err);
        Assert.Contains("--numa", err, StringComparison.Ordinal);
    }

    [Fact]
    public void PresencePenalty_Refused()
    {
        string? err = BindAndValidate("--presence-penalty", "0.5");
        Assert.NotNull(err);
        Assert.Contains("--repeat-penalty", err, StringComparison.Ordinal);
    }

    [Fact]
    public void FrequencyPenalty_Refused()
    {
        string? err = BindAndValidate("--frequency-penalty", "0.5");
        Assert.NotNull(err);
        Assert.Contains("--repeat-penalty", err, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("-b",            "512")]
    [InlineData("--batch-size",  "512")]
    public void BatchSize_Refused(string flag, string value)
    {
        string? err = BindAndValidate(flag, value);
        Assert.NotNull(err);
        Assert.Contains("-b", err, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("-ub",            "128")]
    [InlineData("--ubatch-size",  "128")]
    public void UBatchSize_Refused(string flag, string value)
    {
        string? err = BindAndValidate(flag, value);
        Assert.NotNull(err);
        Assert.Contains("-ub", err, StringComparison.Ordinal);
    }

    // ── Exhaustive: every refused flag must produce a non-null error ──────────

    [Theory]
    [InlineData("-ts",             "0.0")]
    [InlineData("--tensor-split",  "0.5,0.5")]
    [InlineData("-sm",             "none")]
    [InlineData("--split-mode",    "row")]
    [InlineData("-mg",             "0")]
    [InlineData("--main-gpu",      "1")]
    [InlineData("--numa",          "distribute")]
    [InlineData("--presence-penalty",  "0.5")]
    [InlineData("--frequency-penalty", "0.5")]
    [InlineData("-b",              "512")]
    [InlineData("--batch-size",    "512")]
    [InlineData("-ub",             "128")]
    [InlineData("--ubatch-size",   "128")]
    public void RefusedValueFlags_NeverSilentlySucceed(string flag, string value)
    {
        Assert.NotNull(BindAndValidate(flag, value));
    }

    [Theory]
    [InlineData("--mlock")]
    [InlineData("--no-mmap")]
    public void RefusedBoolFlags_NeverSilentlySucceed(string flag)
    {
        Assert.NotNull(BindAndValidate(flag));
    }

    // ── Inert flags must NOT be in the refused sets ───────────────────────────

    [Theory]
    [InlineData("-fa")]
    [InlineData("--flash-attn")]
    [InlineData("--no-warmup")]
    public void InertFlags_AreNotRefused(string flag)
    {
        // These warn and continue; a null error here is correct.
        Assert.Null(BindAndValidate(flag));
    }
}
