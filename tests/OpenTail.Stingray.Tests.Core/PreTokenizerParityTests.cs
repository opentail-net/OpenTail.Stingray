using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Token-ID parity against llama.cpp for each locally available <c>tokenizer.ggml.pre</c> value.
/// <para>
/// This is the only acceptance bar that detects a wrong pre-tokenizer split. A wrong split does not
/// throw, does not produce invalid tokens, and does not fail a round-trip — <c>Decode(Encode(s))</c>
/// still returns <c>s</c> because the pieces reassemble. It shows up only as degraded output. So the
/// test compares against IDs captured from a reference implementation, not against invariants.
/// </para>
/// <para>
/// Reference IDs were captured with <c>tools/llama.cpp/llama-tokenize.exe</c> build <c>b8585-cpu</c>:
/// <c>llama-tokenize.exe -m &lt;model&gt; -p "&lt;probe&gt;" --ids --no-bos</c>.
/// </para>
/// <para>
/// The probe is chosen for the axis these regexes actually disagree on: digit runs. GPT-2 groups
/// them greedily (<c>\p{N}+</c>); the <c>qwen2</c> and <c>smollm</c> pre-types emit one token per
/// digit (<c>\p{N}</c>). Prose alone cannot separate the two, which is why the existing suites pass
/// over this.
/// </para>
/// </summary>
public sealed class PreTokenizerParityTests
{
    private const string DigitProbe = "Sum 1234567890 and 42.";

    /// <summary>
    /// One row per pre-type with a local fixture. <c>vocabSize</c> guards against a same-pre-type
    /// model with a different vocabulary being picked up, which would fail with confusing IDs
    /// rather than reporting that the fixture is not the one these IDs came from.
    /// </summary>
    public static TheoryData<string, int, int[]> DigitProbeReferences() => new()
    {
        // Qwen3 family (Qwen3-0.6B-Q8_0, Qwen3-8B-Q4_K_M). llama.cpp: LLAMA_VOCAB_PRE_TYPE_QWEN2.
        { "qwen2", 151936, new[] { 9190, 220, 16, 17, 18, 19, 20, 21, 22, 23, 24, 15, 323, 220, 19, 17, 13 } },
        // SmolLM2 family. llama.cpp: LLAMA_VOCAB_PRE_TYPE_SMOLLM.
        { "smollm", 49152, new[] { 13764, 216, 33, 34, 35, 36, 37, 38, 39, 40, 41, 32, 284, 216, 36, 34, 30 } },
        // OLMoE. llama.cpp maps LLAMA_VOCAB_PRE_TYPE_OLMO onto the GPT-2 regex, so this row is the
        // control: it is expected to pass both before and after the pre-type table lands, and it
        // fails only if a change breaks the GPT-2 default that most models rely on.
        { "olmo", 50304, new[] { 11808, 1249, 16767, 25025, 2270, 285, 5976, 15 } },
        // Llama-3 family (orpheus-3b-0.1-ft, a Llama-3-tokenizer TTS checkpoint). llama.cpp:
        // LLAMA_VOCAB_PRE_TYPE_LLAMA3. Was unverified against a real model until this row — see
        // docs/01-gguf-model-coverage-plan.md §3 remaining-work item 2. Reference captured with
        // tools/llama.cpp/llama-tokenize.exe build b8585-cpu.
        { "llama-bpe", 156940, new[] { 9370, 220, 4513, 10961, 16474, 15, 323, 220, 2983, 13 } },
        // Qwen-3.5 family (Qwen3.8-27B-UD-Q3_K_XL). llama.cpp: LLAMA_VOCAB_PRE_TYPE_QWEN2 variant
        // with combining-mark handling. Was unverified against a real model until this row — same
        // gap as llama-bpe above.
        { "qwen35", 248320, new[] { 8917, 220, 16, 17, 18, 19, 20, 21, 22, 23, 24, 15, 321, 220, 19, 17, 13 } },
    };

    [Theory]
    [MemberData(nameof(DigitProbeReferences))]
    public void Encode_DigitProbe_MatchesLlamaCppTokenIds(string pre, int vocabSize, int[] expected)
    {
        var tokenizer = FindTokenizerByPre(pre);
        // Skip, never return. A silent return reports as a pass, and a parity suite that passes
        // without comparing anything is worse than no suite at all.
        Assert.SkipWhen(tokenizer is null, $"No local GGUF declares tokenizer.ggml.pre = '{pre}'.");
        Assert.SkipUnless(tokenizer!.VocabSize == vocabSize,
            $"Fixture for pre '{pre}' has vocab {tokenizer.VocabSize}, not the {vocabSize} these reference IDs came from.");

        var actual = tokenizer.Encode(DigitProbe);

        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// Probes chosen for where the <c>qwen2</c> regex differs from GPT-2 in a way the merge table
    /// cannot mask — unlike digits, whose apparent divergence the vocabulary already neutralises.
    /// <list type="bullet">
    /// <item>Uppercase contraction: GPT-2 matches only lowercase <c>'s</c>; qwen2 is case-insensitive.</item>
    /// <item>Punctuation-prefixed word: qwen2's <c>[^\r\n\p{L}\p{N}]?\p{L}+</c> lets any single
    /// non-alphanumeric attach to the following word; GPT-2's <c>?\p{L}+</c> allows only a space.</item>
    /// <item>Multi-space run and a bare digit pair, as controls.</item>
    /// </list>
    /// </summary>
    public static TheoryData<string, int, string, int[]> RegexDivergenceProbes() => new()
    {
        { "qwen2", 151936, "IT'S",    new[] { 952, 13272 } },
        { "qwen2", 151936, "(hello)", new[] { 3203, 4791, 8 } },
        { "qwen2", 151936, "«mot",    new[] { 23703, 46828 } },
        { "qwen2", 151936, "12",      new[] { 16, 17 } },
        { "qwen2", 151936, "a  b",    new[] { 64, 220, 293 } },

        // SmolLM2. Its cascade is two stages (digits, then GPT-2), and llama.cpp keeps a run of
        // n spaces as a single token — it does NOT decompose into repeated single-space tokens.
        // A prior test asserted that 8 spaces must yield more tokens than 4; the oracle says both
        // yield two, so that assertion described the old CodeGenTokenizer path rather than SmolLM2.
        { "smollm", 49152, "    X",     new[] { 333, 2273 } },
        { "smollm", 49152, "        X", new[] { 415, 2273 } },

        // Llama-3 (orpheus-3b-0.1-ft). Same case-insensitive-contraction and
        // punctuation-attachment axes as qwen2, plus one llama3-specific axis: digit runs are
        // capped at 3 characters (\p{N}{1,3}) instead of qwen2's single-digit \p{N} or GPT-2's
        // ungrouped \p{N}+ — "12345" must split as "123"+"45", not one token or five.
        { "llama-bpe", 156940, "IT'S",    new[] { 964, 13575 } },
        { "llama-bpe", 156940, "(hello)", new[] { 3283, 4896, 8 } },
        { "llama-bpe", 156940, "a  b",    new[] { 64, 220, 293 } },
        { "llama-bpe", 156940, "12345",   new[] { 4513, 1774 } },

        // Qwen-3.5 (Qwen3.8-27B-UD-Q3_K_XL). Same probes as qwen2 (regex is identical apart from
        // combining-mark handling), plus a probe on the axis that IS qwen35-specific: qwen2's
        // letter class excludes \p{M}, so a base letter followed by a combining mark splits into
        // two pieces; qwen35's [\p{L}\p{M}]+ keeps them joined. "cafe\u0301(x)" with a DECOMPOSED
        // e + COMBINING ACUTE ACCENT (U+0301, not the precomposed "é" codepoint) is the probe.
        { "qwen35", 248320, "IT'S",    new[] { 922, 12887 } },
        { "qwen35", 248320, "(hello)", new[] { 3101, 4638, 8 } },
        { "qwen35", 248320, "«mot",    new[] { 22961, 45263 } },
        { "qwen35", 248320, "12",      new[] { 16, 17 } },
        { "qwen35", 248320, "a  b",    new[] { 64, 220, 292 } },
        { "qwen35", 248320, "café(x)", new[] { 895, 1795, 52033, 2007, 8 } },
    };

    [Theory]
    [MemberData(nameof(RegexDivergenceProbes))]
    public void Encode_RegexDivergenceProbe_MatchesLlamaCppTokenIds(string pre, int vocabSize, string probe, int[] expected)
    {
        var tokenizer = FindTokenizerByPre(pre);
        Assert.SkipWhen(tokenizer is null, $"No local GGUF declares tokenizer.ggml.pre = '{pre}'.");
        Assert.SkipUnless(tokenizer!.VocabSize == vocabSize,
            $"Fixture for pre '{pre}' has vocab {tokenizer.VocabSize}, not the {vocabSize} these reference IDs came from.");

        var actual = tokenizer.Encode(probe);

        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// Finds a GGUF declaring the given <c>tokenizer.ggml.pre</c>. Keyed on the pre-type rather than
    /// a filename because the behaviour belongs to the tokenizer family, not to one checkpoint.
    /// </summary>
    private static GgufTokenizer? FindTokenizerByPre(string pre)
    {
        foreach (var dir in CandidateModelDirs())
        {
            foreach (var path in Directory.EnumerateFiles(dir, "*.gguf"))
            {
                try
                {
                    using var model = GgufModel.Open(path);
                    if (model.Metadata.TryGetValue("tokenizer.ggml.pre", out var value)
                        && value as string == pre)
                        return GgufTokenizer.FromGgufModel(model);
                }
                catch
                {
                    // Unreadable / partial download — keep looking.
                }
            }
        }
        return null;
    }

    private static IEnumerable<string> CandidateModelDirs()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var models = Path.Combine(dir, "models");
            if (Directory.Exists(models)) yield return models;
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        if (Directory.Exists(@"E:\models")) yield return @"E:\models";
    }
}
