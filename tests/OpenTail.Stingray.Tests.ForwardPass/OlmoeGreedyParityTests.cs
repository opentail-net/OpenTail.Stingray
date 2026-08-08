using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for OLMoE — the receipt that admits <c>olmoe</c> to
/// <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>Why this test exists.</b> `olmoe` was absent from the allowlist while the codebase
/// plainly implements it: <c>ModelGraph</c> carries its <c>norm_topk_prob=false</c> router
/// behaviour, the CUDA and Vulkan backends document its per-channel QK-norm shape, and
/// `docs/cpu-performance-baseline.md` has a measured CPU baseline for it. It ran on the CLI (which
/// applied no gate) and was refused by the server (which did). The gate now runs everywhere, so the
/// architecture needed either a receipt or an explicit reason to stay out. This is the receipt.</para>
///
/// <para><b>Why a test and not a CLI comparison.</b> Our CLI renders the model's chat template and
/// has no raw-completion flag, so `stingray -p …` prefills 17 tokens where `llama-cli -no-cnv`
/// prefills 5. Comparing those two outputs would be comparing different prompts. Driving
/// <see cref="Engine.ForwardPass"/> with the reference token ids directly is the only
/// apples-to-apples form.</para>
///
/// <para><b>Reference.</b> `tools/llama.cpp` build `b8585-cpu`, model
/// `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` (SHA-256 begins `3BD9EC48045F`):
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [510, 5347, 273, 6181, 310]
/// llama-cli -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 -no-cnv
///   -> "The capital of France is Paris. Paris is one of the most popular tourist destinations
///       in the world, known for its iconic"
/// </code>
/// </para>
/// </summary>
public sealed class OlmoeGreedyParityTests
{
    private const string ModelFile = "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [510, 5347, 273, 6181, 310];

    /// <summary>
    /// The continuation llama.cpp produces for those tokens under greedy decoding. Leading space
    /// included — it belongs to the first generated token, not to the prompt.
    /// </summary>
    private const string ReferenceContinuation =
        " Paris. Paris is one of the most popular tourist destinations in the world, known for its iconic";

    [Fact]
    public void Olmoe_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Guard the fixture: a different OLMoE quantisation shares the architecture but not
        // necessarily these exact greedy tokens, and would fail here for the wrong reason.
        Assert.Equal("olmoe", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        var logits = fwd.Prefill(s_promptTokens);
        var generated = new List<int>(24);
        int pos = s_promptTokens.Length;
        for (int i = 0; i < 24; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < 24) logits = fwd.Forward(next, pos++);
        }

        Assert.Equal(ReferenceContinuation, tokenizer.Decode(generated));
    }

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(candidate)) return candidate;
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        var external = Path.Combine(@"E:\models", ModelFile);
        return File.Exists(external) ? external : null;
    }
}
