
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for SmolLM3 — the receipt that admits <c>smollm3</c> to
/// <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>What SmolLM3 needs beyond the plain llama trunk.</b> Exactly one twist: NoPE layers,
/// where every 4th layer skips RoPE entirely (`models/smollm3.cpp` hardcodes
/// `n_no_rope_layer_step = 4`, gated on `(il + 1) % step != 0` — the identical expression this
/// engine already used for Llama-4, so `ModelGraph.cs` just added `isSmolLm3` alongside `isLlama4`
/// in the existing `noRopeStep` computation). RMSNorm, gated SwiGLU FFN, standard GQA attention, no
/// scale trio, no QK-norm — otherwise identical to the Llama family this engine already runs.
/// Tokenizer pre-type `smaug-bpe` was already covered by the ported cascade table (§3 of the
/// coverage plan), so this exercises only the architecture axis.</para>
///
/// <para><b>Why a test and not a CLI comparison.</b> Same reasoning as the OLMoE/Granite receipts:
/// the CLI renders the chat template (and SmolLM3 additionally injects a reasoning-mode wrapper via
/// its template) rather than prefilling a raw completion. Driving <see cref="Engine.ForwardPass"/>
/// directly with the reference token ids is the only apples-to-apples form.</para>
///
/// <para><b>Reference.</b> `tools/llama.cpp` build `b8585-cad2d3884`, model
/// `SmolLM3-3B-Q4_K_M.gguf` (ggml-org, from `HuggingFaceTB/SmolLM3-3B`, Apache-2.0):
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [791, 6864, 315, 9822, 374]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> "The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris. The
///       Eiffel Tower was built for the"
/// </code>
/// `llama-completion`, not `llama-cli`: this build's `llama-cli -no-cnv` silently falls back to
/// interactive conversation mode instead of raising — see the coverage plan's "operational
/// gotcha" note. Raw completion mode also sidesteps SmolLM3's chat-template reasoning wrapper
/// (`[Start thinking]`) entirely, which only applies through the template, not raw continuation.
/// </para>
/// </summary>
public sealed class SmolLm3GreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "SmolLM3-3B-Q4_K_M.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [791, 6864, 315, 9822, 374];

    /// <summary>
    /// The continuation llama-completion produces for those tokens under greedy decoding. Leading
    /// space included — it belongs to the first generated token, not to the prompt.
    /// </summary>
    private const string ReferenceContinuation =
        " Paris. The Eiffel Tower is a famous landmark in Paris. The Eiffel Tower was built for the ";

    [Fact]
    public void SmolLm3_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("smollm3", Convert.ToString(model.Metadata["general.architecture"]));
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

        string continuation = tokenizer.Decode(generated);
        Assert.Equal(ReferenceContinuation, continuation);
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
