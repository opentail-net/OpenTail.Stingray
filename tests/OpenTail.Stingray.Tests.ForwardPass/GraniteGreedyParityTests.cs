using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for Granite — the receipt that admits <c>granite</c>
/// (dense) to <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>What Granite needs beyond the plain llama trunk.</b> Granite is architecturally a
/// standard RMSNorm + GQA + SiLU-gated-FFN transformer with interleaved (non-NEOX) RoPE — the same
/// trunk this engine already runs for Llama/Mistral/SmolLM — plus a "scale trio" plus one attention
/// override, all read from GGUF metadata rather than hardcoded: <c>granite.embedding_scale</c>
/// (multiplies token embeddings before the trunk), <c>granite.residual_scale</c> (multiplies each
/// sublayer's output — attention and FFN independently — before it joins the residual stream),
/// <c>granite.logit_scale</c> (divides the final logits: <c>ModelHyperparams.LogitScale</c> stores
/// the reciprocal so <c>ForwardPass</c> can just multiply), and <c>granite.attention.scale</c> (an
/// explicit <c>kq_scale</c> override — this checkpoint declares 0.015625 = 1/64, NOT
/// 1/sqrt(64) = 0.125, a genuine per-model override rather than a rounding of the usual formula).
/// <c>MiniCPM</c> shares Granite's exact graph in llama.cpp and reuses this same implementation
/// (see docs/01-gguf-model-coverage-plan.md §1d).</para>
///
/// <para><b>Why a test and not a CLI comparison.</b> Same reasoning as
/// <c>OlmoeGreedyParityTests</c>: the CLI renders the chat template and prefills a different token
/// sequence than a raw completion. Driving <see cref="Engine.ForwardPass"/> with the reference
/// token ids directly is the only apples-to-apples form.</para>
///
/// <para><b>Reference.</b> `tools/llama.cpp` build `b8585-cad2d3884`, model
/// `granite-3.3-2b-instruct-Q4_K_M.gguf` (bartowski, from
/// `ibm-granite/granite-3.3-2b-instruct`, Apache-2.0):
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [1318, 18926, 432, 45600, 438]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> "The capital of France is Paris.
///
///       Step 1: Identify the topic.
///       The topic is the capital of France."
/// </code>
/// llama-cli's own `-no-cnv` is not honoured by this build ("--no-conversation is not supported by
/// llama-cli, please use llama-completion instead") — it silently falls back to interactive
/// conversation mode instead of raising, which looks like a hang (see the coverage plan's
/// "operational gotcha" note). llama-completion is the correct raw-completion tool here.
/// </para>
///
/// <para><b>Incidental finding while building this receipt: a real hang/memory-leak bug in the core
/// Jinja engine</b>, unrelated to Granite's forward-pass math. <c>GgufTokenizer.FromGgufModel</c>
/// used to construct <c>JinjaChatTemplate</c> eagerly for every model load; Granite's chat template
/// (4,571 chars — tool-call/citation/hallucination-risk sections, a `strftime_now()` call, a
/// `tojson(indent=4)` filter) hung it indefinitely, discovered as multi-gigabyte unbounded memory
/// growth rather than a clean failure. Root cause: <c>ExprParser.ParseArgList</c> (filter call
/// arguments) had no handling for `key=value` syntax — for `indent=4`, nothing in the expression
/// grammar ever consumes the `=`, so the loop retried the same position forever, appending a new
/// null-literal argument every iteration. Fixed in two parts: `ParseArgList` now recognises and
/// consumes a keyword-argument prefix (dropping the name — `FilterExpr` has no kwargs slot, so only
/// the value is kept, which is enough for something cosmetic like JSON indent), and the loop now
/// asserts forward progress every iteration and throws rather than spin if some future construct
/// hits the same failure mode. Separately, `JinjaChatTemplate` construction is now lazy (built on
/// first access to `GgufTokenizer.ChatTemplate`, not at load time) so a pathological template only
/// costs whichever caller actually renders one, not every model load.</para>
/// </summary>
public sealed class GraniteGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "granite-3.3-2b-instruct-Q4_K_M.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [1318, 18926, 432, 45600, 438];

    /// <summary>
    /// The continuation llama-completion produces for those tokens under greedy decoding. Leading
    /// space included — it belongs to the first generated token, not to the prompt.
    /// </summary>
    private const string ReferenceContinuation =
        " Paris.\n\nStep 1: Identify the topic.\nThe topic is the capital of France.";

    [Fact]
    public void Granite_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Guard the fixture: a different Granite checkpoint shares the architecture but not
        // necessarily these exact greedy tokens, and would fail here for the wrong reason.
        Assert.Equal("granite", Convert.ToString(model.Metadata["general.architecture"]));
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

        // Full 24-token EXACT match — stronger than the OLMoE receipt (which only achieves a
        // 2-token prefix). Granite's scale trio and attention-scale override reproduce llama.cpp
        // token-for-token, not just in aggregate (perplexity).
        string continuation = tokenizer.Decode(generated);
        Assert.Equal(ReferenceContinuation, continuation);
    }

    /// <summary>
    /// Oracle-free invariant, same rationale as <c>Olmoe_DecodeStepwise_AgreesWithSinglePassPrefill</c>:
    /// prefilling the whole sequence in one pass must match stepping the same tokens through decode
    /// one at a time, since both paths compute the same function of the same tokens. This is the
    /// receipt that both CPU dense code paths — <c>PrefillCore</c> (batched, used for N&gt;1 prompts)
    /// and <c>Attention</c>/<c>RunTrunk</c> (single-token decode) — apply the scale trio consistently.
    /// They were NOT consistent by construction: <c>PrefillCore</c> never had an <c>EmbeddingScale</c>
    /// application point before this work, because the only prior architecture to set a non-1
    /// <c>EmbeddingScale</c> (Gemma 4) always takes the sequential decode path instead
    /// (<c>_layerHeadDim is not null</c> forces the per-layer-head-dim fallback), so the two never
    /// coexisted until Granite — a dense model with no per-layer head dim — actually reached
    /// <c>PrefillCore</c> with a real embedding scale for the first time.
    /// </summary>
    [Fact]
    public void Granite_DecodeStepwise_AgreesWithSinglePassPrefill()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this consistency check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Prompt plus the two tokens both the engine and llama.cpp agree on (" Paris", ".").
        int[] full = [.. s_promptTokens, 2716, 297];

        using var backend = new CpuBackend();

        float[] stepwise;
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048))
        {
            fwd.Prefill(s_promptTokens);
            var logits = fwd.Forward(full[^2], s_promptTokens.Length);
            logits = fwd.Forward(full[^1], s_promptTokens.Length + 1);
            stepwise = logits[..tokenizer.VocabSize].ToArray();
        }

        float[] singlePass;
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048))
        {
            singlePass = fwd.Prefill(full)[..tokenizer.VocabSize].ToArray();
        }

        int argmaxStep = Array.IndexOf(stepwise, stepwise.Max());
        int argmaxFull = Array.IndexOf(singlePass, singlePass.Max());

        float maxDiff = 0;
        for (int i = 0; i < stepwise.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(stepwise[i] - singlePass[i]));

        Assert.True(argmaxStep == argmaxFull,
            $"prefill/decode disagree on argmax: stepwise {argmaxStep} "
            + $"({tokenizer.Decode([argmaxStep])!.Replace("\n", "\\n")}) vs single-pass {argmaxFull} "
            + $"({tokenizer.Decode([argmaxFull])!.Replace("\n", "\\n")}), maxDiff {maxDiff:F4}");
        // Bound set above the measured Q8-activation-prefill gap (see the OLMoE receipt for the
        // same int8-approximation discussion), not tightened to zero for the same reason.
        Assert.True(maxDiff < 1.0f,
            $"prefill/decode logits diverge by {maxDiff:F4}, beyond the int8 prefill approximation.");
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
