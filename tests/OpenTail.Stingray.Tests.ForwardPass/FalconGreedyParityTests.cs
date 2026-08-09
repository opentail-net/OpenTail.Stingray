using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Greedy token parity against llama.cpp for Falcon-7B — the receipt that admits
/// <c>falcon</c> to <see cref="ModelCompatibility"/>'s architecture allowlist.
///
/// <para><b>What Falcon needs beyond what GPT-NeoX already built.</b> Falcon-7B reuses every
/// gptneox mechanism (LayerNorm-with-bias, a non-gated GELU FFN, a fused
/// <c>attn_qkv.weight</c> tensor split by contiguous row offset, and
/// <see cref="ModelHyperparams.UseParallelResidual"/>'s 3-way residual sum
/// <c>x + attn(norm) + ffn(norm)</c>) plus exactly one new structural wrinkle: Falcon-7B has NO
/// separate <c>ffn_norm</c> tensor at all — attention and FFN read the SAME LayerNorm output,
/// confirmed directly against <c>examples/llama.cpp/llama.cpp/src/models/falcon.cpp</c>, which
/// literally comments <c>build_ffn(attn_norm, ... // !! use the attn norm, not the result</c>.
/// <c>ForwardPass</c>'s constructor falls <c>_ffnNorm[i]</c>/<c>_bFfnNorm[i]</c> back to
/// <c>_attnNorm[i]</c>/<c>_bAttnNorm[i]</c>'s own <c>TensorRef</c>/pointer whenever
/// <c>blk.*.ffn_norm.{weight,bias}</c> is absent from the GGUF — recomputing the identical
/// LayerNorm a second time rather than caching one activation, which is bit-identical (same
/// deterministic formula, same input) and simpler than adding a second code path.
/// <c>Dispose()</c> was updated to compare <c>_bFfnNorm[i] != _bAttnNorm[i]</c> before freeing,
/// since the fallback makes them alias the same allocation for this architecture — freeing both
/// unconditionally would double-free, the exact class of bug the GPT-NeoX receipt already found
/// once for an unrelated fused-QKV-bias aliasing case.</para>
///
/// <para><b>Falcon carries NO biases anywhere except the norm itself</b> — no QKV bias, no
/// attention-output bias, no FFN bias (confirmed: <c>falcon.cpp</c>'s tensor loader never calls
/// <c>create_tensor</c> for <c>wqkv_b</c>/<c>wo_b</c>/<c>ffn_up_b</c>/<c>ffn_down_b</c> at all).
/// This falls out of the existing tensor-presence-based <c>HasAttnBias</c>/<c>HasFfnBias</c>
/// detection automatically (both resolve false) — no new code needed. Also new here:
/// <c>use_parallel_residual</c> is never a metadata key for this architecture at all — llama.cpp
/// hardcodes the 3-way sum unconditionally in Falcon's graph rather than reading it from GGUF
/// metadata the way GPT-NeoX does — so <c>ModelGraph.cs</c> hardcodes
/// <c>UseParallelResidual = true</c> for <c>arch == "falcon"</c> rather than reading a key that
/// was never written; confirmed absent by inspecting this checkpoint's own metadata dump (no
/// <c>falcon.use_parallel_residual</c> key present).</para>
///
/// <para><b>Multi-Query Attention, not GQA or plain MHA.</b> This checkpoint declares
/// <c>attention.head_count=71</c>, <c>attention.head_count_kv=1</c> — a single shared KV head
/// across 71 query heads (headDim 4544/71=64). Exercises the fused-QKV split's existing
/// <c>_numHeads</c>/<c>_numKvHeads</c> parametrization (built generically for GQA already, e.g.
/// Qwen2) at an extreme ratio for the first time on this non-gated-FFN/parallel-residual profile
/// — no new code needed, but a real stress test of dimension arithmetic the GPT-NeoX receipt
/// (head_count==head_count_kv, no MQA/GQA at all) never exercised.</para>
///
/// <para><b>RoPE.</b> Full rotation, not partial — this checkpoint declares no
/// <c>rope.dimension_count</c> key, so <see cref="ModelHyperparams.RopeDim"/> falls back to the
/// full headDim (64), and <c>falcon.cpp</c> itself asserts
/// <c>GGML_ASSERT(n_embd_head == n_rot)</c>. NEOX convention, same as GPT-NeoX — confirmed via
/// <c>llama_model_rope_type()</c>'s <c>LLM_ARCH_FALCON</c> case (also <c>LLAMA_ROPE_TYPE_NEOX</c>).</para>
///
/// <para><b>Falcon-40B is explicitly NOT covered.</b> 40B carries a second per-layer norm
/// (<c>attn_norm_2</c>) that only the attention branch reads when present, with FFN still reading
/// the plain <c>attn_norm</c> — a different tensor-presence combination this receipt's code never
/// exercises or guards against. No small 40B checkpoint exists to validate against, so this is
/// out of scope, same as every other "moot for now" item in the plan doc; a 40B GGUF routed
/// through this code today would silently ignore <c>attn_norm_2</c> and produce wrong output.</para>
///
/// <para><b>Checkpoint.</b> <c>tiiuae/falcon-7b-instruct</c> (Apache-2.0, TII), Q4_K_M GGUF
/// (4.97 GB). <c>tokenizer.ggml.model = gpt2</c> (byte-BPE), no explicit
/// <c>tokenizer.ggml.pre</c> key on this checkpoint — the prompt-token assertion below exercises
/// whatever pretokenizer this engine's cascade resolves to by default, so a mismatch here would
/// be a tokenizer-axis finding, not an architecture-axis one.</para>
///
/// <para><b>Reference.</b> <c>tools/llama.cpp</c> build <c>b8585-cad2d3884</c>:
/// <code>
/// llama-tokenize -m &lt;model&gt; -p "The capital of France is" --ids --no-bos
///   -> [487, 4236, 275, 5582, 304]
/// llama-completion -m &lt;model&gt; -p "The capital of France is" -n 24 --temp 0 --top-k 1 --seed 0 \
///     -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false
///   -> "The capital of France is Paris.
///       Paris is the capital of France." [end of text after 10 generated tokens]
/// </code>
/// The completion terminates at EOS (token 11, <c>&lt;|endoftext|&gt;</c>) well before the
/// requested 24 tokens — re-tokenizing the exact continuation text
/// (<c>"The capital of France is Paris.\nParis is the capital of France."</c>, no-bos) gives the
/// full id sequence, from which the 10 continuation ids below are the tail after the 5 prompt
/// ids.</para>
///
/// <para><b>Result: all 10 generated tokens EXACT, including the terminating EOS</b> — the
/// engine's greedy continuation matches llama.cpp's reference token-for-token through the entire
/// completion, then predicts EOS (11) as its 11th token, matching llama.cpp's own termination.
/// This is a full, unqualified match (not a partial-prefix acceptance like the OLMoE/Apertus/
/// GPT-NeoX receipts) — the same strength of receipt as Granite's and SmolLM3's.</para>
/// </summary>
public sealed class FalconGreedyParityTests
{
    private const string ModelFile = "falcon-7b-instruct-Q4_K_M.gguf";

    /// <summary>Prompt token ids from llama-tokenize; see the class remarks.</summary>
    private static readonly int[] s_promptTokens = [487, 4236, 275, 5582, 304];

    /// <summary>
    /// The 10 continuation tokens from llama.cpp's greedy decode (re-derived by re-tokenizing
    /// the exact reference completion text, no-bos, and taking the tail past the 5 prompt ids —
    /// see the class remarks). Token 11 (EOS) is expected as the 11th generated token, asserted
    /// separately below rather than folded into this array.
    /// </summary>
    private static readonly int[] s_referenceContinuation =
        [6671, 25, 193, 38765, 304, 248, 4236, 275, 5582, 25];

    private const int EosTokenId = 11;

    [Fact]
    public void Falcon_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("falcon", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        // Guards the LayerNorm / parallel-residual / no-bias / MQA detection: this receipt is
        // worthless if the fixture silently lost its shared-norm shape or somehow picked up
        // biases or a different KV head count than expected.
        Assert.True(hp.HasNormBias, "HasNormBias must be true for LayerNorm in falcon");
        Assert.False(hp.HasAttnBias, "Falcon carries no attention biases");
        Assert.False(hp.HasFfnBias, "Falcon carries no FFN biases");
        Assert.True(hp.UseParallelResidual, "UseParallelResidual must be true for falcon (hardcoded, no metadata key)");
        Assert.Null(hp.XieluAlphaN); // must NOT take Apertus's xIELU branch
        Assert.Equal(71, hp.NumHeads);
        Assert.Equal(1, hp.NumKvHeads); // MQA
        Assert.Equal(64, hp.RopeDim); // full rotation: RopeDim == headDim (4544/71)

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        var logits = fwd.Prefill(s_promptTokens);
        var generated = new List<int>(s_referenceContinuation.Length + 1);
        int pos = s_promptTokens.Length;
        for (int i = 0; i < s_referenceContinuation.Length + 1; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (next == EosTokenId) break;
            logits = fwd.Forward(next, pos++);
        }

        Assert.Equal([.. s_referenceContinuation, EosTokenId], generated);
    }

    /// <summary>
    /// Oracle-free invariant, same rationale as the GPT-NeoX/Apertus/Granite/OLMoE receipts:
    /// prefilling the whole sequence in one pass must match stepping the same tokens through
    /// decode one at a time. Both <c>PrefillCore</c>'s parallel-residual batched branch and
    /// <c>RunTrunk</c>'s parallel-residual single-token branch are the SAME code GPT-NeoX
    /// exercises, but this receipt additionally exercises the shared-attn/ffn-norm fallback and
    /// MQA dimensions through both paths, which GPT-NeoX's checkpoint never did.
    /// </summary>
    [Fact]
    public void Falcon_DecodeStepwise_AgreesWithSinglePassPrefill()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this consistency check.");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Prompt plus the first two reference-continuation tokens (both engine and llama.cpp
        // agree on these, per the greedy-continuation test above).
        int[] full = [.. s_promptTokens, s_referenceContinuation[0], s_referenceContinuation[1]];

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
        Assert.True(maxDiff < 5.0f,
            $"prefill/decode logits diverge by {maxDiff:F4}, beyond the expected int8 prefill "
            + "approximation (bound follows the GPT-NeoX/Apertus/OLMoE receipts' precedent).");
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
