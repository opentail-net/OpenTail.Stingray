
namespace OpenTail.Stingray.Diffusion.AceStep.Text;

/// <summary>
/// Real Qwen3-Embedding-0.6B text encoding for ACE-Step's `text_hidden_states`, reusing this
/// engine's EXISTING GGUF-based `Engine.ForwardPass` rather than a hand-written transformer --
/// see docs/064-acestep-implementation-plan.md's "Corrections and confirmations" section for the
/// real-source-verified reasoning: the real `diffusers` ACE-Step pipeline runs the formatted text
/// prompt through the FULL Qwen3 model with standard CAUSAL masking
/// (`self.text_encoder(input_ids=...).last_hidden_state`), which is exactly what
/// `Engine.ForwardPass` already does for ordinary text generation -- no new attention/RoPE/RMSNorm
/// kernels needed here, only a way to read out per-token hidden states instead of next-token
/// logits.
///
/// <para><b>Real GGUF used</b>: the official `Qwen/Qwen3-Embedding-0.6B-GGUF`, dims confirmed via
/// `stingray list-metadata` to declare `general.architecture=qwen3` matching this project's own
/// `Qwen3-Embedding-0.6B/config.json` capture exactly (28 layers, hidden=1024, 16 heads / 8 kv
/// heads (GQA), head_dim=128, rope_theta=1e6, rms_eps=1e-6) -- not a GGUF this session wrote a
/// converter for; a stock official quant sufficed. No safetensors-lane change to
/// `Core/SafetensorsTextModelPackage.cs` was needed (that lane currently gates to
/// llama/mistral only; this bypasses it entirely by loading the already-available GGUF).</para>
///
/// <para><b>Real bug found and worked around, 2026-09-03</b>: the f16 quant produces NaN at
/// layer 27 (the model's LAST transformer layer) for certain real 13+-token sequences (found via
/// this project's real SFT-formatted ACE-Step prompt, not a synthetic case) -- confirmed via
/// `STINGRAY_TRACE_NORMS=1`: every earlier layer's residual norm stays finite and grows normally
/// (up to ~600-800 by layer 26), and layer 27 alone turns it to NaN. The Q8_0 quant of the SAME
/// checkpoint does NOT reproduce this on the identical token sequence, isolating it to the f16
/// weight-storage/kernel path specifically, not a fundamental architecture bug in this engine's
/// qwen3 support. This class therefore uses the Q8_0 GGUF, not f16 -- see
/// docs/064-acestep-implementation-plan.md for the full repro (exact token IDs) and the case for
/// treating the f16 path as a separate, real engine bug worth its own investigation later.</para>
///
/// <para><b>The one piece of new math needed</b>: `Engine.ForwardPass.EnableHiddenTaps` captures
/// a tapped layer's OUTPUT (HF's `hidden_states[i+1]` convention -- confirmed from
/// `ForwardPassHiddenTapTests.cs`'s doc comment, an existing test in this engine, not written this
/// session), which for the LAST layer is the raw residual stream BEFORE the model's final
/// RMSNorm. Real HF `Qwen3Model.forward().last_hidden_state` is the POST-final-norm output, so
/// this class applies that final RMSNorm itself (`output_norm.weight`, read directly from the
/// GGUF) to each tapped row before returning it -- omitting this would silently produce a
/// wrong-but-plausible-shaped conditioning tensor, not a crash.</para>
/// </summary>
public sealed class AceStepQwen3TextEncoder : IDisposable
{
    private readonly GgufModel _model;
    private readonly GgufTokenizer _tokenizer;
    private readonly CpuBackend _backend;
    private readonly ModelHyperparams _hp;
    private readonly float[] _outputNormWeight;
    private readonly Lazy<float[]> _tokenEmbeddingTable;

    public AceStepQwen3TextEncoder(string ggufPath)
    {
        _model = GgufModel.Open(ggufPath);
        _tokenizer = GgufTokenizer.FromGgufModel(_model);
        _hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _backend = new CpuBackend();

        var normInfo = _model.FindTensor("output_norm.weight")
            ?? throw new InvalidDataException("Qwen3 GGUF missing required tensor 'output_norm.weight'.");
        _outputNormWeight = new float[normInfo.ElementCount];
        Dequantize.ToFloat32(_model.GetTensorData(normInfo), _outputNormWeight, normInfo.DType, normInfo.ElementCount);

        _tokenEmbeddingTable = new Lazy<float[]>(() =>
        {
            var info = _model.FindTensor("token_embd.weight")
                ?? throw new InvalidDataException("Qwen3 GGUF missing required tensor 'token_embd.weight'.");
            var table = new float[info.ElementCount];
            Dequantize.ToFloat32(_model.GetTensorData(info), table, info.DType, info.ElementCount);
            return table;
        });
    }

    /// <summary>Tokenizes real text with this encoder's own real Qwen3 tokenizer -- exposed for callers (e.g. ACE-Step's lyric path) that need raw token IDs without running a full forward pass.</summary>
    public int[] Tokenize(string text) => _tokenizer.Encode(text).ToArray();

    /// <summary>Real Qwen3 `token_embd.weight` (flat, row-major `[vocab, hiddenSize]`) -- exposed for ACE-Step's real lyric path, which embeds lyric tokens via a raw lookup (NOT a full Qwen3 forward pass, confirmed from the real `diffusers` ACE-Step pipeline). Loaded lazily since not every caller needs it.</summary>
    public float[] TokenEmbeddingTable => _tokenEmbeddingTable.Value;

    /// <summary>Tokenizes and encodes real text through the full causal Qwen3 model, applying the final RMSNorm to each position's tap to match real `last_hidden_state`. Returns `[t][hiddenSize]`.</summary>
    public float[][] Encode(string text)
    {
        var tokenIds = _tokenizer.Encode(text).ToArray();

        using var forward = new Engine.ForwardPass(_model, _backend, _hp);
        forward.EnableHiddenTaps([_hp.NumLayers - 1]);
        forward.Prefill(tokenIds);

        var output = new float[tokenIds.Length][];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            var raw = forward.HiddenTapsAt(i);
            var normed = new float[_hp.EmbeddingDim];
            RmsNorm(raw, _outputNormWeight, normed, _hp.RmsNormEps);
            output[i] = normed;
        }
        return output;
    }

    private static void RmsNorm(ReadOnlySpan<float> x, float[] weight, Span<float> output, float eps)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
    }

    public void Dispose()
    {
        _backend.Dispose();
        _model.Dispose();
    }
}
