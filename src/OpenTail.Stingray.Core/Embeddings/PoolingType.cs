namespace OpenTail.Stingray.Core.Embeddings;

/// <summary>
/// Pooling strategies used to aggregate token-level hidden states into sequence-level embedding vectors.
/// </summary>
public enum PoolingType
{
    /// <summary>
    /// No sequence pooling; returns unpooled per-token embeddings [seq_len, d_model].
    /// </summary>
    None = 0,

    /// <summary>
    /// Mean pooling: averages hidden states across all non-padding tokens.
    /// Standard for BGE, ModernBERT, Nomic, GTE, and MiniLM.
    /// </summary>
    Mean = 1,

    /// <summary>
    /// CLS / First token pooling: uses the hidden state of the first sequence token (BOS / [CLS]).
    /// Standard for BERT, RoBERTa, and DeBERTa.
    /// </summary>
    Cls = 2,

    /// <summary>
    /// Last token pooling: uses the hidden state of the final sequence token before EOS / padding.
    /// Standard for causal LLM embedders (Qwen2-Embed, Llama-3-Embed, Snowflake Arctic).
    /// </summary>
    LastToken = 3,

    /// <summary>
    /// Cross-Encoder classification score ranking for Query-Document relevance pairs.
    /// </summary>
    Rank = 4
}
