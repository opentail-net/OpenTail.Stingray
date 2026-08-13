using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core.Grammar;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Extension helpers for schema-constrained generation on <see cref="IInferenceSession"/>.
/// </summary>
public static class InferenceSessionGrammarExtensions
{
    /// <summary>
    /// Generates tokens constrained to the JSON schema of type <typeparamref name="T"/>.
    /// </summary>
    public static IAsyncEnumerable<GenerateChunk> GenerateAsync<[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties)] T>(
        this IInferenceSession session,
        SamplingParams sampling,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(sampling);

        var sm = JsonSchemaGrammarCompiler.CompileForType<T>();
        var tokenizer = session.Tokenizer;

        var effectiveSampling = sampling;
        if (tokenizer is not null)
        {
            var vocab = new GrammarVocabulary(tokenizer);
            var masker = new JsonSchemaGrammarMasker(vocab, sm);
            effectiveSampling = sampling with { Constraint = masker };
        }

        return session.GenerateAsync(effectiveSampling, cancellationToken);
    }
}
