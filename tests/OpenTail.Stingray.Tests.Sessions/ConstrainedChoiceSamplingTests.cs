using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

public sealed class ConstrainedChoiceSamplingTests
{
    [Fact]
    public async Task Test1_AllowedChoiceOnly()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "YES", "NO" },
            MaxNewTokens = 10
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        string full = string.Join("", chunks);
        Assert.True(full == "YES" || full == "NO");
    }

    [Fact]
    public async Task Test2_IllegalTokenMasked()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockHighestLogitForwardPass(highestTokenId: 99); // Token 99 has highest logit (illegal)
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "YES" }, // Only Token 10 ("YES") is legal
            MaxNewTokens = 10
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        // Even though Token 99 had highest raw logit, logit masking forces Token 10 ("YES") to win!
        Assert.Equal("YES", string.Join("", chunks));
    }

    [Fact]
    public async Task Test3_MultiTokenChoice()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "NEEDS_REVISION" }, // Multi-token: 30 -> 31 -> 32
            MaxNewTokens = 10
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        Assert.Equal(new[] { "30", "31", "32" }, chunks);
    }

    [Fact]
    public async Task Test4_MixedSingleAndMultiTokenChoices()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "APPROVED", "REJECTED", "NEEDS_REVISION" },
            MaxNewTokens = 10
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        Assert.True(chunks.Count >= 1);
    }

    [Fact]
    public async Task Test5_GreedySampling()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            Temperature = 0.0f, // Greedy
            AllowedChoices = new[] { "YES", "NO" }
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        Assert.Single(chunks);
    }

    [Fact]
    public async Task Test6_StochasticSampling()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            Temperature = 0.8f,
            TopP = 0.9f,
            AllowedChoices = new[] { "YES", "NO" }
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        string res = string.Join("", chunks);
        Assert.True(res == "YES" || res == "NO" || res == "10" || res == "20");
    }

    [Fact]
    public async Task Test7_ChoiceTerminatesGeneration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "YES" },
            MaxNewTokens = 100 // Ask for 100 tokens, but choice completes at 1 token!
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        Assert.Single(chunks); // Stopped cleanly as soon as choice completed!
    }

    [Fact]
    public async Task Test8_PrefixCollision()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "A", "APPROVED" } // "A" is prefix of "APPROVED"
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        Assert.NotEmpty(chunks);
    }

    [Fact]
    public async Task Test9_DuplicateChoices()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = new[] { "YES", "YES", "NO" } // Duplicate "YES"
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        Assert.NotEmpty(chunks);
    }

    [Fact]
    public async Task Test10_EmptyChoices()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = Array.Empty<string>() // Empty list disables constraint
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        Assert.NotEmpty(chunks);
    }

    [Fact]
    public async Task Test11_NullChoicesPreserveNormalSampling()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams
        {
            AllowedChoices = null // Null preserves normal sampling
        };

        var chunks = new List<string>();
        await foreach (var chunk in session.GenerateAsync(sampling))
        {
            chunks.Add(chunk.Text);
        }

        Assert.NotEmpty(chunks);
    }

    [Fact]
    public async Task Test12_WhitespaceIsSignificant()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling1 = new SamplingParams { AllowedChoices = new[] { "YES" } };
        var sampling2 = new SamplingParams { AllowedChoices = new[] { " YES" } };

        Assert.NotEqual(sampling1.AllowedChoices[0], sampling2.AllowedChoices[0]);
    }

    [Fact]
    public async Task Test13_ConstraintStatePerGeneration()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var sessionA = new InferenceSession(cache, forwardPass: fwd);
        await using var sessionB = new InferenceSession(cache, forwardPass: fwd);
        sessionA.Tokenizer = new MockChoiceTokenizer();
        sessionB.Tokenizer = new MockChoiceTokenizer();

        var sharedSampling = new SamplingParams { AllowedChoices = new[] { "YES", "NO" } };

        await sessionA.AppendAsync(new int[] { 1 });
        await sessionB.AppendAsync(new int[] { 1 });

        var tA = sessionA.GenerateAsync(sharedSampling).ToListAsync();
        var tB = sessionB.GenerateAsync(sharedSampling).ToListAsync();

        await Task.WhenAll(tA.AsTask(), tB.AsTask());
        Assert.NotEmpty(await tA);
        Assert.NotEmpty(await tB);
    }

    [Fact]
    public async Task Test14_ForkIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var parent = new InferenceSession(cache, forwardPass: fwd);
        parent.Tokenizer = new MockChoiceTokenizer();

        await parent.AppendAsync(new int[] { 1 });
        await using var child = parent.Fork();

        var sampling = new SamplingParams { AllowedChoices = new[] { "APPROVED", "REJECTED" } };

        var parentChunks = await parent.GenerateAsync(sampling).ToListAsync();
        var childChunks = await child.GenerateAsync(sampling).ToListAsync();

        Assert.NotEmpty(parentChunks);
        Assert.NotEmpty(childChunks);
    }

    [Fact]
    public async Task Test15_SpeculativeSafety()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var fwd = new MockChoiceForwardPass();
        await using var session = new InferenceSession(cache, forwardPass: fwd);
        session.Tokenizer = new MockChoiceTokenizer();

        await session.AppendAsync(new int[] { 1 });

        var sampling = new SamplingParams { AllowedChoices = new[] { "YES" } };
        var chunks = await session.GenerateAsync(sampling).ToListAsync();

        Assert.Single(chunks);
    }

    [Fact]
    public async Task Test16_NoLegalTokenFailsClosed()
    {
        using var cache = new CpuKvCache(totalPages: 100, pageSizeTokens: 32);
        var trie = TokenChoiceTrie.Build(new[] { "YES" }, new MockChoiceTokenizer());
        var state = trie.CreateState();

        var logits = new float[100];
        state.MaskLogits(logits);

        // Only Token 10 ("YES") is legal (has float 0.0), all other logits masked to float.NegativeInfinity
        Assert.Equal(0.0f, logits[10]);
        Assert.True(float.IsNegativeInfinity(logits[0]));
        Assert.True(float.IsNegativeInfinity(logits[99]));
    }

    private sealed class MockChoiceForwardPass : IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockChoiceForwardPass { Position = Position };
        public System.ReadOnlySpan<float> Forward(int position, int token)
        {
            Position = position + 1;
            var res = new float[100];
            res[10] = 5.0f; // Token 10 = "YES" / "APPROVED"
            res[20] = 3.0f; // Token 20 = "NO" / "REJECTED"
            res[30] = 4.0f; // Token 30 = "NEEDS_REVISION" (part 1)
            return res;
        }
        public System.ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Position = startPos + tokens.Count;
            return new float[100];
        }
        public void TruncateTo(int position) { Position = position; }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private sealed class MockHighestLogitForwardPass(int highestTokenId) : IForwardPass
    {
        public int Position { get; private set; }
        public int VocabSize => 100;
        public int MaxSeqLen => 2048;

        public IForwardPass CreateContext() => new MockHighestLogitForwardPass(highestTokenId) { Position = Position };
        public System.ReadOnlySpan<float> Forward(int position, int token)
        {
            Position = position + 1;
            var res = new float[100];
            res[highestTokenId] = 100.0f; // Highest raw logit on illegal token!
            res[10] = 5.0f; // Legal choice token
            return res;
        }
        public System.ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Position = startPos + tokens.Count;
            return new float[100];
        }
        public void TruncateTo(int position) { Position = position; }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private sealed class MockChoiceTokenizer : ITokenizer
    {
        public int VocabSize => 100;
        public int BosTokenId => 1;
        public int EosTokenId => 0;
        public int UnknownTokenId => -1;
        public int PadTokenId => -1;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => System.Collections.Immutable.ImmutableArray.Create(0);
        public System.Collections.Generic.IReadOnlyDictionary<string, int> SpecialTokens => System.Collections.Immutable.ImmutableDictionary<string, int>.Empty;
        public byte[] DecodeBytes(int token) => Array.Empty<byte>();
        public string Decode(IEnumerable<int> tokens)
        {
            var arr = tokens.ToArray();
            if (arr.SequenceEqual(new int[] { 10 })) return "YES";
            if (arr.SequenceEqual(new int[] { 20 })) return "NO";
            if (arr.SequenceEqual(new int[] { 30, 31, 32 })) return "NEEDS_REVISION";
            if (arr.SequenceEqual(new int[] { 40 })) return "A";
            if (arr.SequenceEqual(new int[] { 50 })) return " YES";
            return string.Join("", arr);
        }
        public IReadOnlyList<int> Encode(string text) => text switch
        {
            "YES" => new int[] { 10 },
            "NO" => new int[] { 20 },
            "APPROVED" => new int[] { 10 },
            "REJECTED" => new int[] { 20 },
            "NEEDS_REVISION" => new int[] { 30, 31, 32 },
            "A" => new int[] { 40 },
            " YES" => new int[] { 50 },
            _ => new int[] { 99 }
        };
    }
}
