using System.Net;
using System.Net.Http.Headers;
using System.Text;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>A fake Hugging Face (model API, 302 to a CDN, ranged files) and a tiny valid-GGUF builder, shared by the remote-scout and quant-picker tests. No network.</summary>
internal static class HubFixtures
{
    internal const string Sha = "0123456789abcdef0123456789abcdef01234567";
    internal const string Hash = "aaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccdddd";
    // ── a tiny GGUF builder (the Core tests have one, but it is internal to that project) ──
    internal static void Str(BinaryWriter w, string s) { var b = Encoding.UTF8.GetBytes(s); w.Write((ulong)b.Length); w.Write(b); }

    internal static byte[] Gguf(string arch, int blocks, int extraTokenBytes = 0, int dataBytes = 1 << 16, bool qkNorm = false, uint? extraTensorType = null)
    {
        var tensors = new List<(string, long[], uint, ulong)>();
        ulong off = 0;
        void Add(string n, long[] d, uint type = 0) { tensors.Add((n, d, type, off)); off += (ulong)(d.Aggregate(1L, (a, b) => a * b) * 4); off = (off + 31) / 32 * 32; }
        Add("token_embd.weight", [16, 8]);
        if (extraTensorType is uint xt) Add("blk.0.extra.weight", [16], xt);
        for (int i = 0; i < blocks; i++)
        {
            Add($"blk.{i}.attn_norm.weight", [16]); Add($"blk.{i}.attn_q.weight", [16, 16]); Add($"blk.{i}.attn_k.weight", [16, 8]);
            Add($"blk.{i}.attn_v.weight", [16, 8]); Add($"blk.{i}.attn_output.weight", [16, 16]);
            if (qkNorm) { Add($"blk.{i}.attn_q_norm.weight", [8]); Add($"blk.{i}.attn_k_norm.weight", [8]); }
            Add($"blk.{i}.ffn_norm.weight", [16]); Add($"blk.{i}.ffn_gate.weight", [16, 32]); Add($"blk.{i}.ffn_up.weight", [16, 32]); Add($"blk.{i}.ffn_down.weight", [32, 16]);
        }
        using var ms = new MemoryStream(); using var w = new BinaryWriter(ms);
        var kvs = new List<(string, uint, Action<BinaryWriter>)>
        {
            ("general.architecture", 8, x => Str(x, arch)),
            ($"{arch}.block_count", 4, x => x.Write((uint)blocks)),
            ($"{arch}.embedding_length", 4, x => x.Write(16u)),
            ($"{arch}.attention.head_count", 4, x => x.Write(2u)),
            ($"{arch}.attention.head_count_kv", 4, x => x.Write(1u)),
            ($"{arch}.feed_forward_length", 4, x => x.Write(32u)),
            ($"{arch}.context_length", 4, x => x.Write(256u)),
            ("tokenizer.ggml.model", 8, x => Str(x, "gpt2")),
            ("tokenizer.ggml.merges", 9, x => { x.Write(8u); x.Write(1UL); Str(x, "a b"); }),
            ("tokenizer.ggml.tokens", 9, x => { x.Write(8u); x.Write(2UL); Str(x, "a"); Str(x, new string('x', Math.Max(1, extraTokenBytes))); }),
        };
        w.Write(0x46554747u); w.Write(3u); w.Write((ulong)tensors.Count); w.Write((ulong)kvs.Count);
        foreach (var (k, t, v) in kvs) { Str(w, k); w.Write(t); v(w); }
        foreach (var (n, d, t, o) in tensors) { Str(w, n); w.Write((uint)d.Length); foreach (var x in d) w.Write((ulong)x); w.Write(t); w.Write(o); }
        while (ms.Length % 32 != 0) w.Write((byte)0);
        w.Write(new byte[Math.Max(dataBytes, (int)off + 64)]);
        return ms.ToArray();
    }

    // ── fake Hub: model API, 302 to a CDN, ranged files ──
    internal sealed record RepoFile(string Path, byte[] Bytes, string? Sha256 = Hash, long? ListedSize = null);

    internal sealed class FakeHub(string repo, IEnumerable<RepoFile> files, string? hubArch = "llama", string? gated = null, HttpStatusCode apiStatus = HttpStatusCode.OK) : HttpMessageHandler
    {
        private readonly List<RepoFile> _files = files.ToList();
        public List<string> Requests { get; } = [];
        public long BodyBytes { get; private set; }

        private string ApiJson()
        {
            var sb = new StringBuilder();
            sb.Append("{\"id\":\"").Append(repo).Append("\",\"private\":false,\"gated\":").Append(gated is null ? "false" : $"\"{gated}\"")
              .Append(",\"downloads\":1234,\"sha\":\"").Append(Sha).Append("\",\"tags\":[\"gguf\",\"license:mit\"]");
            if (hubArch is not null) sb.Append(",\"gguf\":{\"architecture\":\"").Append(hubArch).Append("\",\"total\":1}");
            sb.Append(",\"siblings\":[");
            sb.Append(string.Join(",", _files.Select(f =>
                $"{{\"rfilename\":\"{f.Path}\",\"size\":{f.ListedSize ?? f.Bytes.Length}" + (f.Sha256 is null ? "" : $",\"lfs\":{{\"sha256\":\"{f.Sha256}\",\"size\":{f.Bytes.Length}}}") + "}")));
            sb.Append("]}");
            return sb.ToString();
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage req, CancellationToken ct)
        {
            var uri = req.RequestUri!;
            Requests.Add(uri.Host + uri.AbsolutePath);
            if (uri.AbsolutePath.StartsWith("/api/models/", StringComparison.Ordinal))
                return Task.FromResult(apiStatus == HttpStatusCode.OK
                    ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(ApiJson()) }
                    : new HttpResponseMessage(apiStatus));
            if (uri.Host == "huggingface.co")
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.Found) { Headers = { Location = new Uri("https://cdn-lfs.huggingface.co" + uri.AbsolutePath) } });

            string prefix = $"/{repo}/resolve/{Sha}/";
            if (!uri.AbsolutePath.StartsWith(prefix, StringComparison.Ordinal)) return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
            string path = Uri.UnescapeDataString(uri.AbsolutePath[prefix.Length..]);
            var file = _files.FirstOrDefault(f => f.Path == path);
            if (file is null) return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));

            var range = req.Headers.Range!.Ranges.First();
            long from = range.From!.Value, to = Math.Min(range.To ?? file.Bytes.Length - 1, file.Bytes.Length - 1);
            var body = file.Bytes[(int)from..(int)(to + 1)];
            BodyBytes += body.Length;
            var resp = new HttpResponseMessage(HttpStatusCode.PartialContent) { Content = new ByteArrayContent(body) };
            resp.Content.Headers.ContentRange = new ContentRangeHeaderValue(from, to, file.Bytes.Length);
            return Task.FromResult(resp);
        }
    }


}
