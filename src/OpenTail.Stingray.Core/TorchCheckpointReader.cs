using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace OpenTail.Stingray.Core;

/// <summary>A tensor read from a PyTorch checkpoint, converted to F32, row-major contiguous.</summary>
public sealed record TorchTensor(int[] Shape, float[] Data);

/// <summary>
/// Reads PyTorch zip checkpoints (<c>torch.save</c> of a <c>state_dict</c>: <c>&lt;archive&gt;/data.pkl</c> plus one
/// raw little-endian storage per <c>&lt;archive&gt;/data/&lt;key&gt;</c>), e.g. BGE-M3's <c>sparse_linear.pt</c> /
/// <c>colbert_linear.pt</c> or a <c>pytorch_model.bin</c> with no safetensors conversion.
///
/// This is not a general unpickler and never executes anything: it interprets only the opcodes
/// <c>torch.save</c> emits for nested dicts of tensors, and the only callables it accepts are
/// <c>collections.OrderedDict</c> and <c>torch._utils._rebuild_tensor_v2</c> (storages: Float/Half/BFloat16, plus Long/Int such as BatchNorm's <c>num_batches_tracked</c>, converted to float).
/// Anything else (another global, a non-contiguous view, the legacy tar format) is refused with the name of what
/// was found. Nested dicts are flattened with dotted keys.
/// </summary>
public static class TorchCheckpointReader
{
    public static Dictionary<string, TorchTensor> Read(string path)
    {
        using var zip = ZipFile.OpenRead(path);
        var pkl = zip.Entries.FirstOrDefault(e => e.FullName == "data.pkl" || e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal))
            ?? throw new NotSupportedException($"{path}: no data.pkl (not a PyTorch zip checkpoint; the legacy tar format isn't supported).");
        string root = pkl.FullName[..^"data.pkl".Length];
        var byteOrder = zip.GetEntry(root + "byteorder");
        if (byteOrder is not null)
        {
            using var r = new StreamReader(byteOrder.Open());
            string order = r.ReadToEnd().Trim();
            if (order != "little") throw new NotSupportedException($"{path}: byteorder '{order}' not supported.");
        }

        byte[] pickle;
        using (var s = pkl.Open())
        using (var ms = new MemoryStream())
        {
            s.CopyTo(ms);
            pickle = ms.ToArray();
        }

        byte[] LoadStorage(string key)
        {
            var e = zip.GetEntry(root + "data/" + key) ?? throw new InvalidDataException($"{path}: missing storage data/{key}.");
            using var s = e.Open();
            var buf = new byte[e.Length];
            s.ReadExactly(buf);
            return buf;
        }

        object top = new Unpickler(pickle, LoadStorage).Load();
        var result = new Dictionary<string, TorchTensor>(StringComparer.Ordinal);
        Flatten(top, "", result, path);
        return result;
    }

    private static void Flatten(object o, string prefix, Dictionary<string, TorchTensor> into, string path)
    {
        switch (o)
        {
            case TorchTensor t:
                into[prefix] = t;
                break;
            case List<KeyValuePair<object, object>> dict:
                foreach (var (k, v) in dict)
                {
                    if (k is not string name) throw new NotSupportedException($"{path}: non-string dict key {k}.");
                    Flatten(v, prefix.Length == 0 ? name : prefix + "." + name, into, path);
                }
                break;
            default:
                // Non-tensor leaves (ints, strings, e.g. an 'epoch' next to a 'state_dict') carry no weights.
                break;
        }
    }

    private sealed record Global(string Module, string Name);

    private sealed record StorageRef(string Type, string Key, long Numel);

    private sealed class Mark;

    private sealed class Unpickler(byte[] data, Func<string, byte[]> loadStorage)
    {
        private static readonly Mark s_mark = new();
        private readonly Stack<object?> _stack = new();
        private readonly Dictionary<int, object?> _memo = new();
        private readonly Dictionary<string, byte[]> _storages = new(StringComparer.Ordinal);
        private int _pos;

        private byte U8() => data[_pos++];

        private int I32()
        {
            int v = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(_pos));
            _pos += 4;
            return v;
        }

        private string Utf8(int len)
        {
            string s = Encoding.UTF8.GetString(data, _pos, len);
            _pos += len;
            return s;
        }

        private string Line()
        {
            int end = Array.IndexOf(data, (byte)'\n', _pos);
            if (end < 0) throw new InvalidDataException("pickle: unterminated GLOBAL.");
            string s = Encoding.ASCII.GetString(data, _pos, end - _pos);
            _pos = end + 1;
            return s;
        }

        private List<object?> PopToMark()
        {
            var items = new List<object?>();
            while (true)
            {
                object? o = _stack.Pop();
                if (ReferenceEquals(o, s_mark)) break;
                items.Add(o);
            }
            items.Reverse();
            return items;
        }

        public object Load()
        {
            while (_pos < data.Length)
            {
                byte op = U8();
                switch (op)
                {
                    case 0x80: _pos++; break; // PROTO
                    case 0x95: _pos += 8; break; // FRAME
                    case (byte)'.': return _stack.Pop() ?? throw new InvalidDataException("pickle: STOP on None.");
                    case (byte)'(': _stack.Push(s_mark); break;
                    case (byte)')': _stack.Push(Array.Empty<object?>()); break;
                    case (byte)'}': _stack.Push(new List<KeyValuePair<object, object>>()); break;
                    case (byte)']': _stack.Push(new List<object?>()); break;
                    case (byte)'N': _stack.Push(null); break;
                    case 0x88: _stack.Push(true); break;
                    case 0x89: _stack.Push(false); break;
                    case (byte)'K': _stack.Push((long)U8()); break;
                    case (byte)'M':
                        _stack.Push((long)BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(_pos)));
                        _pos += 2;
                        break;
                    case (byte)'J': _stack.Push((long)I32()); break;
                    case 0x8a: // LONG1
                    {
                        int n = U8();
                        long v = 0;
                        for (int i = 0; i < n; i++) v |= (long)data[_pos + i] << (8 * i);
                        if (n > 0 && n < 8 && (data[_pos + n - 1] & 0x80) != 0) v -= 1L << (8 * n);
                        _pos += n;
                        _stack.Push(v);
                        break;
                    }
                    case (byte)'G':
                        _stack.Push(BinaryPrimitives.ReadDoubleBigEndian(data.AsSpan(_pos)));
                        _pos += 8;
                        break;
                    case (byte)'X': _stack.Push(Utf8(I32())); break;
                    case 0x8c: _stack.Push(Utf8(U8())); break; // SHORT_BINUNICODE
                    case (byte)'c': _stack.Push(new Global(Line(), Line())); break;
                    case 0x93: // STACK_GLOBAL
                    {
                        string name = (string)_stack.Pop()!, module = (string)_stack.Pop()!;
                        _stack.Push(new Global(module, name));
                        break;
                    }
                    case (byte)'q': _memo[U8()] = _stack.Peek(); break;
                    case (byte)'r': _memo[I32()] = _stack.Peek(); break;
                    case 0x94: _memo[_memo.Count] = _stack.Peek(); break; // MEMOIZE
                    case (byte)'h': _stack.Push(_memo[U8()]); break;
                    case (byte)'j': _stack.Push(_memo[I32()]); break;
                    case (byte)'t': _stack.Push(PopToMark().ToArray()); break;
                    case 0x85: _stack.Push(new[] { _stack.Pop() }); break;
                    case 0x86:
                    {
                        object? b = _stack.Pop(), a = _stack.Pop();
                        _stack.Push(new[] { a, b });
                        break;
                    }
                    case 0x87:
                    {
                        object? c = _stack.Pop(), b = _stack.Pop(), a = _stack.Pop();
                        _stack.Push(new[] { a, b, c });
                        break;
                    }
                    case (byte)'a':
                    {
                        object? v = _stack.Pop();
                        ((List<object?>)_stack.Peek()!).Add(v);
                        break;
                    }
                    case (byte)'e':
                    {
                        var items = PopToMark();
                        ((List<object?>)_stack.Peek()!).AddRange(items);
                        break;
                    }
                    case (byte)'s':
                    {
                        object? v = _stack.Pop(), k = _stack.Pop();
                        AsDict(_stack.Peek()).Add(new(k!, v!));
                        break;
                    }
                    case (byte)'u':
                    {
                        var items = PopToMark();
                        var dict = AsDict(_stack.Peek());
                        for (int i = 0; i + 1 < items.Count; i += 2) dict.Add(new(items[i]!, items[i + 1]!));
                        break;
                    }
                    case (byte)'Q': // BINPERSID
                        _stack.Push(PersistentLoad(_stack.Pop()));
                        break;
                    case (byte)'R':
                    {
                        var args = (object?[])_stack.Pop()!;
                        _stack.Push(Reduce(_stack.Pop(), args));
                        break;
                    }
                    case (byte)'b': // BUILD: OrderedDict state (e.g. _metadata) — no weights, ignore
                        _stack.Pop();
                        break;
                    default:
                        throw new NotSupportedException($"pickle opcode 0x{op:x2} at {_pos - 1} is not supported by this state_dict reader.");
                }
            }
            throw new InvalidDataException("pickle: no STOP.");
        }

        private static List<KeyValuePair<object, object>> AsDict(object? o) =>
            o as List<KeyValuePair<object, object>> ?? throw new NotSupportedException($"pickle: SETITEM(S) on {o?.GetType().Name}.");

        private static object PersistentLoad(object? pid)
        {
            // ('storage', <Global torch.HalfStorage>, key, location, numel)
            if (pid is object?[] { Length: 5 } t && t[0] is "storage" && t[1] is Global g && t[2] is string key && t[4] is long numel)
                return new StorageRef(g.Name, key, numel);
            throw new NotSupportedException("pickle: unsupported persistent id (expected a torch storage).");
        }

        private object Reduce(object? callable, object?[] args)
        {
            if (callable is not Global g) throw new NotSupportedException("pickle: REDUCE on a non-global.");
            switch (g.Module, g.Name)
            {
                case ("collections", "OrderedDict"):
                    return new List<KeyValuePair<object, object>>();
                case ("torch._utils", "_rebuild_tensor_v2"):
                    return RebuildTensor(args);
                default:
                    throw new NotSupportedException($"pickle: refusing to call {g.Module}.{g.Name} (only OrderedDict and _rebuild_tensor_v2 are allowed).");
            }
        }

        private TorchTensor RebuildTensor(object?[] args)
        {
            // _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
            var storage = args[0] as StorageRef ?? throw new NotSupportedException("_rebuild_tensor_v2: first arg is not a storage.");
            long offset = (long)args[1]!;
            int[] shape = ((object?[])args[2]!).Select(x => checked((int)(long)x!)).ToArray();
            long[] stride = ((object?[])args[3]!).Select(x => (long)x!).ToArray();
            long count = 1;
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                if (shape[d] != 1 && stride[d] != count)
                    throw new NotSupportedException($"_rebuild_tensor_v2: non-contiguous tensor (shape [{string.Join(",", shape)}], stride [{string.Join(",", stride)}]).");
                count *= shape[d];
            }

            int elemBytes = storage.Type switch
            {
                "LongStorage" => 8,
                "FloatStorage" or "IntStorage" => 4,
                "HalfStorage" or "BFloat16Storage" => 2,
                _ => throw new NotSupportedException($"storage type {storage.Type} not supported (Float/Half/BFloat16/Long/Int only)."),
            };
            if (!_storages.TryGetValue(storage.Key, out var raw)) _storages[storage.Key] = raw = loadStorage(storage.Key);
            if (raw.Length < storage.Numel * elemBytes || (offset + count) > storage.Numel)
                throw new InvalidDataException($"storage {storage.Key}: {raw.Length} bytes, tensor needs [{offset}, {offset + count}) of {storage.Numel}.");

            var result = new float[count];
            var src = raw.AsSpan(checked((int)(offset * elemBytes)), checked((int)(count * elemBytes)));
            switch (storage.Type)
            {
                case "FloatStorage":
                    for (int i = 0; i < result.Length; i++) result[i] = BinaryPrimitives.ReadSingleLittleEndian(src[(i * 4)..]);
                    break;
                case "HalfStorage":
                    for (int i = 0; i < result.Length; i++) result[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(src[(i * 2)..]);
                    break;
                case "LongStorage":
                    for (int i = 0; i < result.Length; i++) result[i] = BinaryPrimitives.ReadInt64LittleEndian(src[(i * 8)..]);
                    break;
                case "IntStorage":
                    for (int i = 0; i < result.Length; i++) result[i] = BinaryPrimitives.ReadInt32LittleEndian(src[(i * 4)..]);
                    break;
                default:
                    for (int i = 0; i < result.Length; i++)
                        result[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(src[(i * 2)..]) << 16);
                    break;
            }
            return new TorchTensor(shape, result);
        }
    }
}
