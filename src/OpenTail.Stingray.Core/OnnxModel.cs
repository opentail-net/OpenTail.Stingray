
namespace OpenTail.Stingray.Core;

/// <summary>
/// One ONNX initializer tensor (TensorProto), with its raw little-endian bytes left undecoded
/// (matching <see cref="GgufTensorInfo"/>'s convention of exposing raw bytes + dtype rather than
/// a pre-materialized float[]).
/// </summary>
public readonly struct OnnxTensor
{
    public required string Name { get; init; }
    public required int[] Dims { get; init; }        // in ONNX/torch order, NOT reversed like GGUF
    public required int DataType { get; init; }       // TensorProto.DataType enum: 1=float32, 6=int32, 7=int64, ...
    public required byte[] RawData { get; init; }

    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (int d in Dims) n *= d;
            return Dims.Length == 0 ? 1 : n;
        }
    }
}

/// <summary>One ONNX graph node (NodeProto): name, op type, and input/output tensor names.</summary>
public sealed record OnnxNode(string Name, string OpType, IReadOnlyList<string> Inputs, IReadOnlyList<string> Outputs);

/// <summary>
/// Minimal ONNX (protobuf) reader: extracts a graph's initializer tensors and input/output
/// names, without pulling in a full protobuf library or ONNX Runtime (neither is referenced
/// anywhere in this codebase -- see docs/audio-review-progress.md's Piper section for why this
/// exists: Piper ships real weights only as .onnx, and this codebase otherwise only knows GGUF/
/// safetensors). Implements just enough of the protobuf wire format (varint/length-delimited
/// field parsing) to walk ModelProto -&gt; GraphProto -&gt; initializer[]/input[]/output[] -- the
/// actual model math is still hand-ported in C#, this only replaces "load the weight bytes".
/// Field numbers are from onnx.proto (stable across ONNX IR versions):
///   ModelProto.graph = 7; GraphProto.node = 1, initializer = 5, input = 11, output = 12;
///   TensorProto.dims = 1, data_type = 2, name = 8, raw_data = 9; ValueInfoProto.name = 1.
/// </summary>
public sealed class OnnxModel
{
    public IReadOnlyDictionary<string, OnnxTensor> Initializers { get; }
    public IReadOnlyList<string> GraphInputs { get; }
    public IReadOnlyList<string> GraphOutputs { get; }

    /// <summary>
    /// Graph nodes, keyed by name. Needed beyond just the initializers because some exporters
    /// (e.g. PyTorch's weight_norm reparametrization) fuse a module's real weight into an
    /// anonymously-named constant (e.g. "onnx::Conv_8240") rather than a module-path-named
    /// initializer -- the only way to find "which tensor is this specific Conv's weight" is to
    /// walk the node graph and read its input list.
    /// </summary>
    public IReadOnlyDictionary<string, OnnxNode> Nodes { get; }

    private OnnxModel(Dictionary<string, OnnxTensor> initializers, List<string> inputs, List<string> outputs, Dictionary<string, OnnxNode> nodes)
    {
        Initializers = initializers;
        GraphInputs = inputs;
        GraphOutputs = outputs;
        Nodes = nodes;
    }

    public static OnnxModel Open(string path) => Parse(File.ReadAllBytes(path));

    public static OnnxModel Parse(byte[] modelBytes)
    {
        var initializers = new Dictionary<string, OnnxTensor>(StringComparer.Ordinal);
        var inputs = new List<string>();
        var outputs = new List<string>();
        var nodes = new Dictionary<string, OnnxNode>(StringComparer.Ordinal);

        foreach (var f in ReadFields(modelBytes, 0, modelBytes.Length))
        {
            if (f.Field == 7 && f.WireType == 2) // ModelProto.graph
                ParseGraph(modelBytes, f.Start, f.Length, initializers, inputs, outputs, nodes);
        }

        return new OnnxModel(initializers, inputs, outputs, nodes);
    }

    public OnnxTensor GetTensor(string name) =>
        Initializers.TryGetValue(name, out var t) ? t : throw new InvalidDataException($"ONNX model missing required initializer '{name}'.");

    public OnnxTensor? TryGetTensor(string name) => Initializers.TryGetValue(name, out var t) ? t : null;

    /// <summary>
    /// Resolves an anonymously-named weight (see <see cref="Nodes"/>'s doc comment) by finding
    /// the named node and reading one of its input tensor names, then looking that up in
    /// Initializers. Throws if the node or the resolved initializer doesn't exist.
    /// </summary>
    public OnnxTensor GetWeightViaNode(string nodeName, int inputIndex)
    {
        if (!Nodes.TryGetValue(nodeName, out var node))
            throw new InvalidDataException($"ONNX model missing required node '{nodeName}'.");
        if (inputIndex >= node.Inputs.Count)
            throw new InvalidDataException($"ONNX node '{nodeName}' has only {node.Inputs.Count} inputs, requested index {inputIndex}.");
        return GetTensor(node.Inputs[inputIndex]);
    }

    /// <summary>Decodes a tensor's raw bytes to float32, converting from DOUBLE/FLOAT16 if needed.</summary>
    public static float[] ToFloat32(OnnxTensor t)
    {
        var dst = new float[t.ElementCount];
        switch (t.DataType)
        {
            case 1: // FLOAT
                Buffer.BlockCopy(t.RawData, 0, dst, 0, dst.Length * 4);
                break;
            case 11: // DOUBLE
                for (int i = 0; i < dst.Length; i++) dst[i] = (float)BitConverter.ToDouble(t.RawData, i * 8);
                break;
            case 10: // FLOAT16
                for (int i = 0; i < dst.Length; i++) dst[i] = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(t.RawData, i * 2));
                break;
            default:
                throw new NotSupportedException($"OnnxModel.ToFloat32: unsupported source DataType {t.DataType} for tensor.");
        }
        return dst;
    }

    /// <summary>Decodes an INT64 tensor's raw bytes to int[] (values must fit in int32).</summary>
    public static int[] ToInt32(OnnxTensor t)
    {
        var dst = new int[t.ElementCount];
        if (t.DataType == 7) // INT64
        {
            for (int i = 0; i < dst.Length; i++) dst[i] = checked((int)BitConverter.ToInt64(t.RawData, i * 8));
        }
        else if (t.DataType == 6) // INT32
        {
            Buffer.BlockCopy(t.RawData, 0, dst, 0, dst.Length * 4);
        }
        else
        {
            throw new NotSupportedException($"OnnxModel.ToInt32: unsupported source DataType {t.DataType}.");
        }
        return dst;
    }

    private static void ParseGraph(byte[] buf, int start, int len, Dictionary<string, OnnxTensor> initializers, List<string> inputs, List<string> outputs, Dictionary<string, OnnxNode> nodes)
    {
        foreach (var f in ReadFields(buf, start, start + len))
        {
            if (f.Field == 1 && f.WireType == 2) // GraphProto.node
            {
                var n = ParseNode(buf, f.Start, f.Length, initializers, inputs, outputs, nodes);
                if (n is { } node && node.Name.Length > 0) nodes[node.Name] = node;
            }
            else if (f.Field == 5 && f.WireType == 2) // GraphProto.initializer
            {
                var t = ParseTensor(buf, f.Start, f.Length);
                if (t is { } tensor) initializers[tensor.Name] = tensor;
            }
            else if (f.Field == 11 && f.WireType == 2) // GraphProto.input
            {
                if (ParseValueInfoName(buf, f.Start, f.Length) is { } name) inputs.Add(name);
            }
            else if (f.Field == 12 && f.WireType == 2) // GraphProto.output
            {
                if (ParseValueInfoName(buf, f.Start, f.Length) is { } name) outputs.Add(name);
            }
        }
    }

    /// <summary>
    /// Also recurses into node attributes of type GRAPH (AttributeProto.g, field 6) -- an `If`
    /// node's then_branch/else_branch is a full nested GraphProto with its own nodes/
    /// initializers, needed for e.g. Silero VAD's real weights, which live entirely inside an
    /// `If(sr==16000)` subgraph rather than the top-level graph (see docs/audio-review-
    /// progress.md's Silero VAD section). Folds the subgraph's nodes/initializers into the SAME
    /// dictionaries the top-level graph uses -- safe because every name this codebase's real
    /// models use inside a subgraph is already uniquely prefixed by the exporter (e.g.
    /// `If_0_then_branch__Inline_0__stft.forward_basis_buffer`), so there is no collision risk
    /// with top-level names, and flattening is strictly additive: existing callers that only
    /// ever used single-branch models (Piper/MeloTTS, no `If` nodes) see zero behavior change.
    /// </summary>
    private static OnnxNode? ParseNode(byte[] buf, int start, int len, Dictionary<string, OnnxTensor> initializers, List<string> inputs, List<string> outputs, Dictionary<string, OnnxNode> nodes)
    {
        var nodeInputs = new List<string>();
        var nodeOutputs = new List<string>();
        string name = "";
        string opType = "";

        foreach (var f in ReadFields(buf, start, start + len))
        {
            switch (f.Field)
            {
                case 1 when f.WireType == 2: // input
                    nodeInputs.Add(Encoding.UTF8.GetString(buf, f.Start, f.Length));
                    break;
                case 2 when f.WireType == 2: // output
                    nodeOutputs.Add(Encoding.UTF8.GetString(buf, f.Start, f.Length));
                    break;
                case 3 when f.WireType == 2: // name
                    name = Encoding.UTF8.GetString(buf, f.Start, f.Length);
                    break;
                case 4 when f.WireType == 2: // op_type
                    opType = Encoding.UTF8.GetString(buf, f.Start, f.Length);
                    break;
                case 5 when f.WireType == 2: // attribute (AttributeProto)
                    ParseAttributeForSubgraph(buf, f.Start, f.Length, initializers, inputs, outputs, nodes);
                    break;
            }
        }
        if (name.Length == 0) return null;
        return new OnnxNode(name, opType, nodeInputs, nodeOutputs);
    }

    /// <summary>Looks for AttributeProto.g (field 6, a single embedded GraphProto -- covers `If`'s then_branch/else_branch) and recurses ParseGraph on it if present. Other attribute kinds (ints, floats, tensors, repeated graphs) are irrelevant to weight loading and skipped.</summary>
    private static void ParseAttributeForSubgraph(byte[] buf, int start, int len, Dictionary<string, OnnxTensor> initializers, List<string> inputs, List<string> outputs, Dictionary<string, OnnxNode> nodes)
    {
        foreach (var f in ReadFields(buf, start, start + len))
        {
            if (f.Field == 6 && f.WireType == 2) // AttributeProto.g
                ParseGraph(buf, f.Start, f.Length, initializers, inputs, outputs, nodes);
        }
    }

    private static string? ParseValueInfoName(byte[] buf, int start, int len)
    {
        foreach (var f in ReadFields(buf, start, start + len))
            if (f.Field == 1 && f.WireType == 2) return Encoding.UTF8.GetString(buf, f.Start, f.Length);
        return null;
    }

    private static OnnxTensor? ParseTensor(byte[] buf, int start, int len)
    {
        var dims = new List<int>();
        int dataType = 0;
        string? name = null;
        byte[]? rawData = null;

        foreach (var f in ReadFields(buf, start, start + len))
        {
            switch (f.Field)
            {
                case 1 when f.WireType == 2: // dims, packed varint
                {
                    int p = f.Start;
                    int end = f.Start + f.Length;
                    while (p < end) dims.Add((int)ReadVarint(buf, ref p));
                    break;
                }
                case 1 when f.WireType == 0: // dims, unpacked (single varint per field occurrence)
                    dims.Add((int)f.VarintValue);
                    break;
                case 2 when f.WireType == 0: // data_type
                    dataType = (int)f.VarintValue;
                    break;
                case 8 when f.WireType == 2: // name
                    name = Encoding.UTF8.GetString(buf, f.Start, f.Length);
                    break;
                case 9 when f.WireType == 2: // raw_data
                    rawData = new byte[f.Length];
                    Array.Copy(buf, f.Start, rawData, 0, f.Length);
                    break;
            }
        }

        if (name is null || rawData is null) return null; // skip tensors with no raw_data (e.g. float_data-encoded scalars we don't need)
        return new OnnxTensor { Name = name, Dims = dims.ToArray(), DataType = dataType, RawData = rawData };
    }

    private readonly record struct ProtoField(int Field, byte WireType, int Start, int Length, long VarintValue);

    private static List<ProtoField> ReadFields(byte[] buf, int start, int end)
    {
        var result = new List<ProtoField>();
        int pos = start;
        while (pos < end)
        {
            long tag = ReadVarint(buf, ref pos);
            int field = (int)(tag >> 3);
            byte wireType = (byte)(tag & 7);
            switch (wireType)
            {
                case 0: // varint
                    long v = ReadVarint(buf, ref pos);
                    result.Add(new ProtoField(field, wireType, pos, 0, v));
                    break;
                case 1: // 64-bit fixed
                    result.Add(new ProtoField(field, wireType, pos, 8, 0));
                    pos += 8;
                    break;
                case 2: // length-delimited
                {
                    long lenLong = ReadVarint(buf, ref pos);
                    int payloadLen = checked((int)lenLong);
                    result.Add(new ProtoField(field, wireType, pos, payloadLen, 0));
                    pos += payloadLen;
                    break;
                }
                case 5: // 32-bit fixed
                    result.Add(new ProtoField(field, wireType, pos, 4, 0));
                    pos += 4;
                    break;
                default:
                    throw new NotSupportedException($"Unsupported protobuf wire type {wireType} at offset {pos}.");
            }
        }
        return result;
    }

    private static long ReadVarint(byte[] buf, ref int pos)
    {
        long result = 0;
        int shift = 0;
        while (true)
        {
            byte b = buf[pos++];
            result |= (long)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }
}
