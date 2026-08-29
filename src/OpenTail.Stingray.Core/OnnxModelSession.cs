using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OpenTail.Stingray.Core;

/// <summary>
/// Universal generic host for ONNX model graph execution across OpenTail.Stingray.
/// Provides safe availability probing, input metadata auto-filtering, and dynamic tensor execution.
/// </summary>
public sealed class OnnxModelSession : IDisposable
{
    private readonly InferenceSession? _session;
    private readonly string? _modelPath;
    private readonly HashSet<string> _inputNames = new(StringComparer.Ordinal);
    private readonly HashSet<string> _outputNames = new(StringComparer.Ordinal);

    public bool IsAvailable => _session != null;
    public string? ModelPath => _modelPath;
    public IReadOnlyCollection<string> InputNames => _inputNames;
    public IReadOnlyCollection<string> OutputNames => _outputNames;

    public OnnxModelSession(string modelPath, SessionOptions? options = null)
    {
        _modelPath = modelPath;
        if (!File.Exists(modelPath)) return;

        try
        {
            options ??= new SessionOptions
            {
                IntraOpNumThreads = Environment.ProcessorCount,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            _session = new InferenceSession(modelPath, options);
            foreach (var name in _session.InputNames) _inputNames.Add(name);
            foreach (var name in _session.OutputNames) _outputNames.Add(name);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[OnnxModelSession] Failed to initialize ONNX session for '{modelPath}': {ex.Message}");
            _session = null;
        }
    }

    /// <summary>
    /// Safely attempts to load an ONNX model. Returns null if the file does not exist or if the native ONNX runtime fails to initialize.
    /// </summary>
    public static OnnxModelSession? TryLoad(string? modelPath, SessionOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            return null;

        try
        {
            var session = new OnnxModelSession(modelPath, options);
            return session.IsAvailable ? session : null;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Executes the ONNX graph with arbitrary named input tensors and returns the first output tensor as a flat float array.
    /// Only inputs declared in the model's input metadata are passed to the session.
    /// </summary>
    public float[]? RunToFloatArray(params (string Name, Array Data, int[] Shape)[] inputs)
    {
        if (_session == null || inputs.Length == 0) return null;

        var onnxInputs = new List<NamedOnnxValue>(inputs.Length);
        foreach (var (name, data, shape) in inputs)
        {
            if (_inputNames.Contains(name))
            {
                onnxInputs.Add(CreateNamedValue(name, data, shape));
            }
        }

        if (onnxInputs.Count == 0) return null;

        using var results = _session.Run(onnxInputs);
        foreach (var r in results)
        {
            if (r.Value is DenseTensor<float> dt)
            {
                return dt.Buffer.ToArray();
            }
        }

        return null;
    }

    /// <summary>
    /// Executes the ONNX graph with arbitrary named inputs and returns all output tensors by name.
    /// Only inputs declared in the model's input metadata are passed to the session.
    /// </summary>
    public Dictionary<string, float[]> Run(params (string Name, Array Data, int[] Shape)[] inputs)
    {
        var dict = new Dictionary<string, float[]>();
        if (_session == null || inputs.Length == 0) return dict;

        var onnxInputs = new List<NamedOnnxValue>(inputs.Length);
        foreach (var (name, data, shape) in inputs)
        {
            if (_inputNames.Contains(name))
            {
                onnxInputs.Add(CreateNamedValue(name, data, shape));
            }
        }

        if (onnxInputs.Count == 0) return dict;

        using var results = _session.Run(onnxInputs);
        foreach (var r in results)
        {
            if (r.Value is DenseTensor<float> dt)
            {
                dict[r.Name] = dt.Buffer.ToArray();
            }
        }

        return dict;
    }

    /// <summary>
    /// Executes the ONNX graph with arbitrary named input tensors and returns the first output
    /// tensor as a flat int array (for graphs whose output is integer token/index IDs, e.g. a
    /// speech tokenizer, rather than float logits/embeddings).
    /// </summary>
    public int[]? RunToIntArray(params (string Name, Array Data, int[] Shape)[] inputs)
    {
        if (_session == null || inputs.Length == 0) return null;

        var onnxInputs = new List<NamedOnnxValue>(inputs.Length);
        foreach (var (name, data, shape) in inputs)
        {
            if (_inputNames.Contains(name))
            {
                onnxInputs.Add(CreateNamedValue(name, data, shape));
            }
        }

        if (onnxInputs.Count == 0) return null;

        using var results = _session.Run(onnxInputs);
        foreach (var r in results)
        {
            switch (r.Value)
            {
                case DenseTensor<int> di:
                    return di.Buffer.ToArray();
                case DenseTensor<long> dl:
                    var arr = new int[dl.Buffer.Length];
                    for (int i = 0; i < arr.Length; i++) arr[i] = checked((int)dl.Buffer.Span[i]);
                    return arr;
            }
        }

        return null;
    }

    private static NamedOnnxValue CreateNamedValue(string name, Array data, int[] shape)
    {
        return data switch
        {
            float[] f  => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(f, shape)),
            long[] l   => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(l, shape)),
            int[] i    => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(i, shape)),
            byte[] b   => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<byte>(b, shape)),
            _          => throw new NotSupportedException($"Tensor element type '{data.GetType()}' is not supported.")
        };
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
