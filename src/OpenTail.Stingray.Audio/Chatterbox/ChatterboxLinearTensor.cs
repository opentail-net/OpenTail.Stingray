using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Encapsulates a linear layer weight (matrix + bias) for Chatterbox-Turbo T3 acoustic LM.
/// Supports zero-copy memory-mapped Q4_K evaluation without prior dequantization to float32 RAM.
/// </summary>
public sealed unsafe class ChatterboxLinearTensor : IDisposable
{
    private readonly GgufModel _model;
    private readonly GgufTensorInfo _info;
    private float[]? _floatWeight;
    private GCHandle _biasPin;
    private readonly float* _biasPtr;

    public DType DType { get; }
    public byte* RawDataPtr { get; }
    public float[] Bias { get; }
    public float* BiasPtr => _biasPtr;
    public int OutFeatures { get; }
    public int InFeatures { get; }

    public ChatterboxLinearTensor(GgufModel model, string weightName, string biasName, int outFeatures, int inFeatures)
    {
        _model = model;
        OutFeatures = outFeatures;
        InFeatures = inFeatures;

        _info = model.FindTensor(weightName) ?? throw new InvalidDataException($"Chatterbox T3 GGUF missing required tensor '{weightName}'.");
        DType = _info.DType;

        if (DType == DType.Q4_K)
        {
            RawDataPtr = model.GetTensorDataPtr(_info);
            _floatWeight = null;
        }
        else if (DType == DType.Float32)
        {
            RawDataPtr = model.GetTensorDataPtr(_info);
            _floatWeight = null;
        }
        else
        {
            RawDataPtr = null;
            var bytes = model.GetTensorData(_info);
            _floatWeight = new float[_info.ElementCount];
            Dequantize.ToFloat32(bytes, _floatWeight, _info.DType, _info.ElementCount);
        }

        var biasInfo = model.FindTensor(biasName) ?? throw new InvalidDataException($"Chatterbox T3 GGUF missing required tensor '{biasName}'.");
        var biasBytes = model.GetTensorData(biasInfo);
        Bias = new float[biasInfo.ElementCount];
        Dequantize.ToFloat32(biasBytes, Bias, biasInfo.DType, biasInfo.ElementCount);
        _biasPin = GCHandle.Alloc(Bias, GCHandleType.Pinned);
        _biasPtr = (float*)_biasPin.AddrOfPinnedObject();
    }

    public float[] GetOrDequantizeFloat()
    {
        if (_floatWeight != null) return _floatWeight;
        var bytes = _model.GetTensorData(_info);
        _floatWeight = new float[_info.ElementCount];
        Dequantize.ToFloat32(bytes, _floatWeight, _info.DType, _info.ElementCount);
        return _floatWeight;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MatVec(float* output, float* input)
    {
        if (DType == DType.Q4_K)
        {
            SimdKernels.MatVecQ4K(output, RawDataPtr, _biasPtr, input, OutFeatures, InFeatures);
        }
        else if (DType == DType.Float32 && RawDataPtr != null)
        {
            SimdKernels.MatVecF32(output, (float*)RawDataPtr, _biasPtr, input, OutFeatures, InFeatures);
        }
        else
        {
            fixed (float* wp = GetOrDequantizeFloat())
            {
                SimdKernels.MatVecF32(output, wp, _biasPtr, input, OutFeatures, InFeatures);
            }
        }
    }

    public void Dispose()
    {
        if (_biasPin.IsAllocated)
        {
            _biasPin.Free();
        }
    }
}
