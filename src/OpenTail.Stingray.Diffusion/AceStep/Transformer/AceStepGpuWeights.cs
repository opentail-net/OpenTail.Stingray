using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.AceStep.Transformer;

/// <summary>
/// GPU-resident VRAM representation of ACE-Step Turbo DiT weights.
/// Holds all 24 transformer layers (SelfAttn Q/K/V/O, CrossAttn Q/K/V/O, SwiGLU FFN),
/// input/output projections, condition embedder, and norm weights resident in VRAM as FP16.
/// Structured for zero-allocation reuse across every Euler ODE step.
/// </summary>
public sealed class AceStepGpuWeights : IDisposable
{
    private readonly IComputeBackend _backend;
    private bool _disposed;

    public sealed class BlockWeights : IDisposable
    {
        private readonly IComputeBackend _backend;

        public float[] ScaleShiftTable { get; }
        public CoreTensor SelfAttnNormW { get; }
        public CoreTensor SelfAttnQkvW { get; }
        public CoreTensor SelfAttnOW { get; }
        public CoreTensor SelfAttnQNormW { get; }
        public CoreTensor SelfAttnKNormW { get; }

        public CoreTensor CrossAttnNormW { get; }
        public CoreTensor CrossAttnQW { get; }
        public CoreTensor CrossAttnKW { get; }
        public CoreTensor CrossAttnVW { get; }
        public CoreTensor CrossAttnOW { get; }
        public CoreTensor CrossAttnQNormW { get; }
        public CoreTensor CrossAttnKNormW { get; }

        public CoreTensor MlpNormW { get; }
        public CoreTensor MlpGateUpW { get; }
        public CoreTensor MlpDownW { get; }

        public BlockWeights(IComputeBackend backend, AceStepDiTLayerWeights lw, int hidden, int qDim, int kvDim, int ffn, int headDim)
        {
            _backend = backend;
            ScaleShiftTable = lw.ScaleShiftTable;

            SelfAttnNormW = backend.Upload(lw.SelfAttnNormWeight, TensorShape.D1(hidden), exact: true);
            // Fuse SelfAttn Q, K, V: [4096, hidden]
            int qkvRows = qDim + 2 * kvDim;
            var qkvData = new float[qkvRows * hidden];
            Array.Copy(lw.SelfAttn.QWeight.F32!, 0, qkvData, 0, qDim * hidden);
            Array.Copy(lw.SelfAttn.KWeight.F32!, 0, qkvData, qDim * hidden, kvDim * hidden);
            Array.Copy(lw.SelfAttn.VWeight.F32!, 0, qkvData, (qDim + kvDim) * hidden, kvDim * hidden);
            SelfAttnQkvW = UploadWeight(backend, qkvData, TensorShape.D2(qkvRows, hidden));

            SelfAttnOW = UploadWeight(backend, lw.SelfAttn.OWeight.F32!, TensorShape.D2(hidden, qDim));
            SelfAttnQNormW = backend.Upload(lw.SelfAttn.QNormWeight, TensorShape.D1(headDim), exact: true);
            SelfAttnKNormW = backend.Upload(lw.SelfAttn.KNormWeight, TensorShape.D1(headDim), exact: true);

            CrossAttnNormW = backend.Upload(lw.CrossAttnNormWeight, TensorShape.D1(hidden), exact: true);
            CrossAttnQW = UploadWeight(backend, lw.CrossAttn.QWeight.F32!, TensorShape.D2(qDim, hidden));
            CrossAttnKW = UploadWeight(backend, lw.CrossAttn.KWeight.F32!, TensorShape.D2(kvDim, hidden));
            CrossAttnVW = UploadWeight(backend, lw.CrossAttn.VWeight.F32!, TensorShape.D2(kvDim, hidden));
            CrossAttnOW = UploadWeight(backend, lw.CrossAttn.OWeight.F32!, TensorShape.D2(hidden, qDim));
            CrossAttnQNormW = backend.Upload(lw.CrossAttn.QNormWeight, TensorShape.D1(headDim), exact: true);
            CrossAttnKNormW = backend.Upload(lw.CrossAttn.KNormWeight, TensorShape.D1(headDim), exact: true);

            MlpNormW = backend.Upload(lw.MlpNormWeight, TensorShape.D1(hidden), exact: true);
            // Fuse MLP Gate and Up: [12288, hidden]
            var gateUpData = new float[2 * ffn * hidden];
            Array.Copy(lw.MlpGateWeight.F32!, 0, gateUpData, 0, ffn * hidden);
            Array.Copy(lw.MlpUpWeight.F32!, 0, gateUpData, ffn * hidden, ffn * hidden);
            MlpGateUpW = UploadWeight(backend, gateUpData, TensorShape.D2(2 * ffn, hidden));

            MlpDownW = UploadWeight(backend, lw.MlpDownWeight.F32!, TensorShape.D2(hidden, ffn));
        }

        public void Dispose()
        {
            _backend.Free(SelfAttnNormW);
            _backend.Free(SelfAttnQkvW);
            _backend.Free(SelfAttnOW);
            _backend.Free(SelfAttnQNormW);
            _backend.Free(SelfAttnKNormW);

            _backend.Free(CrossAttnNormW);
            _backend.Free(CrossAttnQW);
            _backend.Free(CrossAttnKW);
            _backend.Free(CrossAttnVW);
            _backend.Free(CrossAttnOW);
            _backend.Free(CrossAttnQNormW);
            _backend.Free(CrossAttnKNormW);

            _backend.Free(MlpNormW);
            _backend.Free(MlpGateUpW);
            _backend.Free(MlpDownW);
        }
    }

    public BlockWeights[] Layers { get; }

    public CoreTensor ProjInW { get; }
    public CoreTensor ProjInB { get; }
    public CoreTensor ProjOutW { get; }
    public CoreTensor ProjOutB { get; }

    public CoreTensor ConditionEmbedderW { get; }
    public CoreTensor ConditionEmbedderB { get; }

    public CoreTensor NormOutW { get; }
    public float[] FinalScaleShiftTable { get; }

    public AceStepDiTWeights CpuWeights { get; }

    public AceStepGpuWeights(IComputeBackend backend, AceStepDiTWeights cpuWeights)
    {
        _backend = backend;
        CpuWeights = cpuWeights;

        int hidden = AceStepConfig.HiddenSize; // 2048
        int inCh = AceStepConfig.InChannels; // 192
        int patch = AceStepConfig.PatchSize; // 2
        int inDim = inCh * patch; // 384
        int outCh = AceStepConfig.AudioAcousticHiddenDim; // 64
        int outDim = outCh * patch; // 128
        int qDim = AceStepConfig.NumAttentionHeads * AceStepConfig.HeadDim; // 2048
        int kvDim = AceStepConfig.NumKeyValueHeads * AceStepConfig.HeadDim; // 1024
        int ffn = AceStepConfig.IntermediateSize; // 6144
        int headDim = AceStepConfig.HeadDim; // 128

        // 1. Reorder ProjInWeight [hidden, inCh, patch] -> [hidden, inDim(384)]
        var reorderedProjIn = new float[hidden * inDim];
        for (int oc = 0; oc < hidden; oc++)
        {
            int wBase = oc * inCh * patch;
            for (int ic = 0; ic < inCh; ic++)
            {
                for (int k = 0; k < patch; k++)
                {
                    reorderedProjIn[oc * inDim + (k * inCh + ic)] = cpuWeights.ProjInWeight[wBase + ic * patch + k];
                }
            }
        }
        ProjInW = UploadWeight(backend, reorderedProjIn, TensorShape.D2(hidden, inDim));
        ProjInB = backend.Upload(cpuWeights.ProjInBias, TensorShape.D1(hidden), exact: true);

        // 2. Reorder ProjOutWeight [hidden(2048), outCh(64), patch(2)] -> [outDim(128), hidden(2048)]
        var reorderedProjOut = new float[outDim * hidden];
        var projOutBiasFull = new float[outDim];
        for (int k = 0; k < patch; k++)
        {
            for (int oc = 0; oc < outCh; oc++)
            {
                int rowIdx = k * outCh + oc;
                projOutBiasFull[rowIdx] = cpuWeights.ProjOutBias[oc];
                for (int ic = 0; ic < hidden; ic++)
                {
                    reorderedProjOut[rowIdx * hidden + ic] = cpuWeights.ProjOutWeight[(ic * outCh + oc) * patch + k];
                }
            }
        }
        ProjOutW = UploadWeight(backend, reorderedProjOut, TensorShape.D2(outDim, hidden));
        ProjOutB = backend.Upload(projOutBiasFull, TensorShape.D1(outDim), exact: true);

        // 3. Condition Embedder
        ConditionEmbedderW = UploadWeight(backend, cpuWeights.ConditionEmbedderWeight.F32!, TensorShape.D2(hidden, hidden));
        ConditionEmbedderB = backend.Upload(cpuWeights.ConditionEmbedderBias, TensorShape.D1(hidden), exact: true);

        // 4. Final Norm + ScaleShiftTable
        NormOutW = backend.Upload(cpuWeights.NormOutWeight, TensorShape.D1(hidden), exact: true);
        FinalScaleShiftTable = cpuWeights.ScaleShiftTable;

        // 5. All 24 Transformer Layers
        Layers = new BlockWeights[cpuWeights.Layers.Length];
        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i] = new BlockWeights(backend, cpuWeights.Layers[i], hidden, qDim, kvDim, ffn, headDim);
        }
    }

    private static CoreTensor UploadWeight(IComputeBackend backend, float[] f32, TensorShape shape)
    {
        var h = new Half[f32.Length];
        for (int i = 0; i < f32.Length; i++) h[i] = (Half)f32[i];
        return backend.UploadHalf(h, shape);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _backend.Free(ProjInW);
        _backend.Free(ProjInB);
        _backend.Free(ProjOutW);
        _backend.Free(ProjOutB);
        _backend.Free(ConditionEmbedderW);
        _backend.Free(ConditionEmbedderB);
        _backend.Free(NormOutW);

        foreach (var layer in Layers)
        {
            layer.Dispose();
        }
    }
}
