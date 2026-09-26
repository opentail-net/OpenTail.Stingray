using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// GPU (Vulkan) execution of HunyuanVideo's double- and single-stream blocks — the ~13B-parameter
/// trunk that is ~all of a denoising step. The small per-step pieces (timestep/CLIP/guidance
/// embedding, token refiner, img_in, final layer) stay on the CPU path in <see cref="Forward"/>.
/// </summary>
/// <remarks>
/// <para>Structure follows <see cref="FluxDiT"/>'s GPU blocks (same ops: AdaLNModulate, fused QKV
/// unpack, QKNorm, interleaved-pair RoPE, tiled attention, ScaleGateAdd, single-block linear1 /
/// concat / linear2), with HunyuanVideo's differences from FLUX.1 applied exactly as the CPU path
/// does them: affine-free <b>LayerNorm</b> (not RMSNorm) before every modulation, sequence order
/// <b>image first, text second</b>, RoPE on the image rows only, and biases on every linear.</para>
///
/// <para>Weights stay in the checkpoint's own format in VRAM: fp8 E4M3 via
/// <c>Shaders.SgemmFp8W</c> (1 byte/weight — the 720p cfg-distilled checkpoint is ~13 GB), bf16 via
/// <c>Shaders.SgemmBf16W</c>, otherwise fp16/fp32 through the backend's usual path. Both raw paths
/// widen exactly, so the GPU multiplies the same weight values the CPU path decodes.</para>
///
/// <para>Blocks are submitted two per command buffer, like FluxDiT, so the shared iGPU still gets a
/// submission point every few blocks for the desktop compositor.</para>
/// </remarks>
public sealed partial class HunyuanVideoModel
{
    /// <summary>Run the transformer blocks on <c>backend</c> when it supports the needed ops. Defaults
    /// to on when a Vulkan-class backend was given; <c>STINGRAY_HUNYUAN_GPU=0</c> forces the CPU path.</summary>
    public bool UseGpu { get; set; }

    private HunyuanGpuWeights? _gpuW;
    private HunyuanGpuWorkspace? _gpuWs;
    private (float[] src, CoreTensor cos, CoreTensor sin)? _gpuRope;

    private bool GpuCapable =>
        _backend is IVisionOpsBackend && _backend is IImageOpsBackend;

    private void InitGpuDefault() =>
        UseGpu = GpuCapable && Environment.GetEnvironmentVariable("STINGRAY_HUNYUAN_GPU") != "0";

    /// <summary>
    /// Double + single blocks on the GPU. Takes the CPU-computed img/txt tokens and the modulation
    /// vector; returns the image tokens after the last block (host), ready for the final layer.
    /// </summary>
    private float[] BlocksGpu(float[] imgTokens, float[] txtTokens, float[] tEmb,
        float[] cos, float[] sin, int numImg, int numTxt)
    {
        var vis = (IVisionOpsBackend)_backend!;
        var img = (IImageOpsBackend)_backend!;
        int d = _dim;

        _gpuW ??= new HunyuanGpuWeights(this, _backend!);
        if (_gpuWs is null || _gpuWs.NumImg != numImg || _gpuWs.NumTxt != numTxt)
        {
            _gpuWs?.Dispose();
            _gpuWs = new HunyuanGpuWorkspace(_backend!, numImg, numTxt, d);
        }
        var ws = _gpuWs;

        // RoPE table: the CPU table stores each pair's cos/sin twice ([tok, headDim]); the GPU
        // Flux2DRoPE op takes one value per pair ([tok, headDim/2]) — same rotation.
        // The CPU RoPE table itself is cached across steps (same array), so this re-uploads only
        // when the shape changes.
        if (_gpuRope is not { } r || !ReferenceEquals(r.src, cos))
        {
            if (_gpuRope is { } old) { _backend!.Free(old.cos); _backend!.Free(old.sin); }
            int half = _headDim / 2;
            var c = new float[numImg * half];
            var s = new float[numImg * half];
            for (int t = 0; t < numImg; t++)
                for (int i = 0; i < half; i++)
                {
                    c[t * half + i] = cos[t * _headDim + 2 * i];
                    s[t * half + i] = sin[t * _headDim + 2 * i];
                }
            _gpuRope = (cos,
                _backend!.Upload(c, TensorShape.D2(numImg, half), exact: true),
                _backend!.Upload(s, TensorShape.D2(numImg, half), exact: true));
        }
        var (_, ropeCos, ropeSin) = _gpuRope.Value;

        // Per-step inputs.
        var siluVec = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(siluVec);
        using var vecGpu = _backend!.Upload(siluVec, TensorShape.D2(1, d), exact: true);
        ws.SetStreams(_backend!, imgTokens, txtTokens);

        const int chunk = 2;
        for (int b0 = 0; b0 < _depthDouble; b0 += chunk)
        {
            img.BeginBatch();
            int b1 = Math.Min(_depthDouble, b0 + chunk);
            for (int b = b0; b < b1; b++)
                DoubleBlockGpu(ws, _gpuW.Double[b], vecGpu, ropeCos, ropeSin, vis, img);
            if (b1 == _depthDouble && _depthSingle > 0)
                vis.FluxConcatTxtImg(ws.ImgHidden, ws.TxtHidden, ws.X, numImg, numTxt, d);   // image first
            img.EndBatch();
        }

        var result = new float[numImg * d];
        if (_depthSingle == 0)
        {
            _backend!.Download(ws.ImgHidden, result);
            return result;
        }

        for (int b0 = 0; b0 < _depthSingle; b0 += chunk)
        {
            img.BeginBatch();
            int b1 = Math.Min(_depthSingle, b0 + chunk);
            for (int b = b0; b < b1; b++)
                SingleBlockGpu(ws, _gpuW.Single[b], vecGpu, ropeCos, ropeSin, vis, img);
            if (b1 == _depthSingle)
                vis.FluxSliceImg(ws.X, ws.ImgHidden, 0, numImg, d);
            img.EndBatch();
        }
        _backend!.Download(ws.ImgHidden, result);
        return result;
    }

    private void DoubleBlockGpu(HunyuanGpuWorkspace ws, HunyuanGpuWeights.DoubleBlock w, CoreTensor vec,
        CoreTensor ropeCos, CoreTensor ropeSin, IVisionOpsBackend vis, IImageOpsBackend img)
    {
        int d = _dim, nImg = ws.NumImg, nTxt = ws.NumTxt, nSeq = nImg + nTxt;
        int nh = _numHeads, hd = _headDim;

        Linear(vis, img, ws.ImgMod, vec, w.ImgMod, 1, d, 6 * d);
        Linear(vis, img, ws.TxtMod, vec, w.TxtMod, 1, d, 6 * d);

        // Attention: LN -> modulate(shift 0, scale 1) -> qkv -> q/k RMSNorm -> RoPE (img only).
        vis.AdaLNModulate(ws.NormedImg, ws.ImgHidden, ws.ImgMod, nImg, d, shiftOffset: 0, scaleOffset: d, isRmsNorm: false, eps: 1e-6f);
        vis.AdaLNModulate(ws.NormedTxt, ws.TxtHidden, ws.TxtMod, nTxt, d, shiftOffset: 0, scaleOffset: d, isRmsNorm: false, eps: 1e-6f);

        Linear(vis, img, ws.QkvImg, ws.NormedImg, w.ImgQkv, nImg, d, 3 * d);
        vis.FluxUnpackQkv(ws.QkvImg, ws.Q, ws.K, ws.V, nImg, d, dstTokenOffset: 0);
        if (w.ImgQNorm is not null) vis.QKNorm(ws.Q, ws.K, w.ImgQNorm, w.ImgKNorm!, nImg, nh, hd, eps: 1e-6f, startToken: 0);

        Linear(vis, img, ws.QkvTxt, ws.NormedTxt, w.TxtQkv, nTxt, d, 3 * d);
        vis.FluxUnpackQkv(ws.QkvTxt, ws.Q, ws.K, ws.V, nTxt, d, dstTokenOffset: nImg);
        if (w.TxtQNorm is not null) vis.QKNorm(ws.Q, ws.K, w.TxtQNorm, w.TxtKNorm!, nTxt, nh, hd, eps: 1e-6f, startToken: nImg);

        vis.Flux2DRoPE(ws.Q, ws.K, ropeCos, ropeSin, startToken: 0, tokenCount: nImg, nh, hd);
        img.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, nSeq, nSeq, nh, hd);

        LinearRows(vis, img, ws.OutImg, ws.AttnOut, w.ImgProj, nImg, d, d, rowOffset: 0);
        LinearRows(vis, img, ws.OutTxt, ws.AttnOut, w.TxtProj, nTxt, d, d, rowOffset: nImg);
        vis.ScaleGateAdd(ws.ImgHidden, ws.OutImg, ws.ImgMod, nImg, d, gateOffset: 2 * d);
        vis.ScaleGateAdd(ws.TxtHidden, ws.OutTxt, ws.TxtMod, nTxt, d, gateOffset: 2 * d);

        // MLP: LN -> modulate(shift 3, scale 4) -> fc1 -> GELU(tanh) -> fc2 -> gate 5.
        vis.AdaLNModulate(ws.NormedImg, ws.ImgHidden, ws.ImgMod, nImg, d, shiftOffset: 3 * d, scaleOffset: 4 * d, isRmsNorm: false, eps: 1e-6f);
        Linear(vis, img, ws.MlpImg, ws.NormedImg, w.ImgFc1, nImg, d, 4 * d);
        vis.VisionGeluInPlace(ws.MlpImg);
        Linear(vis, img, ws.OutImg, ws.MlpImg, w.ImgFc2, nImg, 4 * d, d);
        vis.ScaleGateAdd(ws.ImgHidden, ws.OutImg, ws.ImgMod, nImg, d, gateOffset: 5 * d);

        vis.AdaLNModulate(ws.NormedTxt, ws.TxtHidden, ws.TxtMod, nTxt, d, shiftOffset: 3 * d, scaleOffset: 4 * d, isRmsNorm: false, eps: 1e-6f);
        Linear(vis, img, ws.MlpTxt, ws.NormedTxt, w.TxtFc1, nTxt, d, 4 * d);
        vis.VisionGeluInPlace(ws.MlpTxt);
        Linear(vis, img, ws.OutTxt, ws.MlpTxt, w.TxtFc2, nTxt, 4 * d, d);
        vis.ScaleGateAdd(ws.TxtHidden, ws.OutTxt, ws.TxtMod, nTxt, d, gateOffset: 5 * d);
    }

    private void SingleBlockGpu(HunyuanGpuWorkspace ws, HunyuanGpuWeights.SingleBlock w, CoreTensor vec,
        CoreTensor ropeCos, CoreTensor ropeSin, IVisionOpsBackend vis, IImageOpsBackend img)
    {
        int d = _dim, nImg = ws.NumImg, nSeq = ws.NumImg + ws.NumTxt;
        int nh = _numHeads, hd = _headDim;

        Linear(vis, img, ws.SingleMod, vec, w.Mod, 1, d, 3 * d);
        vis.AdaLNModulate(ws.NormedSeq, ws.X, ws.SingleMod, nSeq, d, shiftOffset: 0, scaleOffset: d, isRmsNorm: false, eps: 1e-6f);

        // linear1 -> per token [q | k | v | mlp(4d)] (the CPU path's per-token split).
        Linear(vis, img, ws.Lin1, ws.NormedSeq, w.Linear1, nSeq, d, 7 * d);
        vis.FluxUnpackSingleLin1(ws.Lin1, ws.Q, ws.K, ws.V, ws.MlpSeq, nSeq, d);
        if (w.QNorm is not null) vis.QKNorm(ws.Q, ws.K, w.QNorm, w.KNorm!, nSeq, nh, hd, eps: 1e-6f, startToken: 0);
        vis.Flux2DRoPE(ws.Q, ws.K, ropeCos, ropeSin, startToken: 0, tokenCount: nImg, nh, hd);
        img.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, nSeq, nSeq, nh, hd);

        vis.VisionGeluInPlace(ws.MlpSeq);
        vis.FluxConcatAttnMlp(ws.AttnOut, ws.MlpSeq, ws.Combined, nSeq, d);
        Linear(vis, img, ws.AttnOut, ws.Combined, w.Linear2, nSeq, 5 * d, d);
        vis.ScaleGateAdd(ws.X, ws.AttnOut, ws.SingleMod, nSeq, d, gateOffset: 2 * d);
    }

    private static void Linear(IVisionOpsBackend vis, IImageOpsBackend img, CoreTensor output, CoreTensor input,
        HunyuanGpuWeights.Lin lin, int rows, int inDim, int outDim)
    {
        vis.Sgemm(output, input, lin.W, rows, inDim, outDim);
        if (lin.B is not null) img.AddRowBroadcastInPlace(output, lin.B, rows, outDim);
    }

    private static void LinearRows(IVisionOpsBackend vis, IImageOpsBackend img, CoreTensor output, CoreTensor input,
        HunyuanGpuWeights.Lin lin, int rows, int inDim, int outDim, int rowOffset)
    {
        vis.Sgemm(output, input, lin.W, rows, inDim, outDim, inputRowOffsetElements: rowOffset * inDim);
        if (lin.B is not null) img.AddRowBroadcastInPlace(output, lin.B, rows, outDim);
    }

    private void DisposeGpu()
    {
        _gpuW?.Dispose();
        _gpuWs?.Dispose();
        if (_gpuRope is { } r) { _backend!.Free(r.cos); _backend!.Free(r.sin); }
        _gpuW = null; _gpuWs = null; _gpuRope = null;
    }

    /// <summary>All block weights resident on the device, uploaded once.</summary>
    private sealed class HunyuanGpuWeights : IDisposable
    {
        public sealed record Lin(CoreTensor W, CoreTensor? B);
        public sealed record DoubleBlock(Lin ImgMod, Lin TxtMod, Lin ImgQkv, Lin TxtQkv, Lin ImgProj, Lin TxtProj,
            Lin ImgFc1, Lin ImgFc2, Lin TxtFc1, Lin TxtFc2,
            CoreTensor? ImgQNorm, CoreTensor? ImgKNorm, CoreTensor? TxtQNorm, CoreTensor? TxtKNorm);
        public sealed record SingleBlock(Lin Mod, Lin Linear1, Lin Linear2, CoreTensor? QNorm, CoreTensor? KNorm);

        private readonly IComputeBackend _backend;
        private readonly List<CoreTensor> _all = new();
        public DoubleBlock[] Double { get; }
        public SingleBlock[] Single { get; }

        public HunyuanGpuWeights(HunyuanVideoModel m, IComputeBackend backend)
        {
            _backend = backend;
            int d = m._dim;
            Double = new DoubleBlock[m._depthDouble];
            for (int b = 0; b < Double.Length; b++)
            {
                string p = $"double_blocks.{b}";
                Double[b] = new DoubleBlock(
                    L(m, $"{p}.img_mod.linear", 6 * d, d), L(m, $"{p}.txt_mod.linear", 6 * d, d),
                    L(m, $"{p}.img_attn.qkv", 3 * d, d), L(m, $"{p}.txt_attn.qkv", 3 * d, d),
                    L(m, $"{p}.img_attn.proj", d, d), L(m, $"{p}.txt_attn.proj", d, d),
                    L(m, $"{p}.img_mlp.fc1", 4 * d, d), L(m, $"{p}.img_mlp.fc2", d, 4 * d),
                    L(m, $"{p}.txt_mlp.fc1", 4 * d, d), L(m, $"{p}.txt_mlp.fc2", d, 4 * d),
                    N(m, $"{p}.img_attn.norm.query_norm"), N(m, $"{p}.img_attn.norm.key_norm"),
                    N(m, $"{p}.txt_attn.norm.query_norm"), N(m, $"{p}.txt_attn.norm.key_norm"));
            }
            Single = new SingleBlock[m._depthSingle];
            for (int b = 0; b < Single.Length; b++)
            {
                string p = $"single_blocks.{b}";
                Single[b] = new SingleBlock(
                    L(m, $"{p}.modulation.linear", 3 * d, d), L(m, $"{p}.linear1", 7 * d, d),
                    L(m, $"{p}.linear2", d, 5 * d), N(m, $"{p}.q_norm"), N(m, $"{p}.k_norm"));
            }
        }

        private CoreTensor Keep(CoreTensor t) { _all.Add(t); return t; }

        /// <summary>[outDim, inDim] weight in its checkpoint format + optional fp32 bias.</summary>
        private unsafe Lin L(HunyuanVideoModel m, string name, int outDim, int inDim)
        {
            string wName = m.Resolve($"{name}.weight");
            var shape = TensorShape.D2(outDim, inDim);
            long elems = (long)outDim * inDim;
            CoreTensor w;
            if (m._weights is SafetensorsLoader sl
                && sl.TryGetMappedPointer(wName, out byte* ptr, out long bytes, out string dt)
                && dt == "F8_E4M3" && bytes == elems && _backend.SupportsFp8WeightSgemm)
                w = _backend.UploadFp8(new ReadOnlySpan<byte>(ptr, (int)elems), shape);
            else if (m._weights is SafetensorsLoader sl2
                && sl2.TryGetMappedPointer(wName, out byte* ptr2, out long bytes2, out string dt2)
                && dt2 == "BF16" && bytes2 == elems * 2 && _backend.SupportsBf16WeightSgemm)
                w = _backend.UploadBf16(new ReadOnlySpan<ushort>(ptr2, (int)elems), shape);
            else
            {
                var f = m._weights.ReadF32(wName);
                if (_backend.BestSgemmPrecision == SgemmPrecision.Fp16)
                {
                    var h = new Half[f.Length];
                    System.Numerics.Tensors.TensorPrimitives.ConvertToHalf(f, h);
                    w = _backend.UploadHalf(h, shape);
                }
                else w = _backend.Upload(f, shape, exact: true);
            }
            Keep(w);

            string bName = m.Resolve($"{name}.bias");
            CoreTensor? bias = m._weights.Contains(bName)
                ? Keep(_backend.Upload(m._weights.ReadF32(bName), TensorShape.D1(outDim), exact: true))
                : null;
            return new Lin(w, bias);
        }

        /// <summary>Per-head RMSNorm scale ([headDim]) under either naming, or null.</summary>
        private CoreTensor? N(HunyuanVideoModel m, string name)
        {
            var v = m.TryGetWeight($"{name}.weight") ?? m.TryGetWeight($"{name}.scale");
            return v is null ? null : Keep(_backend.Upload(v, TensorShape.D1(v.Length), exact: true));
        }

        public void Dispose()
        {
            foreach (var t in _all) _backend.Free(t);
            _all.Clear();
        }
    }

    private sealed class HunyuanGpuWorkspace : IDisposable
    {
        private readonly IComputeBackend _b;
        public int NumImg { get; }
        public int NumTxt { get; }
        public CoreTensor ImgHidden, TxtHidden;
        public readonly CoreTensor NormedImg, NormedTxt, ImgMod, TxtMod, QkvImg, QkvTxt, Q, K, V, AttnOut,
            OutImg, OutTxt, MlpImg, MlpTxt, X, NormedSeq, SingleMod, Lin1, MlpSeq, Combined;

        public HunyuanGpuWorkspace(IComputeBackend b, int nImg, int nTxt, int d)
        {
            _b = b; NumImg = nImg; NumTxt = nTxt;
            int nSeq = nImg + nTxt;
            CoreTensor A(long r, long c) => b.Allocate(TensorShape.D2(r, c), exact: true);
            ImgHidden = A(nImg, d); TxtHidden = A(nTxt, d);
            NormedImg = A(nImg, d); NormedTxt = A(nTxt, d);
            ImgMod = A(1, 6 * d); TxtMod = A(1, 6 * d);
            QkvImg = A(nImg, 3 * d); QkvTxt = A(nTxt, 3 * d);
            Q = A(nSeq, d); K = A(nSeq, d); V = A(nSeq, d); AttnOut = A(nSeq, d);
            OutImg = A(nImg, d); OutTxt = A(nTxt, d);
            MlpImg = A(nImg, 4 * d); MlpTxt = A(nTxt, 4 * d);
            X = A(nSeq, d); NormedSeq = A(nSeq, d); SingleMod = A(1, 3 * d);
            Lin1 = A(nSeq, 7 * d); MlpSeq = A(nSeq, 4 * d); Combined = A(nSeq, 5 * d);
        }

        /// <summary>Replace the residual streams with this step's CPU-computed tokens.</summary>
        public void SetStreams(IComputeBackend b, float[] img, float[] txt)
        {
            b.Free(ImgHidden); b.Free(TxtHidden);
            ImgHidden = b.Upload(img, TensorShape.D2(NumImg, img.Length / NumImg), exact: true);
            TxtHidden = b.Upload(txt, TensorShape.D2(NumTxt, txt.Length / NumTxt), exact: true);
        }

        public void Dispose()
        {
            foreach (var t in new[] { ImgHidden, TxtHidden, NormedImg, NormedTxt, ImgMod, TxtMod, QkvImg, QkvTxt,
                         Q, K, V, AttnOut, OutImg, OutTxt, MlpImg, MlpTxt, X, NormedSeq, SingleMod, Lin1, MlpSeq, Combined })
                _b.Free(t);
        }
    }
}
