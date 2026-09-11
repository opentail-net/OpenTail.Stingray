namespace OpenTail.Stingray.Audio.ParaformerOnnx;

/// <summary>
/// Real Paraformer greedy decode, ported verbatim from
/// `examples/sherpa-onnx/sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.cc`. This is
/// NOT CTC-collapse (no blank/repeat-collapse logic, unlike e.g. SenseVoice's CTC decode) --
/// Paraformer's decoder output is already a fixed-length non-autoregressive sequence (the CIF
/// predictor inside the fused ONNX graph already determined the output length), so decoding is
/// simply: per output position, take the argmax over the vocab; stop (do not emit) at the first
/// position whose argmax is `eos_id`. `token_num`/`us_cif_peak` (used only for optional timestamp
/// extraction in the real reference) are intentionally not read here -- this checkpoint exports no
/// `us_cif_peak` output at all (confirmed via real ONNX output-name inspection).
/// </summary>
public static class ParaformerOnnxDecoder
{
    public static List<int> GreedyDecode(float[][] logProbs, int eosId)
    {
        var tokens = new List<int>(logProbs.Length);
        foreach (var row in logProbs)
        {
            int best = 0;
            float bestValue = row[0];
            for (int c = 1; c < row.Length; c++)
            {
                if (row[c] > bestValue) { bestValue = row[c]; best = c; }
            }
            if (best == eosId) break;
            tokens.Add(best);
        }
        return tokens;
    }
}
