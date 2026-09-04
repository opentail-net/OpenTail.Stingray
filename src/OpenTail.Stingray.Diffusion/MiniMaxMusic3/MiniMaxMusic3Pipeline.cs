using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 single-chunk end-to-end synthesis: chains the already golden-verified
/// components (<see cref="MiniMaxMusic3ConditionEncoder"/>, <see cref="MiniMaxMusic3Transformer"/>
/// via <see cref="MiniMaxMusic3FlowScheduler"/>, <see cref="MiniMaxMusic3Vocoder"/>) from a
/// <see cref="Music3Representation"/> (produced by <see cref="MiniMaxMusic3AutoregressiveGenerator"/>)
/// to real stereo PCM. V1 scope: single chunk only (song short enough to fit the real
/// `_CHUNK_FRAMES=200`-frame, ~8s window) -- no multi-window overlap-blend/stitching yet. See
/// docs/066-minimax-music3-future-plan.md.
/// </summary>
public static class MiniMaxMusic3Pipeline
{
    public static float[] Synthesize(
        MiniMaxMusic3ConditionEncoderWeights conditionWeights,
        MiniMaxMusic3TransformerWeights transformerWeights,
        MiniMaxMusic3VocoderWeights vocoderWeights,
        Music3Representation representation,
        int numFlowSteps,
        int? seed,
        IComputeBackend? backend = null)
    {
        var conditionInput = ToConditionLayers(representation);
        var condition = MiniMaxMusic3ConditionEncoder.Forward(conditionWeights, conditionInput);

        var latent = MiniMaxMusic3FlowScheduler.Denoise(transformerWeights, condition, numFlowSteps, seed, backend);

        int latentLen = latent.Length;
        int inChannels = MiniMaxMusic3Config.TransformerInChannels;
        var channelMajor = new float[inChannels * latentLen];
        for (int t = 0; t < latentLen; t++)
            for (int c = 0; c < inChannels; c++)
                channelMajor[c * latentLen + t] = latent[t][c];

        return MiniMaxMusic3Vocoder.Decode(vocoderWeights, channelMajor, latentLen);
    }

    /// <summary>Real condition-encoder input shape: `[frame][8 layers][condHiddenDim]` -- layer 0 is
    /// the Global hidden state, layers 1..7 are the 7 residual-codebook depth-decoder hiddens in
    /// real c1..c7 order (already concatenated this way in <see cref="Music3Representation.LocalHiddenStates"/>).</summary>
    private static float[][][] ToConditionLayers(Music3Representation representation)
    {
        int hidden = MiniMaxMusic3Config.LanguageModelHiddenSize;
        int numLayers = MiniMaxMusic3Config.ConditionEncoderNumLayers; // 8
        int frameCount = representation.FrameCount;

        var result = new float[frameCount][][];
        for (int f = 0; f < frameCount; f++)
        {
            var layers = new float[numLayers][];
            layers[0] = representation.GlobalHiddenStates[f];
            for (int l = 1; l < numLayers; l++)
            {
                var row = new float[hidden];
                Array.Copy(representation.LocalHiddenStates[f], (l - 1) * hidden, row, 0, hidden);
                layers[l] = row;
            }
            result[f] = layers;
        }
        return result;
    }
}
