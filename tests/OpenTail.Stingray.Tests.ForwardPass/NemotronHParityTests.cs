
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// NVIDIA Nemotron-H (nemotron_h: every layer is exactly one of Mamba-2 / NoPE attention / ReLU² MLP) against
/// llama-server (vendored tools/llama.cpp, same GGUF: nemotron-nano-12b-v2-vl Q2_K text decoder, raw completion,
/// temperature 0, ignore_eos, n_probs 2; captured 2026-09-27). Q2_K moves individual log-probs by a few tenths of
/// a nat, so rather than pinning a free-running greedy sequence (which forks at the first close pair), the
/// reference continuation is teacher-forced and our argmax must equal llama.cpp's token wherever llama.cpp's
/// top-1 led its top-2 by more than 1.5 nats. Admission evidence is second-half perplexity (ModelCompatibility.cs).
/// </summary>
public sealed class NemotronHParityTests : HeavyTestBase
{
    private const string ModelFile = "nemotron-nano-12b-v2-vl-Q2_K.gguf";

    [Fact]
    public void NemotronH_ShortPrompt_ConfidentTokensMatchLlamaServer()
    {
        // ' Paris."  \n- "The capital of France is Paris."  \n-'
        AssertTeacherForced("The capital of France is",
            [6993, 2613, 1256, 1010, 1045, 1429, 1784, 8961, 1307, 5498, 1395, 6993, 2613, 1256, 1010, 1045],
            [2.474, 0.217, 0.018, 3.063, 0.122, 0.03, 1.873, 2.221, 4.313, 2.69, 5.408, 2.043, 1.569, 3.087, 1.975, 3.583]);
    }

    [Fact]
    public void NemotronH_LongerPrompt_ConfidentTokensMatchLlamaServer()
    {
        // ' the second explained the special theory of relativity. The third paper explained the concept of'
        AssertTeacherForced(
            "In 1905, Albert Einstein published four papers that changed physics. The first explained the photoelectric effect, and",
            [1278, 2667, 13471, 1278, 6061, 9191, 1307, 115847, 1046, 1531, 5888, 7873, 13471, 1278, 7401, 1307],
            [2.763, 1.611, 0.134, 0.279, 0.496, 3.561, 8.038, 5.213, 2.415, 2.031, 2.459, 0.605, 0.458, 1.936, 0.019, 4.517]);
    }

    private static void AssertTeacherForced(string prompt, int[] reference, double[] llamaMargins) =>
        TeacherForcedParity.Assert(ModelFile, "nemotron_h", TeacherForcedParity.Tokenize(ModelFile, prompt), reference, llamaMargins);
}
