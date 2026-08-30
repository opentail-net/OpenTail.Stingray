
namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Full Fish Speech S2 Pro text-to-speech pipeline: text -&gt; <see cref="FishSpeechPipeline"/>'s
/// slow-AR semantic-token generation (which internally also runs the real fast-AR codebook
/// expansion per frame, see <see cref="FishSpeechPipeline.GenerateFrames"/>) -&gt; real
/// <see cref="FishSpeechCodec"/> decode -&gt; mono float32 PCM.
///
/// <para>Analogous to <c>OrpheusPipeline.Synthesize</c>: wires together already golden-verified
/// components (slow-AR, fast-AR, codec -- each independently proven numerically correct against
/// real oracles, see docs/audio-review-progress.md's Fish Speech section) into one callable
/// end-to-end path. No new model math here -- purely plumbing.</para>
/// </summary>
public sealed class FishSpeechFullPipeline : ITextToSpeechPipeline
{
    public string Architecture => "FishSpeech";
    public int SampleRate => 44100;
    public int DefaultSampleRate => 44100;

    private readonly FishSpeechPipeline _talker;
    private readonly FishSpeechCodecWeights _codecWeights;

    public static FishSpeechFullPipeline Load(string modelPath, string? tokDir = null, string? codecGgufPath = null)
    {
        tokDir ??= ResolveTokenizerDir(modelPath);
        codecGgufPath ??= modelPath; // s2-pro checkpoint contains the embedded codec weights
        return new FishSpeechFullPipeline(modelPath, tokDir, codecGgufPath);
    }

    private static string ResolveTokenizerDir(string modelPath)
    {
        string[] candidates = ["examples/s2.cpp", "models/s2.cpp", "models"];
        foreach (var c in candidates)
        {
            if (Directory.Exists(c) && (File.Exists(Path.Combine(c, "tokenizer.json")) || File.Exists(Path.Combine(c, "vocab.json"))))
                return c;
        }
        return "examples/s2.cpp";
    }

    public FishSpeechFullPipeline(string talkerGgufPath, string tokenizerDir, string codecGgufPath, int numLayers = 36, int ctxSize = 2048)
    {
        _talker = new FishSpeechPipeline(talkerGgufPath, tokenizerDir, numLayers, ctxSize);
        _codecWeights = new FishSpeechCodecWeights(codecGgufPath);
    }

    public AudioGenerationResult Generate(AudioGenerationRequest request)
    {
        var pcm = Synthesize(request.Text);
        var result = new AudioGenerationResult(pcm, DefaultSampleRate);
        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            result.SaveWav(request.OutputPath);
        }
        return result;
    }

    public const int SamplesPerFrame = 2048;

    public IAsyncEnumerable<float[]> GenerateStreamAsync(AudioGenerationRequest request, System.Threading.CancellationToken ct = default)
        => GenerateStreamAsync(request.Text, chunkFrames: 1, contextFrames: 8, maxTokens: 200, seed: 42, ct: ct);

    public async IAsyncEnumerable<float[]> GenerateStreamAsync(string text, int chunkFrames = 1, int contextFrames = 8, int maxTokens = 200, int? seed = null, [System.Runtime.CompilerServices.EnumeratorCancellation] System.Threading.CancellationToken ct = default)
    {
        var allSemantic = new List<int>();
        var allCodebooks = new List<int[]>();
        var pendingSemantic = new List<int>();
        var pendingCodebooks = new List<int[]>();

        foreach (var (semCode, codebookValues) in _talker.GenerateFramesStream(text, maxTokens, seed))
        {
            ct.ThrowIfCancellationRequested();
            allSemantic.Add(semCode);
            allCodebooks.Add(codebookValues);
            pendingSemantic.Add(semCode);
            pendingCodebooks.Add(codebookValues);

            if (pendingSemantic.Count >= chunkFrames)
            {
                var chunkPcm = DecodeChunk(allSemantic, allCodebooks, pendingSemantic.Count, contextFrames);
                pendingSemantic.Clear();
                pendingCodebooks.Clear();
                if (chunkPcm.Length > 0)
                {
                    yield return chunkPcm;
                }
            }
        }

        if (pendingSemantic.Count > 0)
        {
            var chunkPcm = DecodeChunk(allSemantic, allCodebooks, pendingSemantic.Count, contextFrames);
            if (chunkPcm.Length > 0)
            {
                yield return chunkPcm;
            }
        }
        await Task.CompletedTask;
    }

    private float[] DecodeChunk(List<int> allSemantic, List<int[]> allCodebooks, int newFramesCount, int contextFrames)
    {
        int totalFrames = allSemantic.Count;
        int startFrame = Math.Max(0, totalFrames - newFramesCount - contextFrames);
        int windowFrames = totalFrames - startFrame;
        int priorOverlapFrames = windowFrames - newFramesCount;

        var semanticSlice = new int[windowFrames];
        int numResidual = allCodebooks[0].Length - 1;
        var residualSlice = new int[numResidual][];
        for (int cb = 0; cb < numResidual; cb++)
            residualSlice[cb] = new int[windowFrames];

        for (int i = 0; i < windowFrames; i++)
        {
            int globalFrame = startFrame + i;
            semanticSlice[i] = allSemantic[globalFrame];
            for (int cb = 0; cb < numResidual; cb++)
                residualSlice[cb][i] = allCodebooks[globalFrame][cb + 1];
        }

        var decodedWindow = FishSpeechCodec.Decode(_codecWeights, semanticSlice, residualSlice);
        int skipSamples = priorOverlapFrames * SamplesPerFrame;
        int takeSamples = newFramesCount * SamplesPerFrame;

        if (skipSamples + takeSamples <= decodedWindow.Length)
        {
            var chunk = new float[takeSamples];
            Array.Copy(decodedWindow, skipSamples, chunk, 0, takeSamples);
            return chunk;
        }
        else if (skipSamples < decodedWindow.Length)
        {
            int available = decodedWindow.Length - skipSamples;
            var chunk = new float[available];
            Array.Copy(decodedWindow, skipSamples, chunk, 0, available);
            return chunk;
        }

        return decodedWindow;
    }

    /// <summary>Full pipeline: text -&gt; mono float32 PCM (44.1kHz, matching the real codec's native rate).</summary>
    public float[] Synthesize(string text, int maxTokens = 200, int? seed = null)
    {
        var (semanticTokens, codebooksPerFrame) = _talker.GenerateFrames(text, maxTokens, seed);
        if (semanticTokens.Count == 0) return [];

        int t = semanticTokens.Count;
        var semanticCodes = semanticTokens.ToArray();

        // codebooksPerFrame[frame] = [semantic, residual_0, .., residual_8] (NumCodebooks=10 total,
        // index 0 duplicates the already-known semantic code -- see FishSpeechPipeline.GenerateFrames).
        int numResidual = codebooksPerFrame[0].Length - 1;
        var residualCodes = new int[numResidual][];
        for (int cb = 0; cb < numResidual; cb++)
        {
            residualCodes[cb] = new int[t];
            for (int ti = 0; ti < t; ti++)
                residualCodes[cb][ti] = codebooksPerFrame[ti][cb + 1];
        }

        var pcm = FishSpeechCodec.Decode(_codecWeights, semanticCodes, residualCodes);

        // Peak normalize to 0.85 full scale
        float peak = 0f;
        for (int i = 0; i < pcm.Length; i++)
        {
            float a = MathF.Abs(pcm[i]);
            if (a > peak) peak = a;
        }
        if (peak > 1e-4f && peak < 0.8f)
        {
            float gain = 0.85f / peak;
            for (int i = 0; i < pcm.Length; i++) pcm[i] *= gain;
        }

        return pcm;
    }

    public void Dispose()
    {
        _talker.Dispose();
        _codecWeights.Dispose();
    }
}
