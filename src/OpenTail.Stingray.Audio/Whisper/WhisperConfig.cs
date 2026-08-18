namespace OpenTail.Stingray.Audio.Whisper;

/// <summary>
/// Architectural hyper-parameters for OpenAI Whisper models.
/// </summary>
public sealed record WhisperConfig
{
    public int VocabSize { get; init; } = 51864;
    public int AudioCtx { get; init; } = 1500;
    public int AudioState { get; init; } = 384;      // d_model for audio encoder
    public int AudioHead { get; init; } = 6;
    public int AudioLayer { get; init; } = 4;
    public int TextCtx { get; init; } = 448;
    public int TextState { get; init; } = 384;        // d_model for text decoder
    public int TextHead { get; init; } = 6;
    public int TextLayer { get; init; } = 4;
    public int NumMels { get; init; } = 80;
    public float LayerNormEps { get; init; } = 1e-5f;

    public static WhisperConfig Tiny => new()
    {
        VocabSize = 51864,
        AudioState = 384,
        AudioHead = 6,
        AudioLayer = 4,
        TextState = 384,
        TextHead = 6,
        TextLayer = 4,
        NumMels = 80
    };

    public static WhisperConfig Base => new()
    {
        VocabSize = 51864,
        AudioState = 512,
        AudioHead = 8,
        AudioLayer = 6,
        TextState = 512,
        TextHead = 8,
        TextLayer = 6,
        NumMels = 80
    };

    public static WhisperConfig Small => new()
    {
        VocabSize = 51864,
        AudioState = 768,
        AudioHead = 12,
        AudioLayer = 12,
        TextState = 768,
        TextHead = 12,
        TextLayer = 12,
        NumMels = 80
    };

    public static WhisperConfig Medium => new()
    {
        VocabSize = 51864,
        AudioState = 1024,
        AudioHead = 16,
        AudioLayer = 24,
        TextState = 1024,
        TextHead = 16,
        TextLayer = 24,
        NumMels = 80
    };

    public static WhisperConfig LargeV3 => new()
    {
        VocabSize = 51865,
        AudioState = 1280,
        AudioHead = 20,
        AudioLayer = 32,
        TextState = 1280,
        TextHead = 20,
        TextLayer = 32,
        NumMels = 128
    };

    public static WhisperConfig LargeV3Turbo => new()
    {
        VocabSize = 51865,
        AudioState = 1280,
        AudioHead = 20,
        AudioLayer = 32,
        TextState = 1280,
        TextHead = 20,
        TextLayer = 4,
        NumMels = 128
    };
}
