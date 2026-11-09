namespace OpenTail.Stingray.Engine;

/// <summary>
/// One video for the qwen4exp pass: decoded frames already in the projector's input form (planar CHW, one array per frame,
/// each <see cref="Side"/> x <see cref="Side"/>), and the frame rate the frames were sampled at. The host decodes; the engine
/// only splices, so no decoder runs in the engine.
/// </summary>
public sealed record QwenVideoInput(IReadOnlyList<float[]> FramesChw, int Side, float Fps);
