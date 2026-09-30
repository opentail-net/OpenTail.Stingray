using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// RVC compatibility wrapper around <see cref="AudioCppPackedTensorSource"/>.
/// </summary>
public sealed class RvcPackedTensorSource : AudioCppPackedTensorSource
{
    public RvcPackedTensorSource(GgufModel model) : base(model)
    {
    }
}
