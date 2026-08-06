using System.Text.Json.Serialization;

namespace OpenTail.Stingray.Core;

/// <summary>The persisted container format from which a model's weights are loaded.</summary>
/// <remarks>
/// <c>Gguf</c> remains zero so execution plans and session envelopes written before this discriminator
/// existed retain their historical, GGUF-only interpretation when decoded by newer builds.
/// </remarks>
[JsonConverter(typeof(JsonStringEnumConverter<ModelFormat>))]
public enum ModelFormat : byte
{
    Gguf = 0,
    SafeTensors = 1,
}
