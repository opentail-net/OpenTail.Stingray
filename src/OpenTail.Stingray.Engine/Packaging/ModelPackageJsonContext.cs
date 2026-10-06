using System.Text.Json.Serialization;

namespace OpenTail.Stingray.Engine.Packaging;

[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.Unspecified)]
[JsonSerializable(typeof(StingraySidecarMetadata))]
[JsonSerializable(typeof(StingraySidecarAdmission))]
[JsonSerializable(typeof(StingraySidecarCapabilities))]
[JsonSerializable(typeof(OllamaManifest))]
[JsonSerializable(typeof(OllamaLayer))]
[JsonSerializable(typeof(ModelPackageIdentity))]
[JsonSerializable(typeof(ModelPackageComponentIdentity))]
[JsonSerializable(typeof(ModelPackageComponent))]
internal sealed partial class ModelPackageJsonContext : JsonSerializerContext
{
}
