using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine.Verification;

/// <summary>
/// Builds the CPU forward pass a golden is checked against, the same way the parity receipts did: the dense <see cref="ForwardPass"/>
/// for ordinary architectures, the RWKV pass for the RWKV family. Owns the compute backend; dispose it after the run. Each call to
/// <see cref="Create"/> returns a fresh pass (the runner needs clean state per check) over the already-open model.
/// </summary>
public sealed class GoldenForwardPassSource : IDisposable
{
    private readonly CpuBackend _backend = new();

    public GoldenForwardPassSource(GgufModel model, int maxContextLength = 2048)
    {
        string arch = model.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a) ?? "" : "";
        var family = ArchitectureRegistry.Find(arch)?.Capabilities.Family ?? ForwardPassFamily.Dense;
        if (family == ForwardPassFamily.Rwkv)
        {
            Create = () => RwkvForwardPassBase.Create(model);
        }
        else
        {
            var hp = ArchitectureModelResolver.ResolveHyperparams(model);
            Create = () => new ForwardPass(model, _backend, hp, maxContextLength);
        }
    }

    public Func<IForwardPass> Create { get; }

    public void Dispose() => _backend.Dispose();
}
