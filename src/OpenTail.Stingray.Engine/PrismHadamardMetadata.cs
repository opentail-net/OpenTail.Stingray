using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// The PRISM rotation contract carried by Bonsai2 GGUFs (<c>prism.hadamard.*</c>). Validation follows
/// TensorSharp's <c>BonsaiHadamardMetadata</c> (BSD-3): anything other than version 1,
/// normalized-sylvester-walsh-hadamard, input-last-dimension and explicit signs is refused, never guessed.
/// "Ported, not verified" (CLAUDE.md rule 14): loading also needs <c>STINGRAY_EXPERIMENTAL_PRISM=1</c>
/// (see <see cref="ModelCompatibility"/>).
/// </summary>
internal sealed class PrismHadamardMetadata
{
    private const string Prefix = "prism.hadamard.";

    public required int BlockSize { get; init; }
    public required bool GdnVGrouped { get; init; }
    public required IReadOnlySet<string> ForwardNames { get; init; }
    public required IReadOnlySet<string> InverseNames { get; init; }
    public required IReadOnlyDictionary<int, float[]> SignsByWidth { get; init; }

    public static bool IsPresent(GgufModel model) =>
        model.Metadata.Keys.Any(k => k.StartsWith(Prefix, StringComparison.Ordinal))
        || model.Tensors.Any(t => BonsaiQuant.IsBonsaiType(t.DType));

    /// <summary>Null for ordinary GGUFs; throws for an incomplete or unknown transform.</summary>
    public static PrismHadamardMetadata? Read(GgufModel model)
    {
        bool present = model.Metadata.Keys.Any(k => k.StartsWith(Prefix, StringComparison.Ordinal));
        bool customQuant = model.Tensors.Any(t => BonsaiQuant.IsBonsaiType(t.DType));
        if (!present)
        {
            if (customQuant)
                throw new InvalidDataException("PQ2_0/PTQ1_0 weights require prism.hadamard metadata.");
            return null;
        }

        string Str(string key) => model.Metadata.TryGetValue(Prefix + key, out var v) ? Convert.ToString(v) ?? "" : "";
        long Num(string key) => model.Metadata.TryGetValue(Prefix + key, out var v) ? Convert.ToInt64(v) : -1;
        if (Num("version") != 1 || Str("transform") != "normalized-sylvester-walsh-hadamard"
            || Str("axis") != "input-last-dimension" || Str("sign_mode") != "explicit")
            throw new NotSupportedException("Unsupported PRISM Hadamard version, transform, axis or sign mode.");

        int blockSize = checked((int)Num("block_size"));
        if (blockSize is not (64 or 128 or 256 or 512 or 1024))
            throw new InvalidDataException("PRISM Hadamard block size must be 64, 128, 256, 512 or 1024.");

        object[] widths = Arr("sign_widths");
        object[] values = Arr("sign_values");
        var signs = new Dictionary<int, float[]>();
        int offset = 0;
        foreach (object wObj in widths)
        {
            int width = Convert.ToInt32(wObj);
            if (width <= 0 || width % blockSize != 0 || signs.ContainsKey(width) || width > values.Length - offset)
                throw new InvalidDataException("PRISM sign widths must be unique, block-aligned and match sign_values.");
            var row = new float[width];
            for (int i = 0; i < width; i++)
            {
                float s = Convert.ToSingle(values[offset++]);
                if (s is not (1f or -1f))
                    throw new InvalidDataException("PRISM Hadamard signs must be exactly -1 or +1.");
                row[i] = s;
            }
            signs.Add(width, row);
        }
        if (offset != values.Length || signs.Count == 0)
            throw new InvalidDataException("PRISM sign_values length does not match sign_widths.");

        var forward = Names("weight_names");
        var inverse = Names("inverse_weight_names");
        foreach (string name in inverse)
        {
            if (forward.Contains(name))
                throw new InvalidDataException($"PRISM tensor '{name}' has both forward and inverse rotations.");
            if (name != "token_embd.weight")
                throw new NotSupportedException($"Unsupported PRISM inverse rotation target '{name}'.");
        }
        foreach (var t in model.Tensors)
            if (BonsaiQuant.IsBonsaiType(t.DType) && !forward.Contains(t.Name) && !inverse.Contains(t.Name))
                throw new InvalidDataException($"PRISM tensor '{t.Name}' ({t.DType}) has no Hadamard rotation metadata.");

        bool grouped = model.Metadata.TryGetValue(Prefix + "gdn_v_grouped", out var g) && Convert.ToBoolean(g);
        return new PrismHadamardMetadata
        {
            BlockSize = blockSize, GdnVGrouped = grouped,
            ForwardNames = forward, InverseNames = inverse, SignsByWidth = signs,
        };

        object[] Arr(string key) =>
            model.Metadata.TryGetValue(Prefix + key, out var v) && v is object[] a
                ? a : throw new InvalidDataException($"Missing PRISM Hadamard {key}.");

        HashSet<string> Names(string key)
        {
            var set = new HashSet<string>(StringComparer.Ordinal);
            foreach (object o in Arr(key))
            {
                string name = Convert.ToString(o) ?? "";
                var info = model.FindTensor(name);
                if (!set.Add(name) || info is null || info.Value.NDimensions != 2
                    || !signs.ContainsKey((int)info.Value.Dimensions[0]))
                    throw new InvalidDataException($"Invalid or duplicate PRISM Hadamard tensor '{name}'.");
            }
            return set;
        }
    }
}
