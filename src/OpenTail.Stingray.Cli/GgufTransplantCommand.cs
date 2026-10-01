using System.ComponentModel;
using System.Text;
using System.Text.RegularExpressions;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Builds a mixed-precision GGUF by copying a BASE checkpoint and replacing selected tensors with the
/// same-named tensors from a DONOR checkpoint (typically a higher-precision quantization of the same model).
/// Used where <c>llama-quantize</c> cannot be (it rejects architectures it does not know, e.g. fish-speech).
/// Usage:
///   opentail-llm-cli gguf-transplant --base s2-pro-q4_k_m.gguf --donor s2-pro-q8_0.gguf
///       --tensors "^fast_layers\.\d+\.(attention\.wo|feed_forward\.w3)\.weight$" -o s2-pro-q4_k_m-fastar-q8.gguf
///
/// Byte-level strategy (never re-encodes the KV metadata): the header and KV section of the base file are copied
/// verbatim (its end is derived from the tensor-info end offset minus the size of the tensor-info entries), the tensor
/// infos are re-emitted in the base file's order with recomputed offsets and the donor's dtype for replaced tensors,
/// then tensor data is written aligned. Replaced tensors must have identical dimensions in the donor. Both inputs are
/// opened read-only. The output is re-opened and every tensor is compared byte-for-byte with its source.
/// </summary>
public sealed unsafe class GgufTransplantCommand : Command<GgufTransplantCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("--base")]
        [Description("GGUF whose header, metadata and (non-replaced) tensors are kept")]
        public string? BasePath { get; init; }

        [CommandOption("--donor")]
        [Description("GGUF supplying the replacement tensors (same tensor names and dimensions as the base)")]
        public string? DonorPath { get; init; }

        [CommandOption("--tensors")]
        [Description("Regex matched against base tensor names; matching tensors are taken from the donor")]
        public string? TensorPattern { get; init; }

        [CommandOption("-o|--out")]
        [Description("Output GGUF path (must not be the base or donor; must not exist unless --force)")]
        public string? OutPath { get; init; }

        [CommandOption("--force")]
        [Description("Overwrite the output if it exists")]
        public bool Force { get; init; }

        [CommandOption("--dry-run")]
        [Description("Print the replacement plan and the resulting size, write nothing")]
        public bool DryRun { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (settings.BasePath is null || settings.DonorPath is null || settings.TensorPattern is null
            || (settings.OutPath is null && !settings.DryRun))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] --base, --donor, --tensors and -o/--out are required (-o optional with --dry-run).");
            return ExitCodes.Usage;
        }
        if (!File.Exists(settings.BasePath) || !File.Exists(settings.DonorPath))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] base or donor file not found.");
            return 1;
        }
        string? outPath = settings.OutPath is null ? null : Path.GetFullPath(settings.OutPath);
        if (outPath is not null)
        {
            if (string.Equals(outPath, Path.GetFullPath(settings.BasePath), StringComparison.OrdinalIgnoreCase)
                || string.Equals(outPath, Path.GetFullPath(settings.DonorPath), StringComparison.OrdinalIgnoreCase))
            {
                AnsiConsole.ErrorLine("[red]Error:[/] the output must not be the base or donor file.");
                return 1;
            }
            if (File.Exists(outPath) && !settings.Force && !settings.DryRun)
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(outPath)} exists; pass --force to overwrite.");
                return 1;
            }
        }

        Regex pattern;
        try { pattern = new Regex(settings.TensorPattern, RegexOptions.CultureInvariant); }
        catch (ArgumentException ex)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] invalid --tensors regex: {Markup.Escape(ex.Message)}");
            return ExitCodes.Usage;
        }

        using var baseModel = GgufModel.Open(settings.BasePath);
        using var donor = GgufModel.Open(settings.DonorPath);
        var baseTensors = baseModel.Tensors;
        if (baseTensors.Any(t => t.ShardIndex != 0))
        {
            AnsiConsole.ErrorLine("[red]Error:[/] sharded base checkpoints are not supported.");
            return 1;
        }

        // Plan: which tensors come from the donor, and what each output entry looks like.
        var donorByName = donor.Tensors.GroupBy(t => t.Name).ToDictionary(g => g.Key, g => g.First());
        var plan = new List<(GgufTensorInfo Info, bool FromDonor, GgufTensorInfo Source)>(baseTensors.Count);
        long deltaBytes = 0;
        foreach (var t in baseTensors)
        {
            if (!pattern.IsMatch(t.Name)) { plan.Add((t, false, t)); continue; }
            if (!donorByName.TryGetValue(t.Name, out var d))
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] donor has no tensor named {Markup.Escape(t.Name)}.");
                return 1;
            }
            if (!d.Dimensions.Take(d.NDimensions).SequenceEqual(t.Dimensions.Take(t.NDimensions)))
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] dimension mismatch for {Markup.Escape(t.Name)}.");
                return 1;
            }
            plan.Add((new GgufTensorInfo(t.Name, d.NDimensions, d.Dimensions, d.DType, 0), true, d));
            deltaBytes += d.ByteSize - t.ByteSize;
        }
        int replaced = plan.Count(p => p.FromDonor);
        if (replaced == 0)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] --tensors matched no base tensors.");
            return 1;
        }

        var summary = plan.Where(p => p.FromDonor)
            .GroupBy(p => (Before: baseModel.FindTensor(p.Info.Name)!.Value.DType, After: p.Info.DType))
            .Select(g => $"{g.Count()} x {g.Key.Before} -> {g.Key.After}");
        AnsiConsole.MarkupLine($"[bold]{replaced}[/] of {plan.Count} tensors replaced: {Markup.Escape(string.Join(", ", summary))}; " +
            $"size change [cyan]{deltaBytes / 1048576.0:+0.0;-0.0} MiB[/].");
        if (settings.DryRun) return 0;

        // Header + KV section, verbatim from the base file.
        long infoBytes = 0;
        foreach (var t in baseTensors)
            infoBytes += 8 + Encoding.UTF8.GetByteCount(t.Name) + 4 + 8L * t.NDimensions + 4 + 8;
        long kvEnd = baseModel.Shard0TensorInfoEndOffset - infoBytes;
        if (kvEnd < 24 || kvEnd > baseModel.Shard0TensorInfoEndOffset)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] could not derive the end of the KV section from the base file.");
            return 1;
        }
        byte[] head = new byte[kvEnd];
        using (var fs = File.OpenRead(settings.BasePath)) fs.ReadExactly(head, 0, head.Length);
        if (Encoding.ASCII.GetString(head, 0, 4) != "GGUF" || BitConverter.ToUInt64(head, 8) != (ulong)baseTensors.Count)
        {
            AnsiConsole.ErrorLine("[red]Error:[/] base header sanity check failed (magic / tensor count).");
            return 1;
        }

        int alignment = baseModel.Metadata.TryGetValue("general.alignment", out var alignObj) ? Convert.ToInt32(alignObj) : 32;
        var offsets = new long[plan.Count];
        long cursor = 0;
        for (int i = 0; i < plan.Count; i++)
        {
            offsets[i] = AlignUp(cursor, alignment);
            cursor = offsets[i] + plan[i].Source.ByteSize;
        }

        using (var outFs = new FileStream(outPath!, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
        {
            outFs.Write(head, 0, head.Length);
            for (int i = 0; i < plan.Count; i++)
            {
                var info = plan[i].Info;
                byte[] name = Encoding.UTF8.GetBytes(info.Name);
                WriteU64(outFs, (ulong)name.Length);
                outFs.Write(name, 0, name.Length);
                WriteU32(outFs, (uint)info.NDimensions);
                for (int d = 0; d < info.NDimensions; d++) WriteU64(outFs, (ulong)info.Dimensions[d]);
                WriteU32(outFs, (uint)info.DType);
                WriteU64(outFs, (ulong)offsets[i]);
            }
            PadTo(outFs, AlignUp(outFs.Position, alignment));
            long dataStart = outFs.Position;

            for (int i = 0; i < plan.Count; i++)
            {
                cancellation.ThrowIfCancellationRequested();
                PadTo(outFs, dataStart + offsets[i]);
                var (_, fromDonor, source) = plan[i];
                byte* p = (fromDonor ? donor : baseModel).GetTensorDataPtr(source);
                WriteBytes(outFs, p, source.ByteSize);
            }
        }

        // Verify: the output parses and every tensor is byte-identical to its source.
        using var result = GgufModel.Open(outPath!);
        if (result.Tensors.Count != plan.Count)
        {
            AnsiConsole.ErrorLine($"[red]Verification failed:[/] output has {result.Tensors.Count} tensors, expected {plan.Count}.");
            return 1;
        }
        for (int i = 0; i < plan.Count; i++)
        {
            var (info, fromDonor, source) = plan[i];
            var written = result.FindTensor(info.Name);
            if (written is null || written.Value.DType != info.DType || written.Value.ByteSize != source.ByteSize
                || !BytesEqual(result.GetTensorDataPtr(written.Value), (fromDonor ? donor : baseModel).GetTensorDataPtr(source), source.ByteSize))
            {
                AnsiConsole.ErrorLine($"[red]Verification failed:[/] tensor {Markup.Escape(info.Name)} differs from its source.");
                return 1;
            }
        }
        AnsiConsole.MarkupLine($"[green]Wrote[/] {Markup.Escape(outPath!)} ({new FileInfo(outPath!).Length / 1048576.0:0.0} MiB); verified {plan.Count} tensors byte-for-byte against their sources.");
        return 0;
    }

    private static long AlignUp(long value, int alignment) => (value + alignment - 1) / alignment * alignment;
    private static void WriteU32(Stream s, uint v) => s.Write(BitConverter.GetBytes(v));
    private static void WriteU64(Stream s, ulong v) => s.Write(BitConverter.GetBytes(v));

    private static void PadTo(Stream s, long position)
    {
        long n = position - s.Position;
        if (n < 0) throw new InvalidOperationException("negative padding");
        if (n == 0) return;
        s.Write(new byte[n]);
    }

    private static void WriteBytes(Stream s, byte* p, long length)
    {
        const int chunk = 64 << 20;
        for (long done = 0; done < length;)
        {
            int n = (int)Math.Min(chunk, length - done);
            s.Write(new ReadOnlySpan<byte>(p + done, n));
            done += n;
        }
    }

    private static bool BytesEqual(byte* a, byte* b, long length)
    {
        const int chunk = 64 << 20;
        for (long done = 0; done < length;)
        {
            int n = (int)Math.Min(chunk, length - done);
            if (!new ReadOnlySpan<byte>(a + done, n).SequenceEqual(new ReadOnlySpan<byte>(b + done, n))) return false;
            done += n;
        }
        return true;
    }
}
