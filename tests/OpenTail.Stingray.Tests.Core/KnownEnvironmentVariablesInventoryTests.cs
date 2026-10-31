using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Tests.Core;

public sealed partial class KnownEnvironmentVariablesTests
{
    /// <summary>
    /// Registry names that still have no row in <c>docs/reference/env-var-inventory.md</c> (found 2026-10-31: the table was last
    /// reconciled at 162 names and the registry has since grown). This list may only shrink: add the row, then delete the name here.
    /// A new variable must arrive with its row; adding it to this list instead is a visible step backwards in review.
    /// </summary>
    private static readonly string[] KnownUndocumented =
    [
        "STINGRAY_PROFILE_VULKAN_UPLOADS",
        "STINGRAY_PROFILE_WEIGHT_READ",
        "STINGRAY_QWEN_GPU_BLOCKS",
        "STINGRAY_QWEN_GPU_FP16",
        "STINGRAY_QWEN_MLP_CHUNKS",
        "STINGRAY_QWEN_PROFILE",
        "STINGRAY_QWENASR_DIAG",
        "STINGRAY_QWENIMAGE_DUMP_COND_PATH",
        "STINGRAY_QWENIMAGE_DUMP_LATENT_PATH",
        "STINGRAY_QWENIMAGE_INJECT_NOISE_PATH",
        "STINGRAY_RECURRENT_BATCHED_PREFILL",
        "STINGRAY_RERANK_MODEL",
        "STINGRAY_RVC_PROFILE",
        "STINGRAY_RVC_TRACE",
        "STINGRAY_SA3_DUMP",
        "STINGRAY_SA3_NOISE",
        "STINGRAY_SD3_DUMP_LATENT_PATH",
        "STINGRAY_SD3_FORCE_GPU_WEIGHTS_F32",
        "STINGRAY_SD3_GPU_FP16",
        "STINGRAY_SD3_INJECT_NOISE_PATH",
        "STINGRAY_SD3_POSEMBED_LEFT",
        "STINGRAY_SD3_POSEMBED_TOP",
        "STINGRAY_SILERO_VAD_PATH",
        "STINGRAY_T5_DUMP_PREFIX",
        "STINGRAY_TTS_PROFILE",
        "STINGRAY_TTS_TRACE",
        "STINGRAY_VISION_GEMM",
        "STINGRAY_VK_MATVEC_V2",
        "STINGRAY_VK_MXFP4_V2",
        "STINGRAY_VK_PATH3_CHUNK",
        "STINGRAY_VK_QGEMM_SMALLM",
        "STINGRAY_VULKAN_DECODE_ATTN_GQA",
        "STINGRAY_VULKAN_DECODE_PROFILE",
        "STINGRAY_VULKAN_FLASH_PREFILL",
        "STINGRAY_VULKAN_HYBRID_PREFILL_CHUNK",
        "STINGRAY_VULKAN_MIN_FREE_RAM_MB",
        "STINGRAY_WAN_DEBUG_ATTNMAP",
        "STINGRAY_WAN_DEBUG_CROSSREF",
        "STINGRAY_WAN_DEBUG_CROSSREF_DIR",
        "STINGRAY_WAN_DEBUG_EXACT_GELU",
        "STINGRAY_WAN_DEBUG_PERBLOCK",
        "STINGRAY_WAN_DEBUG_VELOCITY",
        "STINGRAY_WAN_PROFILE",
        "STINGRAY_WAN_SIGN",
        "STINGRAY_XTTS_LAYER_TRACE",
        "STINGRAY_XTTS_TRACE",
        "STINGRAY_ZIMAGE_DUMP_LATENT",
    ];

    [GeneratedRegex(@"^\|\s*`(STINGRAY_[A-Z0-9_]+)`", RegexOptions.Multiline)]
    private static partial Regex InventoryRow();

    [Fact]
    public void InventoryTable_CoversTheRegistry_ExceptTheListedGap()
    {
        string? root = FindRepoRoot();
        Assert.SkipUnless(root is not null, "repository layout not found (packaged run)");
        string path = Path.Combine(root!, "docs", "reference", "env-var-inventory.md");
        Assert.SkipUnless(File.Exists(path), "env-var-inventory.md not present");

        var rows = InventoryRow().Matches(File.ReadAllText(path)).Select(m => m.Groups[1].Value).ToHashSet(StringComparer.Ordinal);
        var undocumented = KnownEnvironmentVariables.All.Where(n => !rows.Contains(n)).ToHashSet(StringComparer.Ordinal);

        var newlyUndocumented = undocumented.Except(KnownUndocumented, StringComparer.Ordinal).Order(StringComparer.Ordinal).ToList();
        Assert.True(newlyUndocumented.Count == 0,
            "registered but missing from docs/reference/env-var-inventory.md (add a row): " + string.Join(", ", newlyUndocumented));

        var nowDocumented = KnownUndocumented.Where(n => !undocumented.Contains(n)).ToList();
        Assert.True(nowDocumented.Count == 0,
            "now documented (or no longer registered): delete from KnownUndocumented: " + string.Join(", ", nowDocumented));

        var orphanRows = rows.Where(r => !KnownEnvironmentVariables.All.Contains(r)).Order(StringComparer.Ordinal).ToList();
        Assert.True(orphanRows.Count == 0, "inventory rows for names the registry does not contain: " + string.Join(", ", orphanRows));
    }
}
