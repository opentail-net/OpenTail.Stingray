using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Engine.Verification;
using static OpenTail.Stingray.Tests.Cli.HubFixtures;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class LocalInventoryTests : IDisposable
{
    private readonly string _tempRoot = Path.Combine(Path.GetTempPath(), "stingray-inventory-test-" + Guid.NewGuid().ToString("N"));

    public LocalInventoryTests()
    {
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempRoot))
                Directory.Delete(_tempRoot, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [Fact]
    public void Catalogue_states_reported_correctly()
    {
        string homeDir = Path.Combine(_tempRoot, "home");
        Directory.CreateDirectory(homeDir);
        var home = new ModelHome(homeDir);

        // Case 1: Fresh home -> all entries missing
        var report1 = LocalInventory.Scan(home: home);
        Assert.All(report1.CatalogItems, item => Assert.Equal(InstallState.Missing, item.State));

        // Case 2: Create a partial file for first entry
        var entry = ModelCatalog.Entries[0];
        string mainPartPath = home.PathOf(entry.MainFile) + ModelHome.PartialSuffix;
        File.WriteAllBytes(mainPartPath, [1, 2, 3]);

        var report2 = LocalInventory.Scan(home: home);
        var item2 = report2.CatalogItems.First(i => i.Entry.Id == entry.Id);
        Assert.Equal(InstallState.Partial, item2.State);

        // Case 3: Complete all files of first entry with correct sizes
        File.Delete(mainPartPath);
        foreach (var file in entry.Files)
        {
            string path = home.PathOf(file);
            using var fs = File.Create(path);
            fs.SetLength(file.Size);
        }

        var report3 = LocalInventory.Scan(home: home);
        var item3 = report3.CatalogItems.First(i => i.Entry.Id == entry.Id);
        Assert.Equal(InstallState.Installed, item3.State);
    }

    [Fact]
    public void Architecture_admission_distinguishes_admitted_from_unsupported()
    {
        string modelsDir = Path.Combine(_tempRoot, "models");
        Directory.CreateDirectory(modelsDir);

        File.WriteAllBytes(Path.Combine(modelsDir, "a-llama.gguf"), Gguf("llama", blocks: 1));
        File.WriteAllBytes(Path.Combine(modelsDir, "b-deepseek4.gguf"), Gguf("deepseek4", blocks: 1));
        File.WriteAllBytes(Path.Combine(modelsDir, "c-unknown.gguf"), Gguf("unknown_custom_arch", blocks: 1));

        var report = LocalInventory.Scan(home: new ModelHome(Path.Combine(_tempRoot, "empty_home")), extraDirs: [modelsDir]);
        Assert.Equal(3, report.LocalGgufs.Count);

        var llama = report.LocalGgufs.First(g => g.FileName == "a-llama.gguf");
        Assert.Equal("llama", llama.Architecture);
        Assert.Equal("admitted", llama.AdmissionStatus);
        Assert.Equal("Float32", llama.Quantization);
        Assert.True(llama.IsReadable);

        var deepseek4 = report.LocalGgufs.First(g => g.FileName == "b-deepseek4.gguf");
        Assert.Equal("deepseek4", deepseek4.Architecture);
        Assert.Equal("not supported", deepseek4.AdmissionStatus);
        Assert.True(deepseek4.IsReadable);

        var unknown = report.LocalGgufs.First(g => g.FileName == "c-unknown.gguf");
        Assert.Equal("unknown_custom_arch", unknown.Architecture);
        Assert.Equal("not supported", unknown.AdmissionStatus);
        Assert.True(unknown.IsReadable);
    }

    [Fact]
    public void Unreadable_file_does_not_halt_scan()
    {
        string modelsDir = Path.Combine(_tempRoot, "models");
        Directory.CreateDirectory(modelsDir);

        File.WriteAllBytes(Path.Combine(modelsDir, "0-broken.gguf"), [0x01, 0x02, 0x03, 0x04, 0x05]);
        File.WriteAllBytes(Path.Combine(modelsDir, "1-valid.gguf"), Gguf("llama", blocks: 1));

        var report = LocalInventory.Scan(home: new ModelHome(Path.Combine(_tempRoot, "empty_home")), extraDirs: [modelsDir]);
        Assert.Equal(2, report.LocalGgufs.Count);

        var broken = report.LocalGgufs.First(g => g.FileName == "0-broken.gguf");
        Assert.False(broken.IsReadable);
        Assert.Equal("unreadable", broken.Architecture);
        Assert.Equal("not supported", broken.AdmissionStatus);
        Assert.NotNull(broken.ErrorMessage);

        var valid = report.LocalGgufs.First(g => g.FileName == "1-valid.gguf");
        Assert.True(valid.IsReadable);
        Assert.Equal("llama", valid.Architecture);
        Assert.Equal("admitted", valid.AdmissionStatus);
        Assert.Null(valid.ErrorMessage);
    }

    [Fact]
    public void Fit_status_respects_injected_ram()
    {
        string modelsDir = Path.Combine(_tempRoot, "models");
        Directory.CreateDirectory(modelsDir);
        File.WriteAllBytes(Path.Combine(modelsDir, "model.gguf"), Gguf("llama", blocks: 2));

        var home = new ModelHome(Path.Combine(_tempRoot, "empty_home"));

        var reportFits = LocalInventory.Scan(home: home, extraDirs: [modelsDir], ramBytes: 64L * 1024 * 1024 * 1024);
        var modelFits = reportFits.LocalGgufs.Single();
        Assert.Equal("fits", modelFits.FitStatus);

        var reportBlocked = LocalInventory.Scan(home: home, extraDirs: [modelsDir], ramBytes: 100L * 1024 * 1024);
        var modelBlocked = reportBlocked.LocalGgufs.Single();
        Assert.Equal("does not fit", modelBlocked.FitStatus);
    }

    [Fact]
    public void Verify_integrity_checks_hash_and_reports_intact_or_damaged()
    {
        string homeDir = Path.Combine(_tempRoot, "home");
        Directory.CreateDirectory(homeDir);
        var home = new ModelHome(homeDir);

        var entry = ModelCatalog.Entries[0];
        foreach (var file in entry.Files)
        {
            string path = home.PathOf(file);
            using var fs = File.Create(path);
            fs.SetLength(file.Size);

            // Write valid sidecar matching expected sha256, size, and mtime
            long mtime = File.GetLastWriteTimeUtc(path).Ticks;
            File.WriteAllText(ModelFingerprinter.SidecarPath(path), $"{file.Sha256} {file.Size} {mtime}{Environment.NewLine}");
        }

        // Without --verify: NotChecked
        var reportUnverified = LocalInventory.Scan(home: home, verify: false);
        var itemUnverified = reportUnverified.CatalogItems.First(i => i.Entry.Id == entry.Id);
        Assert.Equal(IntegrityState.NotChecked, itemUnverified.Integrity);

        // With --verify: Intact
        var reportIntact = LocalInventory.Scan(home: home, verify: true);
        var itemIntact = reportIntact.CatalogItems.First(i => i.Entry.Id == entry.Id);
        Assert.Equal(IntegrityState.Intact, itemIntact.Integrity);

        // Corrupt sidecar with wrong hash: Damaged
        string firstPath = home.PathOf(entry.Files[0]);
        long mtimeDamaged = File.GetLastWriteTimeUtc(firstPath).Ticks;
        File.WriteAllText(ModelFingerprinter.SidecarPath(firstPath), $"{new string('0', 64)} {entry.Files[0].Size} {mtimeDamaged}{Environment.NewLine}");

        var reportDamaged = LocalInventory.Scan(home: home, verify: true);
        var itemDamaged = reportDamaged.CatalogItems.First(i => i.Entry.Id == entry.Id);
        Assert.Equal(IntegrityState.Damaged, itemDamaged.Integrity);
    }

    [Fact]
    public void Scan_never_modifies_or_deletes_any_model_file()
    {
        string modelsDir = Path.Combine(_tempRoot, "models");
        Directory.CreateDirectory(modelsDir);

        string p1 = Path.Combine(modelsDir, "valid.gguf");
        string p2 = Path.Combine(modelsDir, "corrupt.gguf");

        File.WriteAllBytes(p1, Gguf("llama", blocks: 1));
        File.WriteAllBytes(p2, [1, 2, 3, 4, 5]);

        var info1Before = new FileInfo(p1);
        var info2Before = new FileInfo(p2);
        long len1 = info1Before.Length;
        DateTime mtime1 = info1Before.LastWriteTimeUtc;
        long len2 = info2Before.Length;
        DateTime mtime2 = info2Before.LastWriteTimeUtc;

        _ = LocalInventory.Scan(home: new ModelHome(Path.Combine(_tempRoot, "empty_home")), extraDirs: [modelsDir], verify: true);

        var info1After = new FileInfo(p1);
        var info2After = new FileInfo(p2);

        Assert.True(info1After.Exists);
        Assert.True(info2After.Exists);
        Assert.Equal(len1, info1After.Length);
        Assert.Equal(mtime1, info1After.LastWriteTimeUtc);
        Assert.Equal(len2, info2After.Length);
        Assert.Equal(mtime2, info2After.LastWriteTimeUtc);
    }

    [Fact]
    public void Extra_model_dirs_and_environment_variable_scanned()
    {
        string dir1 = Path.Combine(_tempRoot, "dir1");
        string dir2 = Path.Combine(_tempRoot, "dir2");
        Directory.CreateDirectory(dir1);
        Directory.CreateDirectory(dir2);

        File.WriteAllBytes(Path.Combine(dir1, "model1.gguf"), Gguf("llama", blocks: 1));
        File.WriteAllBytes(Path.Combine(dir2, "model2.gguf"), Gguf("llama", blocks: 1));

        // Scan with explicit extraDirs
        var report1 = LocalInventory.Scan(home: new ModelHome(Path.Combine(_tempRoot, "empty_home")), extraDirs: [dir1, dir2]);
        Assert.Equal(2, report1.LocalGgufs.Count);

        // Scan via STINGRAY_MODEL_DIRS
        string origEnv = Environment.GetEnvironmentVariable(LocalInventory.ModelDirsEnvVar) ?? "";
        try
        {
            Environment.SetEnvironmentVariable(LocalInventory.ModelDirsEnvVar, $"{dir1}{Path.PathSeparator}{dir2}");
            var report2 = LocalInventory.Scan(home: new ModelHome(Path.Combine(_tempRoot, "empty_home")));
            Assert.Equal(2, report2.LocalGgufs.Count);
        }
        finally
        {
            Environment.SetEnvironmentVariable(LocalInventory.ModelDirsEnvVar, string.IsNullOrEmpty(origEnv) ? null : origEnv);
        }
    }

    [Fact]
    public void ModelsCommand_validation_rejects_task_with_local_or_verify()
    {
        var settings1 = new ModelsCommand.Settings { Task = "chat", Local = true };
        Assert.NotNull(settings1.Validate());

        var settings2 = new ModelsCommand.Settings { Task = "chat", Verify = true };
        Assert.NotNull(settings2.Validate());

        var settings3 = new ModelsCommand.Settings { Local = true, Verify = true };
        Assert.Null(settings3.Validate());
    }
}
