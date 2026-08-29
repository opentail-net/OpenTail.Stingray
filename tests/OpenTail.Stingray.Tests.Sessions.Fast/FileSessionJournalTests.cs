namespace OpenTail.Stingray.Tests.Sessions.Fast;

using System.Security.Cryptography;
using System.Text;

public sealed class FileSessionJournalTests
{
    [Fact]
    public void FileSessionJournal_AppendsAndRecoversRecords()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"opentail_journal_test_{Guid.NewGuid():N}.journal");
        try
        {
            var id1 = SessionId.New();
            var rev1 = new SessionRevision(1);
            var op1 = SessionOperationId.New();
            byte[] payload1 = Encoding.UTF8.GetBytes("payload-turn-1");

            var id2 = SessionId.New();
            var rev2 = new SessionRevision(2);
            var op2 = SessionOperationId.New();
            byte[] payload2 = Encoding.UTF8.GetBytes("payload-turn-2");

            var rec1 = new SessionJournalRecord(SessionJournalRecordKind.OperationCommitted, id1, rev1, op1, payload1);
            var rec2 = new SessionJournalRecord(SessionJournalRecordKind.RevisionCheckpoint, id2, rev2, op2, payload2);

            using (var journal = new FileSessionJournal(tempFile))
            {
                journal.Append(rec1);
                journal.Append(rec2);
            }

            using (var recoveryJournal = new FileSessionJournal(tempFile))
            {
                var recovered = recoveryJournal.RecoverAndTruncateCorruptTail();
                Assert.Equal(2, recovered.Count);

                Assert.Equal(rec1, recovered[0]);
                Assert.Equal(rec2, recovered[1]);
            }
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    [Fact]
    public void FileSessionJournal_TruncatesCorruptedTrailingPayload()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"opentail_journal_corrupt_test_{Guid.NewGuid():N}.journal");
        try
        {
            var id1 = SessionId.New();
            var rev1 = new SessionRevision(1);
            var op1 = SessionOperationId.New();
            byte[] payload1 = Encoding.UTF8.GetBytes("valid-record-1");

            using (var journal = new FileSessionJournal(tempFile))
            {
                journal.Append(new SessionJournalRecord(SessionJournalRecordKind.OperationCommitted, id1, rev1, op1, payload1));
            }

            long validLength = new FileInfo(tempFile).Length;

            // Corrupt file by appending garbage trailing bytes
            using (var fs = new FileStream(tempFile, FileMode.Append, FileAccess.Write))
            {
                byte[] garbage = [0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0x99, 0x88];
                fs.Write(garbage, 0, garbage.Length);
                fs.Flush();
            }

            Assert.True(new FileInfo(tempFile).Length > validLength);

            using (var recoveryJournal = new FileSessionJournal(tempFile))
            {
                var recovered = recoveryJournal.RecoverAndTruncateCorruptTail();
                Assert.Single(recovered);
                Assert.Equal(id1, recovered[0].SessionId);
                Assert.Equal("valid-record-1", Encoding.UTF8.GetString(recovered[0].Payload));
            }

            // File length should now be restored to valid length after truncation
            Assert.Equal(validLength, new FileInfo(tempFile).Length);
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    [Fact]
    public void FileSessionJournal_RejectsCorruptedHeaderField()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"opentail_journal_hdr_corrupt_{Guid.NewGuid():N}.journal");
        try
        {
            var id1 = SessionId.New();
            var rev1 = new SessionRevision(1);
            var op1 = SessionOperationId.New();
            byte[] payload1 = Encoding.UTF8.GetBytes("valid-record-1");

            using (var journal = new FileSessionJournal(tempFile))
            {
                journal.Append(new SessionJournalRecord(SessionJournalRecordKind.OperationCommitted, id1, rev1, op1, payload1));
            }

            // Mutate SessionId/Revision bytes in header (located after magic+version 6 bytes)
            using (var fs = new FileStream(tempFile, FileMode.Open, FileAccess.ReadWrite))
            {
                fs.Position = 8; // Bit flip in SessionId header byte
                byte current = (byte)fs.ReadByte();
                fs.Position = 8;
                fs.WriteByte((byte)(current ^ 0xFF));
                fs.Flush();
            }

            using (var recoveryJournal = new FileSessionJournal(tempFile))
            {
                var recovered = recoveryJournal.RecoverAndTruncateCorruptTail();
                Assert.Empty(recovered); // Whole-record checksum must fail and reject record
            }
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    /// <summary>
    /// A torn write leaves a partial record whose length field holds whatever was already on disk —
    /// often an absurd value. That is the ORDINARY crash this journal exists to survive, so recovery
    /// must truncate the tail and return every record committed before it. An earlier revision threw
    /// here, which meant one interrupted write made the journal unopenable and lost all prior work.
    /// </summary>
    [Fact]
    public void FileSessionJournal_TornWriteWithAbsurdLength_RecoversPriorRecordsInsteadOfThrowing()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"opentail_journal_torn_{Guid.NewGuid():N}.journal");
        try
        {
            var id1 = SessionId.New();
            var rev1 = new SessionRevision(7);
            var op1 = SessionOperationId.New();
            byte[] payload1 = Encoding.UTF8.GetBytes("committed-before-the-crash");

            using (var journal = new FileSessionJournal(tempFile))
            {
                journal.Append(new SessionJournalRecord(SessionJournalRecordKind.OperationCommitted, id1, rev1, op1, payload1));
            }

            long validLength = new FileInfo(tempFile).Length;

            // Simulate a torn write: a full-looking header whose payload length is garbage, and no
            // payload or checksum behind it.
            using (var fs = new FileStream(tempFile, FileMode.Append, FileAccess.Write))
            {
                byte[] tornHeader = new byte[46];
                Array.Fill(tornHeader, (byte)0x7F);          // payload length reads as ~2.1 billion
                fs.Write(tornHeader, 0, tornHeader.Length);
                fs.Flush();
            }

            using (var recoveryJournal = new FileSessionJournal(tempFile))
            {
                var recovered = recoveryJournal.RecoverAndTruncateCorruptTail();

                Assert.Single(recovered);
                Assert.Equal(rev1, recovered[0].Revision);
                Assert.Equal("committed-before-the-crash", Encoding.UTF8.GetString(recovered[0].Payload));
                Assert.Equal(46, recoveryJournal.LastRecoveryTruncatedBytes);
            }

            Assert.Equal(validLength, new FileInfo(tempFile).Length);
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    /// <summary>
    /// Payloads larger than the journal's 4 KB FileStream buffer must survive a round trip. A single
    /// Stream.Read may return fewer bytes than requested, and recovery previously treated that short
    /// read as a corrupt tail — silently discarding a valid committed record.
    /// </summary>
    [Fact]
    public void FileSessionJournal_LargePayloadCrossingStreamBuffer_RoundTrips()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"opentail_journal_large_{Guid.NewGuid():N}.journal");
        try
        {
            byte[] large = new byte[512 * 1024];
            new Random(20260804).NextBytes(large);
            var id = SessionId.New();
            var rev = new SessionRevision(3);
            var op = SessionOperationId.New();

            using (var journal = new FileSessionJournal(tempFile))
            {
                journal.Append(new SessionJournalRecord(SessionJournalRecordKind.CursorState, id, rev, op, large));
            }

            using (var recoveryJournal = new FileSessionJournal(tempFile))
            {
                var recovered = recoveryJournal.RecoverAndTruncateCorruptTail();
                Assert.Single(recovered);
                Assert.Equal(large, recovered[0].Payload);
                Assert.Equal(0, recoveryJournal.LastRecoveryTruncatedBytes);
            }
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }
}
