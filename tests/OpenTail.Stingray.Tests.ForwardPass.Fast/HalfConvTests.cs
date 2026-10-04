using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// <see cref="HalfConv.ToFloat(ushort)"/> replaces <c>(float)BitConverter.UInt16BitsToHalf(bits)</c> in every quantised dot kernel (the cast compiles to an
/// un-inlined call). It must be a bit-for-bit drop-in, so check all 65536 patterns, including zeros, subnormals, infinities and NaN payloads.
/// </summary>
public sealed class HalfConvTests
{
    [Fact]
    public void ToFloat_IsBitIdenticalToHalfCast_ForAll65536Patterns()
    {
        var mismatches = new List<string>();
        for (var h = 0; h <= ushort.MaxValue; h++)
        {
            var expected = BitConverter.SingleToUInt32Bits((float)BitConverter.UInt16BitsToHalf((ushort)h));
            var actual = BitConverter.SingleToUInt32Bits(HalfConv.ToFloat((ushort)h));
            if (expected != actual && mismatches.Count < 8) mismatches.Add($"0x{h:X4}: expected 0x{expected:X8}, got 0x{actual:X8}");
        }
        Assert.True(mismatches.Count == 0, string.Join("; ", mismatches));
    }

    [Fact]
    public void ToFloat_ByteOverload_IsLittleEndian()
    {
        Assert.Equal(1.0f, HalfConv.ToFloat(0x00, 0x3C));
        Assert.Equal(-2.0f, HalfConv.ToFloat(0x00, 0xC0));
    }
}
