using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

public sealed class DequantizeMxfp4Tests
{
    [Fact]
    public void ToFloat32_DecodesE8M0ScaleAndPackedNibbles()
    {
        // E8M0 exponent 128 represents 1.0. Each byte contains values for
        // the first and second 16-value halves of the 32-value block.
        byte[] source = new byte[17];
        source[0] = 128;
        source[1] = 0x91; // +1 at index 0; -1 at index 16
        source[2] = 0xE5; // +6 at index 1; -8 at index 17
        source[3] = 0xF7; // +12 at index 2; -12 at index 18

        float[] actual = new float[32];
        Dequantize.ToFloat32(source, actual, DType.MXFP4, actual.Length);

        Assert.Equal(1f, actual[0]);
        Assert.Equal(6f, actual[1]);
        Assert.Equal(12f, actual[2]);
        Assert.Equal(-1f, actual[16]);
        Assert.Equal(-8f, actual[17]);
        Assert.Equal(-12f, actual[18]);
    }

    [Fact]
    public void ToFloat32_DecodesE8M0SubnormalAndNormalScales()
    {
        byte[] source = new byte[34];
        source[0] = 0;   // GGML E8M0 maps this to 2^-128
        source[1] = 0x01;
        source[17] = 129; // 2.0
        source[18] = 0x01;
        float[] actual = new float[64];

        Dequantize.ToFloat32(source, actual, DType.MXFP4, actual.Length);

        Assert.Equal(float.Pow(2f, -128), actual[0]);
        Assert.Equal(2f, actual[32]);
    }

    [Fact]
    public void ToFloat32_RejectsPartialMxfp4Block()
    {
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(new byte[17], new float[31], DType.MXFP4, 31));
    }

    [Fact]
    public void ToFloat32_DecodesAsymmetricQ4AndQ5Blocks()
    {
        byte[] q41 = new byte[20];
        q41[0] = 0; q41[1] = 0x3C; // d = 1
        q41[2] = 0; q41[3] = 0xC0; // m = -2
        q41[4] = 0xF3;
        float[] q41Actual = new float[32];
        Dequantize.ToFloat32(q41, q41Actual, DType.Q4_1, 32);
        Assert.Equal(1f, q41Actual[0]);
        Assert.Equal(13f, q41Actual[16]);

        byte[] q51 = new byte[24];
        q51[0] = 0; q51[1] = 0x3C; // d = 1
        q51[2] = 0; q51[3] = 0xBC; // m = -1
        q51[4] = 1; // high bit for element 0
        q51[8] = 0x02;
        float[] q51Actual = new float[32];
        Dequantize.ToFloat32(q51, q51Actual, DType.Q5_1, 32);
        Assert.Equal(17f, q51Actual[0]);
        Assert.Equal(-1f, q51Actual[16]);
    }

    [Fact]
    public void ToFloat32_DecodesQ8_1AndIq4Nl()
    {
        byte[] q81 = new byte[36];
        q81[0] = 0; q81[1] = 0x38; // d = 0.5
        q81[4] = unchecked((byte)-6);
        float[] q81Actual = new float[32];
        Dequantize.ToFloat32(q81, q81Actual, DType.Q8_1, 32);
        Assert.Equal(-3f, q81Actual[0]);

        byte[] iq4Nl = new byte[18];
        iq4Nl[0] = 0; iq4Nl[1] = 0x3C; // d = 1
        iq4Nl[2] = 0xF0;
        float[] iq4NlActual = new float[32];
        Dequantize.ToFloat32(iq4Nl, iq4NlActual, DType.IQ4_NL, 32);
        Assert.Equal(-127f, iq4NlActual[0]);
        Assert.Equal(113f, iq4NlActual[16]);
    }
}
