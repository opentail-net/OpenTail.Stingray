namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Behaviour of <see cref="TensorShape"/>: immutability, validation, checked element counts,
/// value equality and formatting. See docs/3-product-and-runtime/2026-11-09-tensorshape-hardening-plan.md.
/// </summary>
public sealed class TensorShapeTests
{
    [Fact]
    public void Constructor_CopiesInputArray_SoLaterMutationDoesNotChangeTheShape()
    {
        long[] dims = { 2, 3 };
        var shape = new TensorShape(dims);

        dims[0] = 99;
        dims[1] = 0;

        Assert.Equal(new long[] { 2, 3 }, shape.Dims.ToArray());
        Assert.Equal(2, shape.Rank);
        Assert.Equal(6L, shape.ElementCount);
    }

    [Fact]
    public void CollectionExpression_WidensIntLiterals()
    {
        var shape = new TensorShape([2, 3, 4]);

        Assert.Equal(new long[] { 2, 3, 4 }, shape.Dims.ToArray());
        Assert.Equal(24L, shape.ElementCount);
    }

    [Fact]
    public void NegativeDimension_IsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorShape.D1(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorShape.D2(4, -2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorShape(new long[] { 3, -1, 2 }));
    }

    [Fact]
    public void NullArray_IsRejected()
    {
        Assert.Throws<ArgumentNullException>(() => new TensorShape(null!));
    }

    [Fact]
    public void ZeroDimension_IsAcceptedAsEmptyShape()
    {
        var shape = TensorShape.D2(0, 5);

        Assert.Equal(2, shape.Rank);
        Assert.Equal(0L, shape.ElementCount);
        Assert.Equal(new long[] { 0, 5 }, shape.Dims.ToArray());
    }

    [Fact]
    public void ZeroDimension_GivesZeroElements_EvenWhenOtherDimensionsWouldOverflow()
    {
        var shape = new TensorShape(new long[] { 1L << 40, 1L << 40, 0 });

        Assert.Equal(0L, shape.ElementCount);
    }

    [Fact]
    public void ElementCountOverflow_IsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorShape.D2(1L << 32, 1L << 32));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorShape.D3(1L << 31, 1L << 31, 1L << 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorShape.D2(long.MaxValue, 2));
    }

    [Fact]
    public void ElementCountAtTheLimit_IsAccepted()
    {
        var shape = TensorShape.D2(long.MaxValue, 1);

        Assert.Equal(long.MaxValue, shape.ElementCount);
    }

    [Fact]
    public void DimensionBeyondInt_IsKeptExactly()
    {
        // 2^31 * 4 = 2^33 elements: more than int.MaxValue, so an int element count would be wrong.
        var shape = TensorShape.D2(int.MaxValue + 1L, 4);

        Assert.Equal(8_589_934_592L, shape.ElementCount);
        Assert.Equal(int.MaxValue + 1L, shape.Dims[0]);
    }

    [Fact]
    public void ScalarShape_HasRankZeroAndOneElement()
    {
        var shape = new TensorShape(Array.Empty<long>());

        Assert.Equal(0, shape.Rank);
        Assert.Equal(1L, shape.ElementCount);
        Assert.Equal("()", shape.ToString());
    }

    [Fact]
    public void ElementCount_IsTheProductForHigherRanks()
    {
        var shape = new TensorShape(new long[] { 2, 3, 4, 5, 6, 7 });

        Assert.Equal(6, shape.Rank);
        Assert.Equal(5040L, shape.ElementCount);
    }

    [Fact]
    public void ValueEquality_ComparesDimensions_NotArrayIdentity()
    {
        var a = TensorShape.D2(2, 3);
        var b = TensorShape.D2(2, 3);

        Assert.True(a == b);
        Assert.True(a.Equals(b));
        Assert.True(a.Equals((object)b));
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.False(a != b);
    }

    [Fact]
    public void ValueEquality_DistinguishesOrderAndRank()
    {
        Assert.False(TensorShape.D2(2, 3) == TensorShape.D2(3, 2));
        Assert.False(TensorShape.D1(6) == TensorShape.D2(2, 3));
    }

    [Fact]
    public void HashSet_FindsAnEqualShape()
    {
        var set = new HashSet<TensorShape> { TensorShape.D3(2, 3, 4) };

        Assert.Contains(TensorShape.D3(2, 3, 4), set);
        Assert.DoesNotContain(TensorShape.D3(4, 3, 2), set);
    }

    [Fact]
    public void ToString_FormatsDimensions()
    {
        Assert.Equal("(2, 3, 4)", TensorShape.D3(2, 3, 4).ToString());
        Assert.Equal("(7)", TensorShape.D1(7).ToString());
    }

    [Fact]
    public void DimsSpan_ReflectsTheStoredValues()
    {
        var shape = TensorShape.D4(1, 2, 3, 4);

        Assert.Equal(4, shape.Dims.Length);
        Assert.Equal(3L, shape.Dims[2]);
        Assert.Equal(new long[] { 1, 2, 3, 4 }, shape.Dims.ToArray());
    }
}
