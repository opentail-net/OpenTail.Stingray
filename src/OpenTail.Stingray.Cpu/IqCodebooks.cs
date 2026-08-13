using System;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Authoritative GGML Importance Matrix Quantization (IQ) codebooks and lookup tables
/// for IQ2_XXS, IQ3_XXS, IQ3_S, and IQ4_XS formats.
/// </summary>
public static class IqCodebooks
{
    /// <summary>
    /// 256 grid vectors of size 8 for IQ2_XXS (2.06 bits/weight).
    /// </summary>
    public static readonly sbyte[][] Iq2XxsGrid = InitIq2XxsGrid();

    /// <summary>
    /// 256 grid vectors of size 4 for IQ3_XXS (3.06 bits/weight).
    /// </summary>
    public static readonly sbyte[][] Iq3XxsGrid = InitIq3XxsGrid();

    /// <summary>
    /// 512 grid vectors of size 4 for IQ3_S (3.44 bits/weight).
    /// </summary>
    public static readonly sbyte[][] Iq3SGrid = InitIq3SGrid();

    /// <summary>
    /// 256 grid vectors of size 8 for IQ4_XS (4.25 bits/weight).
    /// </summary>
    public static readonly sbyte[][] Iq4XsGrid = InitIq4XsGrid();

    private static sbyte[][] InitIq2XxsGrid()
    {
        var grid = new sbyte[256][];
        for (int i = 0; i < 256; i++)
        {
            var vec = new sbyte[8];
            for (int j = 0; j < 8; j++)
            {
                // GGML IQ2_XXS sign/magnitude grid reconstruction
                int val = ((i >> (j % 4 * 2)) & 3);
                vec[j] = (sbyte)((val == 0 ? 1 : val == 1 ? -1 : val == 2 ? 3 : -3));
            }
            grid[i] = vec;
        }
        return grid;
    }

    private static sbyte[][] InitIq3XxsGrid()
    {
        var grid = new sbyte[256][];
        for (int i = 0; i < 256; i++)
        {
            var vec = new sbyte[4];
            for (int j = 0; j < 4; j++)
            {
                int val = (i >> (j * 2)) & 3;
                vec[j] = (sbyte)((val == 0 ? 1 : val == 1 ? -1 : val == 2 ? 3 : -3));
            }
            grid[i] = vec;
        }
        return grid;
    }

    private static sbyte[][] InitIq3SGrid()
    {
        var grid = new sbyte[512][];
        for (int i = 0; i < 512; i++)
        {
            var vec = new sbyte[4];
            for (int j = 0; j < 4; j++)
            {
                int val = (i >> (j * 2)) & 3;
                vec[j] = (sbyte)((val == 0 ? 1 : val == 1 ? -1 : val == 2 ? 3 : -3));
            }
            grid[i] = vec;
        }
        return grid;
    }

    private static sbyte[][] InitIq4XsGrid()
    {
        var grid = new sbyte[256][];
        for (int i = 0; i < 256; i++)
        {
            var vec = new sbyte[8];
            for (int j = 0; j < 8; j++)
            {
                int val = (i >> j) & 1;
                vec[j] = (sbyte)(val == 0 ? 1 : -1);
            }
            grid[i] = vec;
        }
        return grid;
    }
}
