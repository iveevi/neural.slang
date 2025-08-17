implementing neural;

// Snapped position and interpolation weights
public struct Snapped<T, int D>
{
    public Array<int, D> lower;
    public Array<int, D> upper;
    public Array<T, D> t;
}

// Snap a position to the lattice, returning the lower and upper indices and interpolation weights
public Snapped<T, D> snapToLattice<Vector, T, int D>(no_diff Vector position, int[D] resolution)
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Vector : IVector<T, D>
{
    Snapped<T, D> result;

    [ForceUnroll]
    for (int d = 0; d < D; d++)
    {
        let scaled = position[d] * T(resolution[d]);
        let index = floor(scaled).toInt();
        result.lower[d] = index;
        result.upper[d] = index + 1;
        result.t[d] = scaled - T(index);
    }

    return result;
}

// Overload that takes a single resolution value for all dimensions
public Snapped<T, D> snapToLattice<Vector, T, int D>(no_diff Vector position, int resolution)
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Vector : IVector<T, D>
{
    Snapped<T, D> result;

    [ForceUnroll]
    for (int d = 0; d < D; d++)
    {
        let scaled = position[d] * T(resolution);
        let index = floor(scaled).toInt();
        result.lower[d] = index;
        result.upper[d] = index + 1;
        result.t[d] = scaled - T(index);
    }

    return result;
}

// Calculate flat index from N-dimensional coordinates using row-major ordering
public int calculateFlatIndex<int D>(int[D] coords, int resolution)
{
    int flatIndex = 0;
    int stride = 1;
    [ForceUnroll]
    for (int d = 0; d < D; d++)
    {
        flatIndex += coords[d] * stride;
        stride *= resolution;
    }
    return flatIndex;
}