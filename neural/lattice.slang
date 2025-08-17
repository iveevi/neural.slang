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
        let scaled = position[d] * T(resolution[d] - 1);
        let index = floor(scaled).toInt();
        // Clamp indices to prevent out-of-bounds access (matching PyTorch exactly)
        result.lower[d] = max(0, min(index, resolution[d] - 2));
        result.upper[d] = max(1, min(index + 1, resolution[d] - 1));
        
        // Handle boundary case: if we're at the very end, ensure proper interpolation
        if (index >= resolution[d] - 1)
        {
            result.lower[d] = resolution[d] - 2;
            result.upper[d] = resolution[d] - 1;
            result.t[d] = T(1.0);  // Use full weight for the last element
        }
        else
        {
            result.t[d] = scaled - T(index);
        }
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
        let scaled = position[d] * T(resolution - 1);
        let index = floor(scaled).toInt();
        // Clamp indices to prevent out-of-bounds access (matching PyTorch exactly)
        result.lower[d] = max(0, min(index, resolution - 2));
        result.upper[d] = max(1, min(index + 1, resolution - 1));
        
        // Handle boundary case: if we're at the very end, ensure proper interpolation
        if (index >= resolution - 1)
        {
            result.lower[d] = resolution - 2;
            result.upper[d] = resolution - 1;
            result.t[d] = T(1.0);  // Use full weight for the last element
        }
        else
        {
            result.t[d] = scaled - T(index);
        }
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