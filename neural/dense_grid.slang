implementing neural;

////////////////
// Dense grid //
////////////////

public struct DenseGridEncoder<T, int Dim, int Levels, int Features, Storage> : ILearnableEncoder<T, Dim, Features * Levels, Storage>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Storage : IStorage<T>
{
    public static const int In = Dim;
    public static const int Out = Features * Levels;

    Storage.Address baseAddress;
    Array<int, Levels> gridResolution;

    [BackwardDifferentiable]
    public static T evalLevelLinear<InputVector>(Storage storage, Storage.Address baseAddress, int levelOffset, int resolution, InputVector input, int feature)
        where InputVector : IVector<T, Dim>
    {
        static_assert(Dim == 1, "Linear interpolation is only supported for 1D inputs");

        let snapped = snapToLattice<InputVector, T, Dim>(input, resolution);
        
        let idx0 = levelOffset + snapped.lower[0] * Features + feature;
        let idx1 = levelOffset + snapped.upper[0] * Features + feature;
        let value0 = storage.read(Storage.getOffset(baseAddress, idx0));
        let value1 = storage.read(Storage.getOffset(baseAddress, idx1));
        return lerp(value0, value1, snapped.t[0]);
    }
    
    [BackwardDifferentiable]
    public static T evalLevelBilinear<InputVector>(Storage storage, Storage.Address baseAddress, int levelOffset, int resolution, InputVector input, int feature)
        where InputVector : IVector<T, Dim>
    {
        static_assert(Dim == 2, "Bilinear interpolation is only supported for 2D inputs");

        let snapped = snapToLattice<InputVector, T, Dim>(input, resolution);
        
        let idx00 = levelOffset + (snapped.lower[1] * resolution + snapped.lower[0]) * Features + feature;
        let idx01 = levelOffset + (snapped.upper[1] * resolution + snapped.lower[0]) * Features + feature;
        let idx10 = levelOffset + (snapped.lower[1] * resolution + snapped.upper[0]) * Features + feature;
        let idx11 = levelOffset + (snapped.upper[1] * resolution + snapped.upper[0]) * Features + feature;
        
        let value00 = storage.read(Storage.getOffset(baseAddress, idx00));
        let value01 = storage.read(Storage.getOffset(baseAddress, idx01));
        let value10 = storage.read(Storage.getOffset(baseAddress, idx10));
        let value11 = storage.read(Storage.getOffset(baseAddress, idx11));

        let lerp0 = lerp(value00, value10, snapped.t[0]);
        let lerp1 = lerp(value01, value11, snapped.t[0]);
        return lerp(lerp0, lerp1, snapped.t[1]);
    }

    [BackwardDifferentiable]
    public static T evalLevelTrilinear<InputVector>(Storage storage, Storage.Address baseAddress, int levelOffset, int resolution, InputVector input, int feature)
        where InputVector : IVector<T, Dim>
    {
        static_assert(Dim == 3, "Trilinear interpolation is only supported for 3D inputs");

        let snapped = snapToLattice<InputVector, T, Dim>(input, resolution);
        
        let idx000 = levelOffset + (snapped.lower[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.lower[0]) * Features + feature;
        let idx001 = levelOffset + (snapped.upper[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.lower[0]) * Features + feature;
        let idx010 = levelOffset + (snapped.lower[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.lower[0]) * Features + feature;
        let idx011 = levelOffset + (snapped.upper[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.lower[0]) * Features + feature;
        let idx100 = levelOffset + (snapped.lower[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.upper[0]) * Features + feature;
        let idx101 = levelOffset + (snapped.upper[2] * resolution * resolution + snapped.lower[1] * resolution + snapped.upper[0]) * Features + feature;
        let idx110 = levelOffset + (snapped.lower[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.upper[0]) * Features + feature;
        let idx111 = levelOffset + (snapped.upper[2] * resolution * resolution + snapped.upper[1] * resolution + snapped.upper[0]) * Features + feature;
        
        let value000 = storage.read(Storage.getOffset(baseAddress, idx000));
        let value001 = storage.read(Storage.getOffset(baseAddress, idx001));
        let value010 = storage.read(Storage.getOffset(baseAddress, idx010));
        let value011 = storage.read(Storage.getOffset(baseAddress, idx011));
        let value100 = storage.read(Storage.getOffset(baseAddress, idx100));
        let value101 = storage.read(Storage.getOffset(baseAddress, idx101));
        let value110 = storage.read(Storage.getOffset(baseAddress, idx110));
        let value111 = storage.read(Storage.getOffset(baseAddress, idx111));

        let lerp0 = lerp(value000, value100, snapped.t[0]);
        let lerp1 = lerp(value001, value101, snapped.t[0]);
        let lerp2 = lerp(value010, value110, snapped.t[0]);
        let lerp3 = lerp(value011, value111, snapped.t[0]);

        let lerp4 = lerp(lerp0, lerp2, snapped.t[1]);
        let lerp5 = lerp(lerp1, lerp3, snapped.t[1]);
        return lerp(lerp4, lerp5, snapped.t[2]);
    }

    [BackwardDifferentiable]
    public static T evalLevelGeneral<InputVector>(Storage storage, Storage.Address baseAddress, int levelOffset, int resolution, InputVector input, int feature)
        where InputVector : IVector<T, Dim>
    {
        // N-dimensional multilinear interpolation
        // We need 2^Dim corner values for full interpolation
        
        let snapped = snapToLattice<InputVector, T, Dim>(input, resolution);
        
        // Interpolate over 2^Dim corners
        T result = T(0.0);
        int numCorners = 1;
        [ForceUnroll]
        for (int d = 0; d < Dim; d++)
            numCorners *= 2;
            
        [ForceUnroll]
        for (int corner = 0; corner < numCorners; corner++)
        {
            // Decode corner index to coordinate selection
            int[Dim] cornerCoords;
            T cornerWeight = T(1.0);
            int temp = corner;
            
            [ForceUnroll]
            for (int d = 0; d < Dim; d++)
            {
                int bit = temp % 2;
                temp /= 2;
                
                if (bit == 0)
                {
                    cornerCoords[d] = snapped.lower[d];
                    cornerWeight *= (T(1.0) - snapped.t[d]);
                }
                else
                {
                    cornerCoords[d] = snapped.upper[d];
                    cornerWeight *= snapped.t[d];
                }
            }
            
            // Calculate flat index for this corner
            int flatIndex = calculateFlatIndex<Dim>(cornerCoords, resolution);
            
            // Read value and accumulate weighted result
            let value = storage.read(Storage.getOffset(baseAddress, levelOffset + flatIndex * Features + feature));
            result += cornerWeight * value;
        }
        
        return result;
    }

    [BackwardDifferentiable]
    public T evalLevel<InputVector>(Storage storage, int levelOffset, InputVector input, int level, int feature)
        where InputVector : IVector<T, Dim>
    {
        let resolution = gridResolution[level];
        if (Dim == 1)
            return evalLevelLinear(storage, baseAddress, levelOffset, resolution, input, feature);
        else if (Dim == 2)
            return evalLevelBilinear(storage, baseAddress, levelOffset, resolution, input, feature);
        else if (Dim == 3)
            return evalLevelTrilinear(storage, baseAddress, levelOffset, resolution, input, feature);
        else
            return evalLevelGeneral(storage, baseAddress, levelOffset, resolution, input, feature);
    }

    [BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels>
    {
        OutputVector output;
        
        int levelOffset = 0;
        
        [ForceUnroll]
        for (int level = 0; level < Levels; level++)
        {
            [ForceUnroll]
            for (int feature = 0; feature < Features; feature++)
            {
                output[level * Features + feature] = evalLevel<InputVector>(
                    storage,
                    levelOffset,
                    input,
                    level,
                    feature
                );
            }
            
            // Calculate elements for this level
            int elementsThisLevel = Features;
            [ForceUnroll]
            for (int i = 0; i < Dim; i++)
                elementsThisLevel *= gridResolution[level];
            levelOffset += elementsThisLevel;
        }

        return output;
    }
}
