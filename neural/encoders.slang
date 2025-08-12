implementing neural;

//////////////////////
// Identity encoder //
//////////////////////

public struct IdentityEncoder<T, int N> : IEncoder<T, N, N>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, N>
        where OutputVector : IVector<T, N>
    {
        static_assert(InputVector is OutputVector,
            "Identity encoder requires input and "
            "output vectors to be the same");
        return __slang_noop_cast<OutputVector>(input);
    }
}

/////////////////////////
// Spherical harmonics //
/////////////////////////

public struct SphericalHarmonicsEncoder<T, int Levels> : IEncoder<T, 3, 2 * Levels + 1>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, 3>
        where OutputVector : IVector<T, 2 * Levels + 1>
    {
        OutputVector output;
        static_assert(false, "Not implemented");
        return output;
    }
}

///////////////////////
// Frequency encoder //
///////////////////////

public struct FrequencyEncoder<T, int Dim, int Levels> : IEncoder<T, Dim, 2 * Levels * Dim>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDerivative(evalBwd)]
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, 2 * Levels * Dim>
    {
        static_assert(Levels > 0, "Levels must be greater than 0");

        OutputVector output;

        [ForceUnroll]
        for (int i = 0; i < Dim; i++)
        {
            T k = T(1.0);

            for (int j = 0; j < Levels; j++)
            {
                let frequency = k * T.getPi();
                let sin = sin(frequency * input[i]);
                let cos = cos(frequency * input[i]);
                output[2 * j * Dim + i] = sin;
                output[2 * j * Dim + i + Dim] = cos;
                k *= T(2.0);
            }
        }

        return output;
    }

    void evalBwd<InputVector, OutputVector>(inout DifferentialPair<InputVector> dinput, OutputVector.Differential doutput)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, 2 * Levels * Dim>
    {
        let p = __slang_noop_cast<InputVector>(dinput.p);
        var d = __slang_noop_cast<InputVector>(dinput.d);
        let dout = __slang_noop_cast<OutputVector>(doutput);

        [ForceUnroll]
        for (int i = 0; i < Dim; i++)
        {
            T k = T(1.0);
            for (int j = 0; j < Levels; j++)
            {
                let frequency = k * T.getPi();
                let sinVal = sin(frequency * p[i]);
                let cosVal = cos(frequency * p[i]);
                let dsinOut = dout[2 * j * Dim + i];
                let dcosOut = dout[2 * j * Dim + i + Dim];
                d[i] += dsinOut * cosVal * frequency;
                d[i] += dcosOut * -sinVal * frequency;
                k *= T(2.0);
            }
        }

        dinput = DifferentialPair<InputVector>(p, __slang_noop_cast<InputVector.Differential>(d));
    }
}

////////////////
// Dense grid //
////////////////

public struct DenseGridEncoder<T, int Dim, int Levels, int Features, Storage> : ILearnableEncoder<T, Dim, Features * Levels, Storage>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable & IConvertibleToInt
    where Storage : IStorage<T>
{
    Array<Storage.Address, Levels> addresses;
    int gridResolution;

    public T evalLevel<InputVector>(Storage storage, Storage.Address levelAddressOffset, InputVector input, int level, int feature)
        where InputVector : IVector<T, Dim>
    {
        if (Dim == 1)
        {
            // Linear interpolation
            let index = input[0] * T(gridResolution);
            let index0 = floor(index);
            let index1 = ceil(index);

            let t = index - index0;
            let value0 = storage.read(Storage.getOffset(levelAddressOffset, index0.toInt()));
            let value1 = storage.read(Storage.getOffset(levelAddressOffset, index1.toInt()));
            return lerp(value0, value1, t);
        }
        else if (Dim == 2)
        {
            // Bilinear interpolation
            let x = input[0] * T(gridResolution);
            let y = input[1] * T(gridResolution);
            
            let x0 = floor(x);
            let x1 = ceil(x);
            let y0 = floor(y);
            let y1 = ceil(y);
            
            let tx = x - x0;
            let ty = y - y0;
            
            // Calculate grid dimensions
            let gridSize = gridResolution + 1;
            let gridElementsPerFeature = gridSize * gridSize;
            
            // Calculate grid indices for the four corners
            let idx00 = y0.toInt() * gridSize + x0.toInt();
            let idx01 = y1.toInt() * gridSize + x0.toInt();
            let idx10 = y0.toInt() * gridSize + x1.toInt();
            let idx11 = y1.toInt() * gridSize + x1.toInt();
            
            // Read values from the four corners with feature offset
            let value00 = storage.read(Storage.getOffset(levelAddressOffset, idx00));
            let value01 = storage.read(Storage.getOffset(levelAddressOffset, idx01));
            let value10 = storage.read(Storage.getOffset(levelAddressOffset, idx10));
            let value11 = storage.read(Storage.getOffset(levelAddressOffset, idx11));
            
            // Perform bilinear interpolation
            let lerp0 = lerp(value00, value10, tx);
            let lerp1 = lerp(value01, value11, tx);
            return lerp(lerp0, lerp1, ty);
        }
        else if (Dim == 3)
        {
            static_assert(false, "Not implemented");
        }
        else
        {
            static_assert(false, "Not implemented");
        }
    }

    public OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels>
    {
        OutputVector output;
        [ForceUnroll]
        for (int level = 0; level < Levels; level++)
        {
            [ForceUnroll]
            for (int feature = 0; feature < Features; feature++)
            {
                output[level * Features + feature] = evalLevel<InputVector>(
                    storage,
                    addresses[level],
                    input,
                    level,
                    feature
                );
            }
        }

        return output;
    }
}

///////////////
// Hash grid //
///////////////

public struct HashGridEncoder<T, int Dim, int Levels, int Features, Storage> : ILearnableEncoder<T, Dim, Features * Levels, Storage>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable & IConvertibleToInt
    where Storage : IStorage<T>
{
    Array<Storage.Address, Levels> addresses;
    int gridResolution;
    int hashTableSize;

    [BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels>
    {
        OutputVector output;
        static_assert(false, "Not implemented");
        return output;
    }

    static uint spatialHash<InputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
    {
        static_assert(false, "Not implemented");
    }
}
