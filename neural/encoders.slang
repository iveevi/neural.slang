implementing neural;

//////////////////////
// Identity encoder //
//////////////////////

public struct IdentityEncoder<T, int N> : IEncoder<T, N, N>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    public static const int Out = N;
    
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
    public static const int Out = 2 * Levels * Dim;

    [BackwardDerivative(evalBwd)]
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Out>
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
        where OutputVector : IVector<T, Out>
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
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Storage : IBindlessStorage<T>
{
    public typealias Differential = This;

    public static const int Out = Features * Levels;
    
    Array<Storage, Levels> addresses;
    Array<int, Levels> gridResolutions;

    public __init(Array<Storage, Levels> addresses, Array<int, Levels> gridResolutions)
    {
        this.addresses = addresses;
        this.gridResolutions = gridResolutions;
    }

    // Dimension 1
    [BackwardDifferentiable]
    public void evalLevelLinear<InputVector, OutputVector>(InputVector input, int level, out OutputVector output)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Out>
    {
        let address = addresses[level];
        let resolution = gridResolutions[level];
        
        let scaledInput = input[0] * T(resolution - 1);
        let i0 = clamp(floor(scaledInput).toInt(), 0, resolution - 1);
        let i1 = clamp(ceil(scaledInput).toInt(), 0, resolution - 1);

        let t = scaledInput - T(i0);

        [ForceUnroll]
        for (int feature = 0; feature < Features; feature++)
        {
            let value0 = address.getOffset(i0 * Features + feature).read();
            let value1 = address.getOffset(i1 * Features + feature).read();
            output[Features * level + feature] = lerp(value0, value1, t);
        }
    }

    // Dimension 2
    [BackwardDifferentiable]
    public void evalLevelBilinear<InputVector, OutputVector>(InputVector input, int level, out OutputVector output)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Out>
    {
        let address = addresses[level];
        let resolution = gridResolutions[level];

        let scaledInput0 = input[0] * T(resolution - 1);
        let scaledInput1 = input[1] * T(resolution - 1);

        let i0 = clamp(floor(scaledInput0).toInt(), 0, resolution - 1);
        let i1 = clamp(ceil(scaledInput0).toInt(), 0, resolution - 1);

        let j0 = clamp(floor(scaledInput1).toInt(), 0, resolution - 1);
        let j1 = clamp(ceil(scaledInput1).toInt(), 0, resolution - 1);

        let tX = scaledInput0 - T(i0);
        let tY = scaledInput1 - T(j0);

        [ForceUnroll]
        for (int feature = 0; feature < Features; feature++)
        {
            let value00 = address.getOffset(i0 * Features + feature).read();
            let value01 = address.getOffset(i0 * Features + feature + 1).read();
            let value10 = address.getOffset(i1 * Features + feature).read();
            let value11 = address.getOffset(i1 * Features + feature + 1).read();

            output[Features * level + feature] = lerp(
                lerp(value00, value01, tX),
                lerp(value10, value11, tX),
                tY
            );
        }
    }

    [BackwardDifferentiable]
    public void evalLevel<InputVector, OutputVector>(InputVector input, int level, out OutputVector output)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Out>
    {
        if (Dim == 1)
            evalLevelLinear<InputVector, OutputVector>(input, level, output);
        else if (Dim == 2)
            evalLevelBilinear<InputVector, OutputVector>(input, level, output);
        else
            static_assert(false, "Not implemented");
    }

    [BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Out>
    {
        OutputVector output;
        [ForceUnroll]
        for (int level = 0; level < Levels; level++)
            evalLevel<InputVector, OutputVector>(input, level, output);
        return output;
    }
}

///////////////
// Hash grid //
///////////////

public struct HashGridEncoder<T, int Dim, int Levels, int Features, Storage>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Storage : IBindlessStorage<T>
{
}
