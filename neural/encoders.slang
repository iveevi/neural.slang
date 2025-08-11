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

struct SphericalHarmonicsEncoder<T, int Levels> : IEncoder<T, 3, 2 * Levels + 1>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
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

///////////////
// Hash grid //
///////////////

struct HashGridEncoder<T, int Dim, int Levels, int Features, Storage> : ILearnableEncoder<T, Dim, Features * Levels, Storage>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
    where Storage : IStorage<T>
{
    Storage.Address address;
    int resolution;
    int size;

    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels>
    {
        OutputVector output;
        static_assert(false, "Not implemented");
        return output;
    }
}

// Overload for bindless storage
extension<T, int Dim, int Levels, int Features, Storage> HashGridEncoder<T, Dim, Levels, Features, Storage>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
    where Storage : IPointerLikeStorage<T>
{
    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, Dim>
        where OutputVector : IVector<T, Features * Levels>
    {
        OutputVector output;
        static_assert(false, "Not implemented");
        return output;
    }
}
