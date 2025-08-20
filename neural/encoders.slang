implementing neural;

//////////////////////
// Identity encoder //
//////////////////////

public struct IdentityEncoder<T, int N> : IEncoder<T, N, N>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    public static const int In = N;
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
    public static const int In = Dim;
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

//////////////////////
// One-blob encoder //
//////////////////////

// Gaussian kernel
public struct OneBlobEncoder<T, int N, int K> : IEncoder<T, N, N * K>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    public static const int In = N;
    public static const int Out = N * K;

    [BackwardDerivative(evalBwd)]
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, N>
        where OutputVector : IVector<T, N * K>
    {
        OutputVector output;

        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            let s = input[i];
            [ForceUnroll]
            for (int j = 0; j < K; j++)
            {
                let x = T(j) / T(K - 1);
                let diff = s - x;
                let k = exp(-diff * diff / (T(K) * T(K)));
                output[i * K + j] = k;
            }
        }

        return output;
    }

    void evalBwd<InputVector, OutputVector>(inout DifferentialPair<InputVector> dinput, OutputVector.Differential doutput)
        where InputVector : IVector<T, N>
        where OutputVector : IVector<T, N * K>
    {
        let p = __slang_noop_cast<InputVector>(dinput.p);
        var d = __slang_noop_cast<InputVector>(dinput.d);
        let dout = __slang_noop_cast<OutputVector>(doutput);

        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            let s = p[i];
            [ForceUnroll]
            for (int j = 0; j < K; j++)
            {
                let x = T(j) / T(K - 1);
                let diff = s - x;
                let k = exp(-diff * diff / (T(K) * T(K)));
                let dk = dout[i * K + j];
                d[i] += dk * k * (-T(2.0) * diff) / (T(K) * T(K));
            }
        }

        dinput = DifferentialPair<InputVector>(p, __slang_noop_cast<InputVector.Differential>(d));
    }
}