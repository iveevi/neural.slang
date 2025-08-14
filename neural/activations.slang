implementing neural;

//////////////
// Identity //
//////////////

public struct Identity<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        return input;
    }
}

//////////
// ReLU //
//////////

public struct ReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDerivative(evalBwd)]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        return input.max(T(0.0));
    }

    void evalBwd<int N, Vector>(inout DifferentialPair<Vector> dinput, Vector.Differential doutput)
        where Vector : IVector<T, N>
        where Vector.Differential  == Vector
    {
        let d = dinput.p.step(T(0.0)).mul(doutput).add(__slang_noop_cast<Vector>(dinput.d));
        dinput = DifferentialPair<Vector>(dinput.p, d);
    }
}

////////////////
// Leaky ReLU //
////////////////

public struct LeakyReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    T alpha;

    public __init()
    {
        this.alpha = T(0.01);
    }

    public __init(T alpha)
    {
        this.alpha = alpha;
    }

    [NoDiffThis, BackwardDerivative(evalBwd)]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        return input.max(input.mul(Vector(alpha)));
    }

    void evalBwd<int N, Vector>(inout DifferentialPair<Vector> dinput, Vector.Differential doutput)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        let positive = dinput.p.step(T(0.0));
        let negative = dinput.p.neg().step(T(0.0)).mul(Vector(alpha));
        let d = positive.mul(doutput).add(negative.mul(doutput));
        dinput = DifferentialPair<Vector>(dinput.p, d);
    }
}

//////////
// Sine //
//////////

public struct Sine<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        return input.sin();
    }
}

/////////
// Exp //
/////////

public struct Exp<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public Vector eval<int N, Vector>(Vector input)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        return input.exp();
    }
}