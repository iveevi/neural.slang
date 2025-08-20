implementing neural;

////////////////////////
// Mean squared error //
////////////////////////

public struct MeanSquaredError<T> : ILoss<T>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    [BackwardDifferentiable]
    public static T eval<int N, Vector>(Vector predicted, no_diff Vector expected)
        where Vector : IVector<T, N>
    {
        let diff = predicted.sub(expected);
        let square = diff.mul(diff);
        return square.sum() / T(N);
    }
}

// Shorthand
public typealias MSE<T>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    = MeanSquaredError<T>;

/////////////////////////
// Mean absolute error //
/////////////////////////

public struct MeanAbsoluteError<T> : ILoss<T>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    [BackwardDifferentiable]
    public static T eval<int N, Vector>(Vector predicted, no_diff Vector expected)
        where Vector : IVector<T, N>
    {
        let diff = predicted.sub(expected);
        let abs = diff.abs();
        return abs.sum() / T(N);
    }
}

// Shorthand
public typealias MAE<T>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    = MeanAbsoluteError<T>;

////////////////////////////////////
// Mean absolute percentage error //
////////////////////////////////////

public struct MeanAbsolutePercentageError<T> : ILoss<T>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
{
    [BackwardDifferentiable]
    public static T eval<int N, Vector>(Vector predicted, no_diff Vector expected)
        where Vector : IVector<T, N>
    {
        static const let epsilon = T(1e-8);
        let diff = predicted.sub(expected);
        let abs = diff.abs();
        let abs_expected = expected.abs();
        let percentage = abs.div(abs_expected.add(Vector(epsilon)));
        return percentage.sum() / T(N);
    }
}

// Shorthand
public typealias MAPE<T>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    = MeanAbsolutePercentageError<T>;