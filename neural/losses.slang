implementing neural;

////////////////////////
// Mean squared error //
////////////////////////

public struct MeanSquaredError<T> : ILoss<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    [NoDiffThis, BackwardDifferentiable]
    public T eval<int N, Vector>(Vector predicted, no_diff Vector expected)
        where Vector : IVector<T, N>
        where Vector.Differential == Vector
    {
        let diff = predicted.sub(expected);
        let square = diff.mul(diff);
        return square.sum() / T(N);
    }
}
