implementing neural;

// Vector concatenation
[BackwardDerivative(concatBwd)]
public static OutputVector concat<OutputVector, V1, V2, T, int N1, int N2>(V1 v1, V2 v2)
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where V1 : IVector<T, N1>
    where V2 : IVector<T, N2>
    where OutputVector : IVector<T, N1 + N2>
{
    var output = OutputVector(T(0.0));

    [ForceUnroll]
    for (int i = 0; i < N1; i++)
        output[i] = v1[i];

    [ForceUnroll]
    for (int i = 0; i < N2; i++)
        output[N1 + i] = v2[i];

    return output;
}

static void concatBwd<OutputVector, V1, V2, T, int N1, int N2>(
    inout DifferentialPair<V1> dv1,
    inout DifferentialPair<V2> dv2,
    OutputVector.Differential doutput
)
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where V1 : IVector<T, N1>
    where V2 : IVector<T, N2>
    where OutputVector : IVector<T, N1 + N2>
{
    var d1 = V1(T(0.0));
    var d2 = V2(T(0.0));

    [ForceUnroll]
    for (int i = 0; i < N1; i++)
        d1[i] = __slang_noop_cast<OutputVector>(doutput)[i];
            
    [ForceUnroll]
    for (int i = 0; i < N2; i++)
        d2[i] = __slang_noop_cast<OutputVector>(doutput)[N1 + i];

    dv1 = DifferentialPair<V1>(dv1.p, __slang_noop_cast<V1.Differential>(d1));
    dv2 = DifferentialPair<V2>(dv2.p, __slang_noop_cast<V2.Differential>(d2));
}