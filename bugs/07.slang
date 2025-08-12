interface IVector<T, int N> {}

interface IEncoder<T, int In>
{
	static const int In;
	static const int Out;

    [BackwardDifferentiable]
    OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>;
}
