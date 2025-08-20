implementing neural;

/////////////////////////
// Normal feed forward //
/////////////////////////

public struct FFLayer<T, int In, int Out, Storage, Activation>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Storage : IStorage<T>
    where Storage.Differential == Storage
    where Activation : IActivation<T>
{
    internal Storage.Address parameters;
    internal no_diff Activation activation;

    public __init(Storage.Address parameters, Activation activation)
    {
        this.parameters = parameters;
        this.activation = activation;
    }

    [NoDiffThis, BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>
        where OutputVector.Differential == OutputVector
    {
        let output = input.apply<Out, true, Storage, OutputVector>(storage, parameters);
        return activation.eval<Out, OutputVector>(output);
    }
}

///////////////////////////
// Residual feed forward //
///////////////////////////

public struct ResidualFFLayer<T, int Size, Storage, Activation>
    where T : __BuiltinFloatingPointType & IArithmeticAtomicable
    where Storage : IStorage<T>
    where Storage.Differential == Storage
    where Activation : IActivation<T>
{
    internal Storage.Address parameters;
    internal no_diff Activation activation;

    public __init(Storage.Address parameters, Activation activation)
    {
        this.parameters = parameters;
        this.activation = activation;
    }

    [NoDiffThis, BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(Storage storage, InputVector input)
        where InputVector : IVector<T, Size>
        where OutputVector : IVector<T, Size>
        where OutputVector.Differential == OutputVector
    {
        static_assert(InputVector is OutputVector, "Input and output vectors must be the same");
        let output = input.apply<Size, true, Storage, OutputVector>(storage, parameters);
        let outputCasted = __slang_noop_cast<OutputVector>(output);
        return activation.eval<Size, OutputVector>(output).add(outputCasted);
    }
}