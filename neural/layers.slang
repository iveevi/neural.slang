implementing neural;

//////////////////
// Feed forward //
//////////////////

// Ordinary storage
public struct FeedForward<T, int In, int Out, Storage, Activation>
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
        let output = input.apply<Out, Storage, OutputVector>(storage, parameters);
        return activation.eval<Out, OutputVector>(output);
    }
}