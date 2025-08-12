implementing neural;

//////////////////
// Feed forward //
//////////////////

// Ordinary storage
public struct FeedForward<T, int In, int Out, Storage, Activation>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
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

// Overload for bindless storage
public struct BindlessFeedForward<T, int In, int Out, Storage, Activation>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
    where Storage : IBindlessStorage<T>
    where Storage.Differential == Storage
    where Activation : IActivation<T>
{
    internal Storage weights;
    internal Storage biases;
    internal no_diff Activation activation;

    public __init(Storage weights, Storage biases, Activation activation)
    {
        this.weights = weights;
        this.biases = biases;
        this.activation = activation;
    }

    [NoDiffThis, BackwardDifferentiable]
    public OutputVector eval<InputVector, OutputVector>(InputVector input)
        where InputVector : IVector<T, In>
        where OutputVector : IVector<T, Out>
        where OutputVector.Differential == OutputVector
    {
        let output = input.applyBindless<Out, Storage, OutputVector>(weights, biases);
        return activation.eval<Out, OutputVector>(output);
    }
}