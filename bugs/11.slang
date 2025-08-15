interface IVector<T, int Dim> {}

struct MyVector<T, int Dim> : IVector<T, Dim>
{
    T[Dim] data;

    __init(T value)
    {
        [ForceUnroll]
        for (int i = 0; i < Dim; i++)
            data[i] = value;
    }
}

interface IMLP<T, int In, int Out>
{
    associatedtype InputVec : IVector<T, In>;
    associatedtype OutputVec : IVector<T, Out>;
    
    associatedtype Parameters : IDifferentiablePtrType;
    associatedtype Configuration;

    [BackwardDifferentiable]
    static OutputVec eval(Parameters parameters, Configuration configuration, InputVec input);
}

struct MyMLP<T, int In, int Out> : IMLP<T, In, Out>
    where T : __BuiltinFloatingPointType
{
    typealias InputVec = MyVector<T, In>;
    typealias OutputVec = MyVector<T, Out>;

    struct MyParameters : IDifferentiablePtrType
    {
        typealias Differential = This;
    }

    typealias Parameters = MyParameters;
    typealias Configuration = MyVector<T, In>;

    [BackwardDifferentiable]
    static OutputVec eval(Parameters parameters, Configuration configuration, InputVec input)
    {
        return OutputVec(T(0.0));
    }
}