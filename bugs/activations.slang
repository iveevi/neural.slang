module activations;

// public interface IActivation
// {
//     float eval(float x);
// }

// public struct LeakyReLU : IActivation
// {
//     float alpha;

//     public __init(float alpha)
//     {
//         this.alpha = alpha;
//     }

//     public float eval(float x)
//     {
//         return max(x, alpha * x);
//     }
// }

public interface IActivation<T>
    where T : __BuiltinFloatingPointType
{
    T eval(T x);
}

public struct LeakyReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
{
    T alpha;

    public __init(T alpha)
    {
        this.alpha = alpha;
    }

    public T eval(T x)
    {
        return max(x, alpha * x);
    }
}