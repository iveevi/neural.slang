implementing neural;

//////////
// Adam //
//////////

public struct Adam<T> : IOptimizer<T>
    where T : __BuiltinFloatingPointType
{
    public struct AdamState {
        T m;
        T v;
        int t;
    };

    public typealias State = AdamState;

    T alpha;
    T beta1;
    T beta2;
    T epsilon;

    public __init()
    {
        this.alpha = T(0.001);
        this.beta1 = T(0.9);
        this.beta2 = T(0.999);
        this.epsilon = T(1e-8);
    }

    public __init(T alpha, T beta1, T beta2, T epsilon)
    {
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public void step(inout State state, inout T parameter, T gradient)
    {
        state.t += 1;
        state.m = beta1 * state.m + (T(1.0) - beta1) * gradient;
        state.v = beta2 * state.v + (T(1.0) - beta2) * gradient * gradient;

        T m_hat = state.m / (T(1.0) - pow(beta1, T(state.t)));
        T v_hat = state.v / (T(1.0) - pow(beta2, T(state.t)));

        parameter -= alpha * m_hat / (sqrt(v_hat) + epsilon);
    }
}

// Overload for storages
public extension<T> Adam<T> : IOptimizer<T>
    where T : __BuiltinFloatingPointType
    where T : IArithmeticAtomicable
{
    // public void step<int N>(inout Array<State, N> state, inout Array<T, N> parameter, Array<T, N> gradient)
    // {
    //     for (int i = 0; i < N; i++)
    //         step(state[i], parameter[i], gradient[i]);
    // }
}
