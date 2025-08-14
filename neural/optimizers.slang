implementing neural;

/////////
// SGD //
/////////

public struct SGD<T> : IOptimizer<T>
    where T : __BuiltinFloatingPointType
{
    // Optional momentum
    public struct SGDState {
        T velocity;
    };

    public typealias State = SGDState;

    T alpha;
    T beta;

    public __init()
    {
        this.alpha = T(0.001);
        this.beta = T(0.0); // No momentum by default
    }

    public __init(T alpha, T beta)
    {
        this.alpha = alpha;
        this.beta = beta;
    }

    public void step(inout State state, inout T parameter, T gradient)
    {
        state.velocity = beta * state.velocity + alpha * gradient;
        parameter -= state.velocity;
    }
}

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
        this.alpha = T(1e-3);
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

// TODO: overload for storages