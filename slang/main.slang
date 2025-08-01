import neural;

// Test basic activation functions
struct GlobalParameters
{
    RWStructuredBuffer<float> input;
    RWStructuredBuffer<float> output;
}

ParameterBlock<GlobalParameters> globals;

[shader("compute")]
void main()
{
    var activation = ReLU<float>();
    var input = InlineVector<float, 1>(globals.input[0]);
    var output = activation.eval(input);
    globals.output[0] = output[0];
}