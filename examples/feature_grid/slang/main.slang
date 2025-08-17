import feature_grid;
import neural;

extern static const int Dimension;
extern static const int Features;

typealias InputVector = InlineVector<float, Dimension>;
typealias FeatureVector = InlineVector<float, Features>;

typealias FeatureGridType = FeatureGrid<Dimension, Features>;
typealias Optimizer = Adam<float>;

uniform FeatureGridType featureGrid;
uniform Optimizer optimizer;
uniform RWStructuredBuffer<Optimizer.State> optimizerStates;

[BackwardDifferentiable]
FeatureVector forward(FeatureGridType.Parameters parameters, no_diff FeatureGridType.Configuration configuration, InputVector input)
{
    return FeatureGridType.sample(parameters, configuration, input);
}

[BackwardDifferentiable]
float sample_loss(
    FeatureGridType.Parameters parameters,
    no_diff FeatureGridType.Configuration configuration,
    no_diff InputVector input,
    no_diff FeatureVector target)
{
    return MeanSquaredError<float>().eval(forward(parameters, configuration, input), target);
}

uniform RWStructuredBuffer<InputVector> inputBuffer;
uniform RWStructuredBuffer<FeatureVector> targetBuffer;
uniform RWStructuredBuffer<FeatureVector> outputBuffer;
uniform RWStructuredBuffer<float> lossBuffer;
uniform uint sampleCount;

[shader("compute")]
[numthreads(32, 1, 1)]
void forward_pass(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= sampleCount)
        return;
        
    let input = inputBuffer[tid.x];
    let result = forward(featureGrid.parameters, featureGrid.configuration, input);
    outputBuffer[tid.x] = result;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void backward_pass(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= sampleCount)
        return;
        
    let input = inputBuffer[tid.x];
    let target = targetBuffer[tid.x];
    
    let current_loss = sample_loss(featureGrid.parameters, featureGrid.configuration, input, target);
    lossBuffer[tid.x] = current_loss;
    

    bwd_diff(sample_loss)(
        featureGrid.dual,
        featureGrid.configuration,
        input,
        target,
        1.0 / float(sampleCount)
    );
}

[shader("compute")]
[numthreads(32, 1, 1)]
void update_parameters(uint3 tid : SV_DispatchThreadID)
{
    featureGrid.update(optimizer, optimizerStates[tid.x], tid.x);
} 