RWStructuredBuffer<float> input;
RWStructuredBuffer<float> output;

[shader("compute")]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint dst = int(sin(tid.x) * 1000) % 16;
    InterlockedAdd(output[dst], input[tid.x]);
}