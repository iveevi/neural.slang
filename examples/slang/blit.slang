// Upscaled rendering with quad rendering
uniform Texture2D referenceTexture;
uniform SamplerState referenceSampler;
uniform uint2 upscaledResolution;

[shader("vertex")]
float4 upscaled_vertex(float2 position : POSITION) : SV_Position
{
    return float4(position, 0.0, 1.0);
}

[shader("fragment")]
float4 upscaled_fragment(float4 position : SV_Position) : SV_Target
{
    let uv = position.xy / upscaledResolution;
    return referenceTexture.Sample(referenceSampler, uv);
}