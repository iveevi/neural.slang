// import neural;
// import camera;
// import network_with_addresses;

// struct Payload
// {
//     float3 color;
// };

// // Rendering the reference (primary with albedo)
// uniform RayFrame rayFrame;
// uniform RWTexture2D targetTexture;
// uniform uint2 targetResolution;

[shader("raygeneration")]
// [numthreads(32, 1, 1)]
void render_reference()
{
    // uint2 index = DispatchRaysIndex().xy;
    // uint2 size = DispatchRaysDimensions().xy;

    // float2 uv = (index.xy + 0.5) / targetResolution;

    // let ray = rayFrame.rayAt(uv);

    // targetTexture[index.xy] = float4(ray.direction * 0.5 + 0.5, 1.0);
}