import camera;
import sdf;
import mesh;

import network_with_addresses;

// TODO: environment map

// Rendering the reference mesh
ParameterBlock<TriangleMesh> mesh;

uniform RayFrame rayFrame;
uniform RWTexture2D targetTexture;
uniform uint2 targetResolution;

[shader("compute")]
void reference(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = (tid.xy + 0.5) / targetResolution;
    Ray ray = rayFrame(uv.x, uv.y);

    targetTexture.Store(tid.xy, float4(ray.direction * 0.5 + 0.5, 1));
}