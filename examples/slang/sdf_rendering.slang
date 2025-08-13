import camera;
import sdf;
import mesh;
import shapes;

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
    uv.y = 1 - uv.y;

    Ray ray = rayFrame.rayAt(uv);

    // Sphere sphere = Sphere(float3(0, 0, 0), 1);
    // let intx = PositionalSphereTrace.sphereTrace(sphere, ray, 100, 0.001, 10.0);
    
    // Box box = Box(float3(0, 0, 0), float3(1, 1, 1));
    // let intx = PositionalSphereTrace.sphereTrace(box, ray, 100, 0.001, 10.0);
   
    // Triangle triangle = Triangle(float3(0, 0, 0), float3(1, 0, 0), float3(0, 1, 0));
    // let intx = PositionalSphereTrace.sphereTrace(triangle, ray, 100, 0.001, 10.0);

    TriangleMesh triangleMesh = mesh;
    let intx = PositionalSphereTrace.sphereTrace(triangleMesh, ray, 100, 0.001, 10.0);
    
    // Check if we hit the sphere (t > 0 indicates a valid intersection)
    if (intx.t > 0.0)
    {
        targetTexture.Store(tid.xy, float4(intx.p * 0.5 + 0.5, 1));
    }
    else
    {
        // Background color (black)
        targetTexture.Store(tid.xy, float4(0, 0, 0, 1));
    }
}