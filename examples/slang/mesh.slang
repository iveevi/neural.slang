module mesh;

public struct TriangleMesh
{
    public RWStructuredBuffer<float3>.Handle vertices;
    public RWStructuredBuffer<uint3>.Handle triangles;
};
