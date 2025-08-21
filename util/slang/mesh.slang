module mesh;

public struct TriangleMesh
{
    // TODO: make this bindless
    public RWStructuredBuffer<float3, ScalarDataLayout> vertices;
    public RWStructuredBuffer<uint3, ScalarDataLayout> triangles;
    public uint triangleCount;
};
