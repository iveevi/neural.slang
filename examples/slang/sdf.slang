module sdf;

import camera;

interface ISignedDistanceField
{
    float operator()(float3 p);
}

// Essential primitives
struct Sphere : ISignedDistanceField
{
    float3 center;
    float radius;

    float operator()(float3 p)
    {
        return length(p - center) - radius;
    }
}

struct Box : ISignedDistanceField
{
    float3 low;
    float3 high;

    float operator()(float3 p)
    {
        return max(max(p.x - high.x, low.x - p.x), max(max(p.y - high.y, low.y - p.y), max(p.z - high.z, low.z - p.z)));
    }
}

// Intersection routines
interface ISphereTraceIntersection
{
    static This sphereTrace<SDF>(SDF sdf, Ray ray, int maxIters, float tMin, float tMax)
        where SDF : ISignedDistanceField;
}

struct PositionalSphereTrace : ISphereTraceIntersection
{
    float t;
    float3 p;

    static This sphereTrace<SDF>(SDF sdf, Ray ray, int maxIters, float tMin, float tMax)
        where SDF : ISignedDistanceField
    {
        float t = 0.0;
        for (int i = 0; i < maxIters; i++)
        {
            float3 p = ray.origin + t * ray.direction;
            float d = sdf(p);
            if (d < 0.001 || t > tMax)
                return This(t, p);
            if (t < tMin)
                return This(0.0, float3(0.0));
            t += d;
        }

        return This(0.0, float3(0.0));
    }
}

struct PositionNormalSphereTrace : ISphereTraceIntersection
{
    float t;
    float3 p;
    float3 n;

    static This sphereTrace<SDF>(SDF sdf, Ray ray, int maxIters, float tMin, float tMax)
        where SDF : ISignedDistanceField
    {
        float t = 0.0;
        for (int i = 0; i < maxIters; i++)
        {
            float3 p = ray.origin + t * ray.direction;
            float d = sdf(p);
            if (d < 0.001 || t > tMax)
            {
                // Calculate gradient
                // TODO: use autodiff for this

                float3 px_plus_eps = p + float3(1e-4, 0.0, 0.0);
                float3 px_minus_eps = p - float3(1e-4, 0.0, 0.0);
                float3 py_plus_eps = p + float3(0.0, 1e-4, 0.0);
                float3 py_minus_eps = p - float3(0.0, 1e-4, 0.0);
                float3 pz_plus_eps = p + float3(0.0, 0.0, 1e-4);
                float3 pz_minus_eps = p - float3(0.0, 0.0, 1e-4);

                float dx = sdf(px_plus_eps) - sdf(px_minus_eps);
                float dy = sdf(py_plus_eps) - sdf(py_minus_eps);
                float dz = sdf(pz_plus_eps) - sdf(pz_minus_eps);

                float3 n = normalize(float3(dx, dy, dz));
                
                return This(t, p, n);
            }
            if (t < tMin)
                return This(0.0, float3(0.0), float3(0.0));
            t += d;
        }

        return This(0.0, float3(0.0), float3(0.0));
    }
}