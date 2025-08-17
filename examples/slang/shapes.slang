module shapes;

import camera;

public struct Sphere
{
    public float3 center;
    public float radius;
}

public struct Box
{
    public float3 low;
    public float3 high;

    [NoDiffThis, Differentiable]
    public float3 normalized(float3 point)
    {
        return (point - low) / (high - low);
    }

    public bool contains(float3 point)
    {
        return all(point >= low && point <= high);
    }

    public Optional<float> intersection(Ray ray)
    {
        float3 dir = ray.direction;
        float3 invDir = 1.0 / dir;
        
        float tMin = (low.x - ray.origin.x) * invDir.x;
        float tMax = (high.x - ray.origin.x) * invDir.x;
        
        if (tMin > tMax)
        {
            float temp = tMin;
            tMin = tMax;
            tMax = temp;
        }
        
        float tyMin = (low.y - ray.origin.y) * invDir.y;
        float tyMax = (high.y - ray.origin.y) * invDir.y;
        
        if (tyMin > tyMax)
        {
            float temp = tyMin;
            tyMin = tyMax;
            tyMax = temp;
        }
        
        if (tMin > tyMax || tyMin > tMax)
            return Optional<float>();
            
        tMin = max(tMin, tyMin);
        tMax = min(tMax, tyMax);
        
        float tzMin = (low.z - ray.origin.z) * invDir.z;
        float tzMax = (high.z - ray.origin.z) * invDir.z;
        
        if (tzMin > tzMax)
        {
            float temp = tzMin;
            tzMin = tzMax;
            tzMax = temp;
        }
        
        if (tMin > tzMax || tzMin > tMax)
            return Optional<float>();
            
        tMin = max(tMin, tzMin);
        tMax = min(tMax, tzMax);
        
        if (tMax < 0.0)
            return Optional<float>();
            
        return Optional<float>(tMin >= 0.0 ? tMin : tMax);
    }

    public static Box fromCenterAndSize(float3 center, float3 size)
    {
        return Box(center - size * 0.5, center + size * 0.5);
    }
}

public struct Cylinder
{
    public float3 center;
    public float radius;
    public float height;

    public Optional<float3> intersection(Ray ray)
    {
        float3 relativeOrigin = ray.origin - center;
        float3 dir = ray.direction;
        
        float a = dir.x * dir.x + dir.z * dir.z;
        float b = 2.0 * (relativeOrigin.x * dir.x + relativeOrigin.z * dir.z);
        float c = relativeOrigin.x * relativeOrigin.x + relativeOrigin.z * relativeOrigin.z - radius * radius;
        
        float discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0.0 || a == 0.0)
            return {};
        
        float sqrtDiscriminant = sqrt(discriminant);
        float t1 = (-b - sqrtDiscriminant) / (2.0 * a);
        float t2 = (-b + sqrtDiscriminant) / (2.0 * a);
        
        float t = (t1 > 0.0) ? t1 : t2;
        if (t <= 0.0)
            return {};
        
        float3 hitPoint = ray.origin + t * dir;
        float3 relativeHit = hitPoint - center;

        if (abs(relativeHit.y) > height * 0.5)
            return {};
        
        return hitPoint;
    }

    public float3 toCylindrical(float3 point)
    {
        float3 relative = point - center;
        float r = length(relative.xz) / radius;
        float theta = atan2(relative.z, relative.x);
        float y = (relative.y + height * 0.5) / height;
        return float3(r, theta, y);
    }
}

public struct Triangle
{
    public float3 v0;
    public float3 v1;
    public float3 v2;
}