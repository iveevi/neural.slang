module cmap;

// Matplotlib colormaps
public interface IColorMap
{
    public __init(float vmin, float vmax);

    public float3 eval(float t);
}

// Helper function to clamp and normalize input
float clamp_normalize(float t, float vmin, float vmax)
{
    return clamp((t - vmin) / (vmax - vmin), 0.0, 1.0);
}

// Viridis colormap - perceptually uniform purple to yellow
public struct Viridis : IColorMap
{
    float vmin = 0.0;
    float vmax = 1.0;

    public __init(float vmin, float vmax)
    {
        this.vmin = vmin;
        this.vmax = vmax;
    }

    public float3 eval(float t)
    {
        t = clamp_normalize(t, vmin, vmax);
        
        // Viridis color control points (RGB values)
        if (t <= 0.25)
        {
            float local_t = t / 0.25;
            return lerp(float3(0.267004, 0.004874, 0.329415), 
                        float3(0.229739, 0.322361, 0.545706), local_t);
        }
        else if (t <= 0.5)
        {
            float local_t = (t - 0.25) / 0.25;
            return lerp(float3(0.229739, 0.322361, 0.545706),
                        float3(0.127568, 0.566949, 0.550556), local_t);
        }
        else if (t <= 0.75)
        {
            float local_t = (t - 0.5) / 0.25;
            return lerp(float3(0.127568, 0.566949, 0.550556),
                        float3(0.369214, 0.788888, 0.382914), local_t);
        }
        else
        {
            float local_t = (t - 0.75) / 0.25;
            return lerp(float3(0.369214, 0.788888, 0.382914),
                        float3(0.993248, 0.906157, 0.143936), local_t);
        }
    }
}

// Turbo colormap - improved version of jet with better perceptual properties
public struct Turbo : IColorMap
{
    float vmin = 0.0;
    float vmax = 1.0;

    public __init(float vmin, float vmax)
    {
        this.vmin = vmin;
        this.vmax = vmax;
    }

    public float3 eval(float t)
    {
        t = clamp_normalize(t, vmin, vmax);
        
        // Turbo colormap approximation using polynomial fit
        float r = clamp(34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05)))), 0.0, 1.0);
        float g = clamp(23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56)))), 0.0, 1.0);
        float b = clamp(27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66)))), 0.0, 1.0);
        
        return float3(r, g, b);
    }
}

// RdBu colormap - diverging red to blue through white
public struct RdBu : IColorMap
{
    float vmin = 0.0;
    float vmax = 1.0;

    public __init(float vmin, float vmax)
    {
        this.vmin = vmin;
        this.vmax = vmax;
    }

    public float3 eval(float t)
    {
        t = clamp_normalize(t, vmin, vmax);
        
        // RdBu color control points
        if (t <= 0.125)
        {
            float local_t = t / 0.125;
            return lerp(float3(0.403922, 0.000000, 0.121569),
                        float3(0.698039, 0.094118, 0.168627), local_t);
        }
        else if (t <= 0.25)
        {
            float local_t = (t - 0.125) / 0.125;
            return lerp(float3(0.698039, 0.094118, 0.168627),
                        float3(0.839216, 0.376471, 0.301961), local_t);
        }
        else if (t <= 0.375)
        {
            float local_t = (t - 0.25) / 0.125;
            return lerp(float3(0.839216, 0.376471, 0.301961),
                        float3(0.956863, 0.647059, 0.509804), local_t);
        }
        else if (t <= 0.5)
        {
            float local_t = (t - 0.375) / 0.125;
            return lerp(float3(0.956863, 0.647059, 0.509804),
                        float3(0.992157, 0.858824, 0.780392), local_t);
        }
        else if (t <= 0.625)
        {
            float local_t = (t - 0.5) / 0.125;
            return lerp(float3(0.992157, 0.858824, 0.780392),
                        float3(0.819608, 0.898039, 0.941176), local_t);
        }
        else if (t <= 0.75)
        {
            float local_t = (t - 0.625) / 0.125;
            return lerp(float3(0.819608, 0.898039, 0.941176),
                        float3(0.572549, 0.772549, 0.870588), local_t);
        }
        else if (t <= 0.875)
        {
            float local_t = (t - 0.75) / 0.125;
            return lerp(float3(0.572549, 0.772549, 0.870588),
                        float3(0.262745, 0.576471, 0.764706), local_t);
        }
        else
        {
            float local_t = (t - 0.875) / 0.125;
            return lerp(float3(0.262745, 0.576471, 0.764706),
                        float3(0.129412, 0.400000, 0.674510), local_t);
        }
    }
}