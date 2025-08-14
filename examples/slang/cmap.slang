module cmap;

public interface IColorMap
{
    public __init(float vmin, float vmax);
    public float3 eval(float t);
}

float clamp_normalize(float t, float vmin, float vmax)
{
    return clamp((t - vmin) / (vmax - vmin), 0.0, 1.0);
}

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

public struct Plasma : IColorMap
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
        
        if (t <= 0.2)
        {
            float local_t = t / 0.2;
            return lerp(float3(0.050383, 0.029803, 0.527975),
                        float3(0.287076, 0.010834, 0.627065), local_t);
        }
        else if (t <= 0.4)
        {
            float local_t = (t - 0.2) / 0.2;
            return lerp(float3(0.287076, 0.010834, 0.627065),
                        float3(0.534773, 0.088915, 0.563010), local_t);
        }
        else if (t <= 0.6)
        {
            float local_t = (t - 0.4) / 0.2;
            return lerp(float3(0.534773, 0.088915, 0.563010),
                        float3(0.758002, 0.214982, 0.389561), local_t);
        }
        else if (t <= 0.8)
        {
            float local_t = (t - 0.6) / 0.2;
            return lerp(float3(0.758002, 0.214982, 0.389561),
                        float3(0.922421, 0.481158, 0.196348), local_t);
        }
        else
        {
            float local_t = (t - 0.8) / 0.2;
            return lerp(float3(0.922421, 0.481158, 0.196348),
                        float3(0.940015, 0.975158, 0.131326), local_t);
        }
    }
}

public struct Inferno : IColorMap
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
        
        if (t <= 0.25)
        {
            float local_t = t / 0.25;
            return lerp(float3(0.001462, 0.000466, 0.013866),
                        float3(0.251462, 0.051932, 0.232077), local_t);
        }
        else if (t <= 0.5)
        {
            float local_t = (t - 0.25) / 0.25;
            return lerp(float3(0.251462, 0.051932, 0.232077),
                        float3(0.584071, 0.120565, 0.193794), local_t);
        }
        else if (t <= 0.75)
        {
            float local_t = (t - 0.5) / 0.25;
            return lerp(float3(0.584071, 0.120565, 0.193794),
                        float3(0.888699, 0.330987, 0.171103), local_t);
        }
        else
        {
            float local_t = (t - 0.75) / 0.25;
            return lerp(float3(0.888699, 0.330987, 0.171103),
                        float3(0.988362, 0.998364, 0.644924), local_t);
        }
    }
}

public struct Magma : IColorMap
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
        
        if (t <= 0.25)
        {
            float local_t = t / 0.25;
            return lerp(float3(0.001462, 0.000466, 0.013866),
                        float3(0.282623, 0.140926, 0.457517), local_t);
        }
        else if (t <= 0.5)
        {
            float local_t = (t - 0.25) / 0.25;
            return lerp(float3(0.282623, 0.140926, 0.457517),
                        float3(0.621563, 0.204321, 0.569633), local_t);
        }
        else if (t <= 0.75)
        {
            float local_t = (t - 0.5) / 0.25;
            return lerp(float3(0.621563, 0.204321, 0.569633),
                        float3(0.933154, 0.423062, 0.431098), local_t);
        }
        else
        {
            float local_t = (t - 0.75) / 0.25;
            return lerp(float3(0.933154, 0.423062, 0.431098),
                        float3(0.987053, 0.991438, 0.749504), local_t);
        }
    }
}

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
        
        float r = clamp(0.23062 + t * (4.22861 - t * (5.04358 - t * (2.38026 - t * 0.39896))), 0.0, 1.0);
        float g = clamp(0.00030 + t * (3.35943 - t * (4.17441 - t * 1.81831)), 0.0, 1.0);
        float b = clamp(0.75368 + t * (-0.06401 + t * (0.44291 - t * 0.12720)), 0.0, 1.0);
        
        return float3(r, g, b);
    }
}

public struct Coolwarm : IColorMap
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
        
        if (t <= 0.125)
        {
            float local_t = t / 0.125;
            return lerp(float3(0.230, 0.299, 0.754),
                        float3(0.410, 0.467, 0.844), local_t);
        }
        else if (t <= 0.25)
        {
            float local_t = (t - 0.125) / 0.125;
            return lerp(float3(0.410, 0.467, 0.844),
                        float3(0.584, 0.635, 0.915), local_t);
        }
        else if (t <= 0.375)
        {
            float local_t = (t - 0.25) / 0.125;
            return lerp(float3(0.584, 0.635, 0.915),
                        float3(0.752, 0.800, 0.969), local_t);
        }
        else if (t <= 0.5)
        {
            float local_t = (t - 0.375) / 0.125;
            return lerp(float3(0.752, 0.800, 0.969),
                        float3(0.865, 0.865, 0.865), local_t);
        }
        else if (t <= 0.625)
        {
            float local_t = (t - 0.5) / 0.125;
            return lerp(float3(0.865, 0.865, 0.865),
                        float3(0.956, 0.800, 0.752), local_t);
        }
        else if (t <= 0.75)
        {
            float local_t = (t - 0.625) / 0.125;
            return lerp(float3(0.956, 0.800, 0.752),
                        float3(0.915, 0.635, 0.584), local_t);
        }
        else if (t <= 0.875)
        {
            float local_t = (t - 0.75) / 0.125;
            return lerp(float3(0.915, 0.635, 0.584),
                        float3(0.844, 0.467, 0.410), local_t);
        }
        else
        {
            float local_t = (t - 0.875) / 0.125;
            return lerp(float3(0.844, 0.467, 0.410),
                        float3(0.706, 0.016, 0.150), local_t);
        }
    }
}

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