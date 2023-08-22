#include <metal_stdlib>

using namespace metal;

struct Uniforms {
    uint particle_count;
    uint time_delta_ms;
    uint total_time_ms;
};

struct Rng {
    uint state;
};

uint init_rng(uint seed) {
  uint x = seed + 1;
  x += x << 10;
  x ^= x >> 6;
  x += x << 3;
  x ^= x >> 11;
  x += x << 15;
  return x;
}

uint rand(thread Rng& rng) {
    uint val = rng.state;
    val ^= val << 13;
    val ^= val >> 17;
    val ^= val << 5;
    rng.state = val;
    return val;
}

float rand_float(thread Rng& rng) {
    return as_type<float>(0x3f800000 | (rand(rng) >> 9)) - 1.;
}

struct VSInstance {
    float2 position [[attribute(0)]];
    float2 velocity [[attribute(1)]];
};

struct VSOut {
    float4 pos [[position]];
    float2 ipos;
    float2 vert;
};

constant float2 triangle[3] = {
    { 0.5, -0.5},
    { 0.0,  0.5},
    {-0.5, -0.5}
};

// The `uniforms` buffer is implicitly assigned to index 1 since the vertex attribute buffers
// come before pipeline buffers in the IndexedByOrder scheme.
vertex VSOut vs_main(VSInstance         instance [[stage_in]],
                     uint               vid      [[vertex_id]],
                     constant Uniforms& uniforms [[buffer(1)]]) {
    VSOut out;
    float2 dir = normalize(instance.velocity);
    float2x2 m = float2x2(dir.x, -dir.y,
                          dir.y,  dir.x);
    float2 p = 0.01 * m * triangle[vid] + 0.1 * instance.position;
    out.pos = float4(p, 0.0, 1.0);
    out.ipos = 0.1 * instance.position;
    out.vert = triangle[vid] * 2.0;//3. *  sin(float(uniforms.total_time_ms) * 0.001);
    return out;
}
fragment float4 fs_main(VSOut input [[stage_in]],
                        constant Uniforms& uniforms [[buffer(1)]]) {
    float l = 4.0 * length(input.vert);
    float d = length(input.ipos.xy);
    return float4(1.3 * d, 1. - d * d, 1. - d,
                  0.2025 + 0.0625 * sin(float(uniforms.total_time_ms) * 0.003)) / l;
}

struct Particle {
    float2 pos;
    float2 velocity;
};

kernel void init_particles(constant Uniforms& uniforms  [[buffer(0)]],
                           device Particle*   particles [[buffer(1)]],
                           uint3              global_id [[thread_position_in_grid]]) {
    uint idx = global_id.x;
    Rng rng{init_rng(idx)};
    if (idx < uniforms.particle_count) {
        float theta = rand_float(rng) * 6.283185;
        float r = sqrt(rand_float(rng)) * 10.;
        float3 p = float3(r * cos(theta), r * sin(theta), 0.);
        float3 angular = cross(p, float3(0., 0., 1.)) * 0.2;
        particles[idx] = Particle {
            p.xy,
            angular.xy + (float2(rand_float(rng), rand_float(rng)) * 2. - float2(1.)) * 2. * rand_float(rng),
        };
    }   
}

kernel void simulate(constant Uniforms& uniforms       [[buffer(0)]],
                     device Particle*   prev_particles [[buffer(1)]],
                     device Particle*   next_particles [[buffer(2)]],
                     device float2*     centers        [[buffer(3)]],
                     uint3              global_id      [[thread_position_in_grid]]) {
    uint idx = global_id.x;
    float dt = float(uniforms.time_delta_ms) * 0.0008;
    if (idx < uniforms.particle_count) {
        float2 p0 = prev_particles[idx].pos;
        float2 v0 = prev_particles[idx].velocity;
        float2 a = float2(0);
        for (int i = 0; i < 3; i++) {
          a += 0.5 * normalize(centers[i] - p0);
        }
        next_particles[idx] = Particle {
            p0 + v0 * dt + a * dt * dt * 0.5,
            v0 + a * dt,
        };
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx == 0) {
        float t = float(uniforms.total_time_ms) * 0.000001;
        float c = cos(t);
        float s = sin(t);
        float2x2 m(c, -s, s, c);
        for (int i = 0; i < 3; i++) {
          centers[i] = m * centers[i];
        }
    }
}
