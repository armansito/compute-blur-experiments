struct VSInstance {
  @location(0) pos: vec2<f32>,
  @location(1) velocity: vec2<f32>,
}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) ipos: vec2<f32>,
  @location(1) @interpolate(linear, center) vert: vec2<f32>,
}

alias TriangleVertices = array<vec2<f32>, 3>;
var<private> triangle: TriangleVertices = TriangleVertices(
  vec2<f32>( 0.5, -0.5),
  vec2<f32>( 0.0,  0.5),
  vec2<f32>(-0.5, -0.5)
);

struct Uniforms {
  particle_count: u32,
  time_delta_ms: u32,
  total_time_ms: u32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(instance: VSInstance, @builtin(vertex_index) vid: u32) -> VSOut {
  let dir = normalize(instance.velocity);
  let m = mat2x2f(dir.x, -dir.y,
                  dir.y,  dir.x);
  let p = 0.01 * m * triangle[vid] + 0.1 * instance.pos;
  return VSOut(
      /*pos=*/vec4f(p, 0., 1.),
      /*ipos=*/0.1 * instance.pos,
      /*vert=*/triangle[vid] * 2.0
  );
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
  let l = 4. * length(input.vert);
  let d = length(input.ipos.xy);
  return vec4f(1.3 * d, 1. - d * d, 1. - d,
               0.2025 + 0.0625 * sin(f32(uniforms.total_time_ms) * 0.003)) / l;
}

struct Particle {
  pos: vec2<f32>,
  velocity: vec2<f32>,
}

@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

struct Rng {
  state: u32,
};
var<private> rng: Rng;

fn init_rng(rng: ptr<private, Rng>, seed: u32) {
  var x: u32 = seed + 1u;
  x += x << 10u;
  x ^= x >> 6u;
  x += x << 3u;
  x ^= x >> 11u;
  x += x << 15u;
  (*rng).state = x;
}

// The xorshift rng from Ray Tracing Gems II, Section 14.3.4
fn rand(rng: ptr<private, Rng>) -> u32 {
  var val = (*rng).state;
  val ^= val << 13u;
  val ^= val >> 17u;
  val ^= val << 5u;
  (*rng).state = val;
  return val;
}

// Returns a random float in the range [0...1]. This uses the 23 most significant bits as the
// mantissa and sets the sign bits and exponent to zero, yielding a random float in the [1, 2]
// range, which is then mapped to [0, 1] via subtraction. Borrowed from Ray Tracing Gems II,
// Section 14.3.4.
fn rand_f32(rng: ptr<private, Rng>) -> f32 {
  return bitcast<f32>(0x3f800000u | (rand(rng) >> 9u)) - 1.;
}

@compute @workgroup_size(256)
fn init_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  init_rng(&rng, idx);
  if idx < uniforms.particle_count {
    let theta = rand_f32(&rng) * 6.283185;
    let r = sqrt(rand_f32(&rng)) * 10.;
    let p = vec3f(r * cos(theta), r * sin(theta), 0.);
    let angular = cross(p, vec3f(0., 0., 1.)) * 0.2;
    particles[idx] = Particle(
        p.xy,
        angular.xy + (vec2f(rand_f32(&rng), rand_f32(&rng)) * 2. - vec2f(1.)) * 2. * rand_f32(&rng),
    );
  }
}

@group(0) @binding(1) var<storage, read>       prev_particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> next_particles: array<Particle>;
@group(0) @binding(3) var<storage, read_write> centers: array<vec2<f32>, 3>;

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let dt = f32(uniforms.time_delta_ms) * 0.0008;
  if idx < uniforms.particle_count {
    let p0 = prev_particles[idx].pos;
    let v0 = prev_particles[idx].velocity;
    var a = vec2f(0.);
    for (var i = 0; i < 3; i += 1) {
      a += 0.5 * normalize(centers[i] - p0);
    }
    next_particles[idx] = Particle(
        p0 + v0 * dt + a * dt * dt * 0.5,
        v0 + a * dt,
    );
  }

  workgroupBarrier();

  if idx == 0u {
    let t = f32(uniforms.total_time_ms) * 0.000001;
    let c = cos(t);
    let s = sin(t);
    let m = mat2x2f(c, -s, s, c);
    for (var i = 0; i < 3; i += 1) {
      centers[i] = m * centers[i];
    }
  }
}
