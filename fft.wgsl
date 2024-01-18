// TODO: put attribution about porting

const WORKGROUP_SIZE_X: u32 = 256u;
const PIXEL_BUFFER_SIZE: u32 = 32u;
const SHARED_BUFFER_SIZE: u32 = 1024u;

const PI: f32 = 3.14159265358979323846264338327950288;

struct Uniforms {
  input_width: u32,
  input_height: u32,
  output_width: u32,
  output_height: u32,
  logtwo_width: u32,
  logtwo_height: u32,
  clz_width: u32,
  clz_height: u32,
  no_of_channels: u32,
  blur_value: u32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(0) @binding(1) var inputImage: texture_2d<f32>;
@group(0) @binding(2) var realPart: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var imagPart: texture_storage_2d<rgba32float, read_write>;

var<workgroup> real_cache: array<f32, SHARED_BUFFER_SIZE>;
var<workgroup> imag_cache: array<f32, SHARED_BUFFER_SIZE>;

var<private> pixel_buffer_real: array<vec4f, PIXEL_BUFFER_SIZE>;
var<private> pixel_buffer_imag: array<vec4f, PIXEL_BUFFER_SIZE>;

fn complex_mul(lhs: vec2f, rhs: vec2f) -> vec2f {
  let a = lhs * rhs;
  let b = lhs.yx * rhs;
  return vec2(a.x - a.y, b.x + b.y);
}

fn index_map(tid: u32, current_iteration: u32, N: u32) -> u32 {
  return ((tid & (N - (1u << current_iteration))) << 1u)
         | (tid & ((1u << current_iteration) - 1u));
}

fn twiddle_map(tid: u32, current_iteration: u32, log_two: u32, N: u32) -> u32 {
  return (tid & (N / (1u << (log_two - current_iteration)) - 1u)) * (1u << (log_two - current_iteration)) >> 1u;
}

fn twiddle(q: f32, is_inverse: bool, N: f32) -> vec2f {
  let theta = f32(i32(!is_inverse) * 2 - 1) * 2. * PI * q / N;
  let r = cos(theta);
  let i = sqrt(1. - r * r) * f32(i32(theta < 0.) * 2 - 1);
  return vec2(r, i);
}

fn fft_radix2(log_two: i32, btid: i32, g_offset: i32, is_inverse: bool, N: f32) {
  for (var i = 0; i < log_two; i += 1) {
    for (var j = btid; j < btid + g_offset; j += 1) {
      let even = index_map(u32(j), u32(i), u32(N));
      let odd = even + (1u << u32(i));
      let even_val = vec2(real_cache[even], imag_cache[even]);
      let q = twiddle_map(u32(j), u32(i), u32(log_two), u32(N));
      let e = complex_mul(twiddle(f32(q), is_inverse, N), vec2(real_cache[odd], imag_cache[odd]));

      let calc_even = even_val + e;
      let calc_odd = even_val - e;

      real_cache[even] = calc_even.x;
      imag_cache[even] = calc_even.y;
      real_cache[odd] = calc_odd.x;
      imag_cache[odd] = calc_odd.y;
    }
    workgroupBarrier();
  }
}

fn load_stage0(btid: i32, g_offset: i32, scanline: i32) {
  for (var i = btid * 2; i < btid * 2 + g_offset * 2; i += 1) {
    let j = i32(reverseBits(u32(i)) >> uniforms.clz_width);
    pixel_buffer_real[i - btid * 2] = textureLoad(inputImage, vec2(j, scanline), 0);
    pixel_buffer_imag[i - btid * 2] = vec4(0.);
  }
}

fn store_stage0(btid: i32, g_offset: i32, scanline: i32) {
  for (var i = btid * 2; i < btid * 2 + g_offset * 2; i += 1) {
    let ix = vec2(i, scanline);
    textureStore(realPart, ix, pixel_buffer_real[i - btid * 2]);
    textureStore(imagPart, ix, pixel_buffer_imag[i - btid * 2]);
  }
}

fn load_stage1(btid: i32, g_offset: i32, scanline: i32, is_inverse: bool) {
  for (var i = btid * 2; i < btid * 2 + g_offset * 2; i += 1) {
    let j = i32(reverseBits(u32(i)) >> uniforms.clz_height);
    let ix = vec2(scanline, j);
    let real = textureLoad(realPart, ix);
    let imag = textureLoad(imagPart, ix);

    // Apply a triangle filter in the frequency domain.
    let v = uniforms.blur_value;
    let height = uniforms.output_height;
    let x = select(u32(j), height - u32(j), u32(j) > height / 2u);
    let s = select(0., 1. - f32(x) / f32(v), x < v);
    let f = select(1., s, is_inverse);

    pixel_buffer_real[i - btid * 2] = real * f;//select(vec4(0.), real, good);
    pixel_buffer_imag[i - btid * 2] = imag * f;//select(vec4(0.), imag, good);
  }
}

fn store_stage1(btid: i32, g_offset: i32, scanline: i32, N: f32) {
  for (var i = btid * 2; i < btid * 2 + g_offset * 2; i += 1) {
    let ix = vec2(scanline, i);
    textureStore(realPart, ix, pixel_buffer_real[i - btid * 2] * N);
    textureStore(imagPart, ix, pixel_buffer_imag[i - btid * 2] * N);
  }
}

fn load_stage3(btid: i32, g_offset: i32, scanline: i32) {
  for (var i = btid * 2; i < btid * 2 + g_offset * 2; i += 1) {
    let j = i32(reverseBits(u32(i)) >> uniforms.clz_width);
    let ix = vec2(j, scanline);
    let real = textureLoad(realPart, ix);
    let imag = textureLoad(imagPart, ix);

    // Apply a triangle filter in the frequency domain.
    let v = uniforms.blur_value;
    let width = uniforms.output_width;
    let x = select(u32(j), width - u32(j), u32(j) > width / 2u);
    let s = select(0., 1. - f32(x) / f32(v), x < v);

    pixel_buffer_real[i - btid * 2] = real * s;
    pixel_buffer_imag[i - btid * 2] = imag * s;
  }
}

fn store_stage3(btid: i32, g_offset: i32, scanline: i32, N: f32) {
  for (var i = btid * 2; i < btid * 2 + g_offset * 2; i += 1) {
    if u32(i) < uniforms.input_width {
      let ix = vec2(i, scanline);
      textureStore(outputImage, ix, pixel_buffer_real[i - btid * 2] * N);
    }
  }
}

fn load_into_cache(btid: i32, g_offset: i32, channel: i32) {
	for(var i = btid * 2; i < btid * 2 + g_offset * 2; i += 1) {
		real_cache[i] = pixel_buffer_real[i - btid * 2][channel];
		imag_cache[i] = pixel_buffer_imag[i - btid * 2][channel];
	}
}

fn load_from_cache(btid: i32, g_offset: i32, channel: i32) {
	for(var i = btid * 2; i < btid * 2 + g_offset * 2; i += 1) {
		pixel_buffer_real[i - btid * 2][channel] = real_cache[i];
		pixel_buffer_imag[i - btid * 2][channel] = imag_cache[i];
	}
}

@compute @workgroup_size(256, 1, 1)
fn stage0(@builtin(local_invocation_id) local_id: vec3u,
          @builtin(workgroup_id) wg_id: vec3u) {
  let N = i32(uniforms.output_width);
  let g_offset = i32(u32(N) / 2u / WORKGROUP_SIZE_X);
  let btid = i32(g_offset * i32(local_id.x));

  load_stage0(btid, g_offset, i32(wg_id.x));
  workgroupBarrier();

  for (var channel = 0; channel < i32(uniforms.no_of_channels); channel += 1) {
    load_into_cache(btid, g_offset, channel);
    workgroupBarrier();

    fft_radix2(i32(uniforms.logtwo_width), i32(btid), g_offset, false, f32(N));
    workgroupBarrier();

    load_from_cache(btid, g_offset, channel);
  }
  
  workgroupBarrier();
  store_stage0(btid, g_offset, i32(wg_id.x));
}

@compute @workgroup_size(256, 1, 1)
fn stage1(@builtin(local_invocation_id) local_id: vec3u,
          @builtin(workgroup_id) wg_id: vec3u) {
  let N = i32(uniforms.output_height);
  let g_offset = i32(u32(N) / 2u / WORKGROUP_SIZE_X);
  let btid = i32(g_offset * i32(local_id.x));
  let divisor = 1.;

  load_stage1(btid, g_offset, i32(wg_id.x), false);
  workgroupBarrier();

  for (var channel = 0; channel < i32(uniforms.no_of_channels); channel += 1) {
    load_into_cache(btid, g_offset, channel);
    workgroupBarrier();

    fft_radix2(i32(uniforms.logtwo_height), i32(btid), g_offset, false, f32(N));
    workgroupBarrier();

    load_from_cache(btid, g_offset, channel);
  }
  
  workgroupBarrier();
  store_stage1(btid, g_offset, i32(wg_id.x), divisor);
}

@compute @workgroup_size(256, 1, 1)
fn stage1_inverse(@builtin(local_invocation_id) local_id: vec3u,
                  @builtin(workgroup_id) wg_id: vec3u) {
  let N = i32(uniforms.output_height);
  let g_offset = i32(u32(N) / 2u / WORKGROUP_SIZE_X);
  let btid = i32(g_offset * i32(local_id.x));
  let divisor = 1. / f32(N);

  load_stage1(btid, g_offset, i32(wg_id.x), true);
  workgroupBarrier();

  for (var channel = 0; channel < i32(uniforms.no_of_channels); channel += 1) {
    load_into_cache(btid, g_offset, channel);
    workgroupBarrier();

    fft_radix2(i32(uniforms.logtwo_height), i32(btid), g_offset, true, f32(N));
    workgroupBarrier();

    load_from_cache(btid, g_offset, channel);
  }
  
  workgroupBarrier();
  store_stage1(btid, g_offset, i32(wg_id.x), divisor);
}

@group(0) @binding(1) var outputImage: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(256, 1, 1)
fn stage_inverse_final(@builtin(local_invocation_id) local_id: vec3u,
                       @builtin(workgroup_id) wg_id: vec3u) {
  let N = i32(uniforms.output_width);
  let g_offset = i32(u32(N) / 2u / WORKGROUP_SIZE_X);
  let btid = i32(g_offset * i32(local_id.x));
  let divisor = 1. / f32(N);

  load_stage3(btid, g_offset, i32(wg_id.x));
  workgroupBarrier();

  for (var channel = 0; channel < i32(uniforms.no_of_channels); channel += 1) {
    load_into_cache(btid, g_offset, channel);
    workgroupBarrier();

    fft_radix2(i32(uniforms.logtwo_width), i32(btid), g_offset, true, f32(N));
    workgroupBarrier();

    load_from_cache(btid, g_offset, channel);
  }
  
  workgroupBarrier();
  store_stage3(btid, g_offset, i32(wg_id.x), divisor);
}

@compute @workgroup_size(16, 16, 1)
fn clear(@builtin(global_invocation_id) tid: vec3u) {
  if (tid.x < uniforms.output_width && tid.y < uniforms.output_height) {
    textureStore(realPart, tid.xy, vec4<f32>(0.));
    textureStore(imagPart, tid.xy, vec4<f32>(0.));
  }
}
