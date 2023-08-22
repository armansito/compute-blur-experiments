// Adapted from https://webgpu.github.io/webgpu-samples/samples/imageBlur

struct Params {
  filterDim : i32,
  blockDim : u32,
}

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var inputTex : texture_2d<f32>;
@group(0) @binding(3) var outputTex : texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(4) var<uniform> flip: u32;

// This shader blurs the input texture in one direction, depending on whether
// |flip.value| is 0 or 1.
// It does so by running (128 / 4) threads per workgroup to load 128
// texels into 4 rows of shared memory. Each thread loads a
// 4 x 4 block of texels to take advantage of the texture sampling
// hardware.
// Then, each thread computes the blur result by averaging the adjacent texel values
// in shared memory.
// Because we're operating on a subset of the texture, we cannot compute all of the
// results since not all of the neighbors are available in shared memory.
// Specifically, with 128 x 128 tiles, we can only compute and write out
// square blocks of size 128 - (filterSize - 1). We compute the number of blocks
// needed in Javascript and dispatch that amount.

var<workgroup> tile : array<array<vec3<f32>, 128>, 4>;

@compute @workgroup_size(32, 1, 1)
fn blur(
  @builtin(workgroup_id) WorkGroupID : vec3<u32>,
  @builtin(local_invocation_id) LocalInvocationID : vec3<u32>
) {
  let filterOffset = (params.filterDim - 1) / 2;
  let dims = vec2<i32>(textureDimensions(inputTex, 0));
  let baseIndex = vec2<i32>(WorkGroupID.xy * vec2<u32>(params.blockDim, 4u) +
                            LocalInvocationID.xy * vec2<u32>(4u, 1u))
                  - vec2(filterOffset, 0);

  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 4; c++) {
      var loadIndex = baseIndex + vec2(c, r);
      if (flip != 0u) {
        loadIndex = loadIndex.yx;
      }

      tile[r][4u * LocalInvocationID.x + u32(c)] = textureSampleLevel(
        inputTex,
        samp,
        (vec2<f32>(loadIndex) + vec2<f32>(0.25, 0.25)) / vec2<f32>(dims),
        0.0
      ).rgb;
    }
  }

  workgroupBarrier();

  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 4; c++) {
      var writeIndex = baseIndex + vec2(c, r);
      if (flip != 0u) {
        writeIndex = writeIndex.yx;
      }

      let center = i32(4u * LocalInvocationID.x) + c;
      if (center >= filterOffset &&
          center < 128 - filterOffset &&
          all(writeIndex < dims)) {
        var acc = vec3(0.0, 0.0, 0.0);
        for (var f = 0; f < params.filterDim; f++) {
          var i = center + f - filterOffset;
          acc = acc + (1.0 / f32(params.filterDim)) * tile[r][i];
        }
        textureStore(outputTex, writeIndex, vec4(acc, 1.0));
      }
    }
  }
}




// render ==========

@group(0) @binding(0) var mySampler : sampler;
@group(0) @binding(1) var myTexture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4<f32>,
  @location(0) fragUV : vec2<f32>,
}

var<private> pos: array<vec2f, 6> = array(
  vec2( 1.0,  1.0),
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2( 1.0,  1.0),
  vec2(-1.0,  1.0),
  vec2(-1.0, -1.0),
);

var<private> uv: array<vec2f, 6>  = array(
  vec2(1.0, 0.0),
  vec2(0.0, 1.0),
  vec2(1.0, 1.0),
  vec2(1.0, 0.0),
  vec2(0.0, 0.0),
  vec2(0.0, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
  var output : VertexOutput;
  output.Position = vec4(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = uv[VertexIndex];
  return output;
}

@fragment
fn fs_main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(myTexture, mySampler, fragUV);
}
