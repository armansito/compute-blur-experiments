use crate::resource::PixelFormat;

#[derive(Debug)]
pub enum ShaderModule {
    #[cfg(feature = "metal")]
    Metal(mtl::Library),

    #[cfg(feature = "wgpu")]
    Wgpu(wgpu::ShaderModule),

    Unsupported,
}

#[derive(Debug)]
pub enum RenderPipelineHandle {
    #[cfg(feature = "metal")]
    Metal(mtl::RenderPipelineState),

    #[cfg(feature = "wgpu")]
    Wgpu {
        inner: wgpu::RenderPipeline,
        bind_group_layout: Option<wgpu::BindGroupLayout>,
        pipeline_layout: wgpu::PipelineLayout,
    },

    Unsupported,
}

#[derive(Debug)]
pub enum ComputePipelineHandle {
    #[cfg(feature = "metal")]
    Metal(mtl::ComputePipelineState),

    #[cfg(feature = "wgpu")]
    Wgpu {
        inner: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        pipeline_layout: wgpu::PipelineLayout,
    },

    Unsupported,
}

#[derive(PartialEq, Eq)]
pub enum BindingType {
    Uniform,
    Storage,
    ReadOnlyStorage,
    Sampler(SamplerBindingType),
    Texture {
        filterable: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: PixelFormat,
    },
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum StorageTextureAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

pub enum BindingLayout<'a> {
    Empty,
    IndexedByOrder(&'a [BindingType]),
}

pub enum ShaderModuleDescriptor<'a> {
    Source(&'a str),
}

#[derive(Copy, Clone)]
pub enum PrimitiveTopology {
    Point,
    Line,
    LineStrip,
    Triangle,
    TriangleStrip,
}

#[derive(Copy, Clone)]
pub enum FillMode {
    Fill,
    Lines,
}

pub struct ProgrammableStage<'a> {
    pub module: &'a ShaderModule,
    pub entry_point: &'a str,
}

pub enum VertexStepMode {
    Vertex,
    Instance,
}

pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
}

pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u32,
    pub shader_location: u32,
}

pub struct VertexBufferLayout<'a> {
    pub array_stride: u32,
    pub step_mode: VertexStepMode,
    pub attributes: &'a [VertexAttribute],
}

pub struct VertexStage<'a> {
    pub program: ProgrammableStage<'a>,
    pub buffers: Option<&'a [VertexBufferLayout<'a>]>,
}

#[derive(Copy, Clone)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Copy, Clone)]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
}

pub struct BlendComponent {
    pub op: BlendOp,
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
}

pub struct Blend {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

pub struct ColorTarget {
    pub format: PixelFormat,
    pub blend: Option<Blend>,
}

pub struct FragmentStage<'a> {
    pub program: ProgrammableStage<'a>,
    pub color_targets: &'a [ColorTarget],
}

pub struct RenderPipelineDescriptor<'a> {
    pub label: Option<&'static str>,
    pub topology: PrimitiveTopology,
    pub fill_mode: FillMode,
    pub layout: BindingLayout<'a>,
    pub vertex_stage: VertexStage<'a>,
    pub fragment_stage: Option<FragmentStage<'a>>,
}

pub struct RenderPipelineState {
    pub handle: RenderPipelineHandle,
    pub topology: PrimitiveTopology,
    pub fill_mode: FillMode,
}

pub struct ComputePipelineDescriptor<'a> {
    pub label: Option<&'static str>,
    pub program: ProgrammableStage<'a>,
    pub layout: BindingLayout<'a>,
}

pub struct ComputePipelineState {
    pub handle: ComputePipelineHandle,
}
