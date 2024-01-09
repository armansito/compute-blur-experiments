#![allow(non_upper_case_globals)]

#[derive(Debug)]
pub enum BufferUsage {
    Vertex,
    VertexStorage,
    Index,
    IndexStorage,
    Uniform,
    Storage,
}

#[derive(Debug, PartialEq)]
pub enum AccessPattern {
    GpuOnly,
    HostVisible,
}

#[derive(Debug)]
pub enum BufferHandle {
    #[cfg(feature = "metal")]
    Metal(mtl::Buffer),

    #[cfg(feature = "wgpu")]
    Wgpu(wgpu::Buffer),

    Unsupported,
}

#[derive(Debug)]
pub struct Buffer {
    pub handle: BufferHandle,
    pub size: u64,
    pub usage: BufferUsage,
    pub access: AccessPattern,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PixelFormat {
    BGRA8Unorm,
    RGBA8Unorm,
    BGRA8Unorm_sRGB,
    RGBA16Float,
    RGBA32Float,
}

#[cfg(feature = "wgpu")]
impl From<wgpu::TextureFormat> for PixelFormat {
    fn from(src: wgpu::TextureFormat) -> Self {
        match src {
            wgpu::TextureFormat::Bgra8Unorm => PixelFormat::BGRA8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb => PixelFormat::BGRA8Unorm_sRGB,
            wgpu::TextureFormat::Rgba16Float => PixelFormat::RGBA16Float,
            wgpu::TextureFormat::Rgba32Float => PixelFormat::RGBA32Float,
            _ => unimplemented!(),
        }
    }
}

pub enum TextureDimension {
    D1 {
        width: u32,
        array_layers: u32,
    },
    D2 {
        width: u32,
        height: u32,
        array_layers: u32,
    },
    D3 {
        width: u32,
        height: u32,
        depth: u32,
    },
}

impl TextureDimension {
    pub fn make_2d(width: u32, height: u32) -> TextureDimension {
        TextureDimension::D2 {
            width,
            height,
            array_layers: 1,
        }
    }
}

bitflags! {
    pub struct TextureUsage: u8 {
        const CopySrc = 0x01;
        const CopyDst = 0x02;
        const Sample  = 0x04;
        const Storage = 0x08;
        const RenderTarget = 0x10;
    }
}

pub struct TextureDescriptor {
    pub dimension: TextureDimension,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub format: PixelFormat,
    pub usage: TextureUsage,
}

#[derive(Debug)]
pub enum TextureHandle {
    #[cfg(feature = "metal")]
    Metal(mtl::Texture),

    #[cfg(feature = "wgpu")]
    Wgpu {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },

    #[cfg(feature = "wgpu")]
    WgpuView(wgpu::TextureView),

    Unsupported,
}

#[derive(Copy, Clone)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder,
}

#[derive(Copy, Clone)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Copy, Clone)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Copy, Clone)]
pub enum SamplerBorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

#[derive(Copy, Clone)]
pub struct SamplerMipFilter {
    pub filter: FilterMode,
    pub min_lod_clamp: f32,
    pub max_lod_clamp: f32,
}

pub struct SamplerDescriptor {
    pub address_mode: (AddressMode, AddressMode, AddressMode), // u, v, w
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mip_filter: Option<SamplerMipFilter>,
    pub compare: Option<CompareFunction>,
    pub max_anisotropy: u16,
    pub border_color: Option<SamplerBorderColor>,
}

impl SamplerDescriptor {
    pub fn new(
        address_mode: AddressMode,
        filter_mode: FilterMode,
        border_color: Option<SamplerBorderColor>,
    ) -> Self {
        Self {
            address_mode: (address_mode, address_mode, address_mode),
            min_filter: filter_mode,
            mag_filter: filter_mode,
            mip_filter: None,
            compare: None,
            max_anisotropy: 1,
            border_color,
        }
    }
}

#[derive(Debug)]
pub enum SamplerHandle {
    #[cfg(feature = "metal")]
    Metal(mtl::SamplerState),

    #[cfg(feature = "wgpu")]
    Wgpu(wgpu::Sampler),

    Unsupported,
}
