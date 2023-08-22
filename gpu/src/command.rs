use crate::{
    pipeline::{ComputePipelineState, RenderPipelineState},
    resource::{Buffer, SamplerHandle, TextureHandle},
};

pub enum Command<'a> {
    Optional(Option<&'a Command<'a>>),
    WriteBuffer {
        buffer: &'a Buffer,
        data: &'a [u8],
        offset: u32,
    },
    RenderPass(RenderParams<'a>),
    ComputePass(ComputeParams<'a>),
}

pub struct RenderParams<'a> {
    pub target: &'a TextureHandle,
    pub clear_color: Option<[f32; 4]>,
    pub pipeline: Option<&'a RenderPipelineState>,

    pub bindings: BindingList<'a>,
    pub vertex_buffers: BindingList<'a>,

    pub commands: &'a [RenderCommand],
}

pub struct ComputeParams<'a> {
    pub pipeline: &'a ComputePipelineState,
    pub bindings: BindingList<'a>,
    pub commands: &'a [ComputeCommand],
}

pub enum BindingList<'a> {
    /// No resources will be bound when executing the command.
    Empty,

    /// Resources will be assigned a binding index based on its declaration order. Binding indices
    /// will be contiguous and in increasing order and will be transformed based on backend-specific
    /// resource type groupings. For instance:
    ///
    ///    IndexedByOrder(&[
    ///       Resource::Buffer{ .. },
    ///       Resource::Texture(..),
    ///       Resource::Buffer{ .. },
    ///    ])
    ///
    /// corresponds to:
    ///
    ///    WGSL:
    ///         @group(0) @binding(0) var ... buffer0
    ///         @group(0) @binding(1) var ... texture
    ///         @group(0) @binding(2) var ... buffer1
    ///
    ///    MSL:
    ///         buffer0 [[buffer(0)]]
    ///         texture [[texture(0)]]
    ///         buffer1 [[buffer(1)]]
    IndexedByOrder(&'a [Resource<'a>]),
}

pub struct Binding<'a> {
    pub index: u32,
    pub resource: Resource<'a>,
}

#[derive(Debug)]
pub enum Resource<'a> {
    Buffer { offset: u32, buffer: &'a Buffer },
    Texture(&'a TextureHandle),
    Sampler(&'a SamplerHandle),
}

pub enum RenderCommand {
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    },
}

impl RenderCommand {
    pub fn draw(vertex_count: u32) -> Self {
        RenderCommand::Draw {
            vertex_count,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        }
    }

    pub fn draw_indexed(index_count: u32) -> Self {
        RenderCommand::DrawIndexed {
            index_count,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }
}

// TODO: make this part of the pipeline state once there is a mechanism to read this from shader
// text (needed for other platforms).
#[derive(Copy, Clone)]
pub struct WorkgroupExtent {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl WorkgroupExtent {
    pub fn new(width: u32, height: u32, depth: u32) -> WorkgroupExtent {
        WorkgroupExtent {
            width,
            height,
            depth,
        }
    }
}

pub enum ComputeCommand {
    Dispatch {
        workgroup_size: WorkgroupExtent,
        workgroup_count: WorkgroupExtent,
    },
}
