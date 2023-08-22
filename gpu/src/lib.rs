use {std::boxed::Box, thiserror::Error};

#[macro_use]
extern crate bitflags;

mod backend;
mod pipeline;
mod resource;

pub mod command;
pub mod sync;
pub use backend::Adapter;
pub use command::{Command, WorkgroupExtent};
pub use pipeline::*;
pub use resource::*;
pub use sync::Semaphore;

#[cfg(feature = "metal")]
pub fn metal_context(device: &mtl::Device) -> Context<backend::metal::Backend> {
    backend::metal::new_context(device)
}

#[cfg(feature = "wgpu")]
pub fn wgpu_context(
    device: std::rc::Rc<wgpu::Device>,
    queue: wgpu::Queue,
) -> Context<backend::wgpu::Backend> {
    backend::wgpu::new_context(device, queue)
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("failed to compile shader module: {0}")]
    ShaderCompilation(String),

    #[error("failed to create pipeline: {0}")]
    PipelineCreation(String),

    #[error("invalid function entry-point: {0}")]
    ModuleEntryPoint(String),

    #[error("invalid render command: {0}")]
    InvalidRenderCommand(String),

    #[error("buffer usage is not mappable")]
    InvalidBufferUsage,

    #[error("buffer not large enough: {0}")]
    InvalidBufferSize(u64),
}

pub struct Context<A> {
    backend: A,
}

pub struct SubmitHandlers {
    pub scheduled: Option<Box<dyn Fn() + Send>>,
    pub completion: Option<Box<dyn Fn() + Send>>,
}

impl<A> Context<A>
where
    A: Adapter,
{
    pub fn new_buffer(&self, length: u64, usage: BufferUsage, access: AccessPattern) -> Buffer {
        self.backend.new_buffer(length, usage, access)
    }

    pub fn new_buffer_with_data<T>(
        &self,
        data: &[T],
        usage: BufferUsage,
        access: AccessPattern,
    ) -> Buffer {
        self.backend.new_buffer_with_data(data, usage, access)
    }

    pub fn new_texture(&self, descriptor: &TextureDescriptor) -> TextureHandle {
        self.backend.new_texture(descriptor)
    }

    pub fn new_sampler(&self, descriptor: &SamplerDescriptor) -> SamplerHandle {
        self.backend.new_sampler(descriptor)
    }

    pub fn new_shader_module(
        &self,
        descriptor: ShaderModuleDescriptor,
    ) -> Result<ShaderModule, Error> {
        self.backend.new_shader_module(&descriptor)
    }

    pub fn new_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor,
    ) -> Result<RenderPipelineState, Error> {
        self.backend.new_render_pipeline(descriptor)
    }

    pub fn new_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<ComputePipelineState, Error> {
        self.backend.new_compute_pipeline(descriptor)
    }

    pub fn submit(
        &mut self,
        commands: &[Command],
        handlers: Option<SubmitHandlers>,
    ) -> Result<(), Error> {
        self.backend.submit_commands(commands, handlers)
    }

    pub fn upload_texture(&self, texture: &TextureHandle, data: &[u8], bytes_per_row: u32) {
        self.backend.upload_texture(texture, data, bytes_per_row);
    }
}
