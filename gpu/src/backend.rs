use crate::{
    command::Command,
    pipeline::{
        ComputePipelineDescriptor, ComputePipelineState, RenderPipelineDescriptor,
        RenderPipelineState, ShaderModule, ShaderModuleDescriptor,
    },
    resource::{
        AccessPattern, Buffer, BufferUsage, SamplerDescriptor, SamplerHandle, TextureDescriptor,
        TextureHandle,
    },
    Error, SubmitHandlers,
};

#[cfg(feature = "metal")]
pub(crate) mod metal;

#[cfg(feature = "wgpu")]
pub(crate) mod wgpu;

pub trait Adapter {
    fn submit_commands(
        &mut self,
        commands: &[Command],
        handlers: Option<SubmitHandlers>,
    ) -> Result<(), Error>;

    fn new_buffer(&self, length: u64, usage: BufferUsage, access: AccessPattern) -> Buffer;
    fn new_buffer_with_data<T>(
        &self,
        data: &[T],
        usage: BufferUsage,
        access: AccessPattern,
    ) -> Buffer;

    fn new_texture(&self, descriptor: &TextureDescriptor) -> TextureHandle;
    fn new_sampler(&self, descriptor: &SamplerDescriptor) -> SamplerHandle;

    fn new_shader_module(&self, descriptor: &ShaderModuleDescriptor)
        -> Result<ShaderModule, Error>;
    fn new_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor,
    ) -> Result<RenderPipelineState, Error>;
    fn new_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<ComputePipelineState, Error>;

    fn upload_texture(&self, texture: &TextureHandle, data: &[u8], bytes_per_row: u32);
}
