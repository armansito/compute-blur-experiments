use {
    anyhow::{Context, Result},
    bytemuck::{Pod, Zeroable},
    image::EncodableLayout,
    gpu::command::*,
    std::path::Path,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    filter_dim: u32,
    block_dim: u32,
}

pub struct Blurs {
    render_pp: gpu::RenderPipelineState,
    blur_pp: gpu::ComputePipelineState,

    src_texture: gpu::TextureHandle,
    blur_textures: [gpu::TextureHandle; 2],
    sampler: gpu::SamplerHandle,

    uniforms: Uniforms,
    uniforms_buffer: gpu::Buffer,
    flip_buffers: [gpu::Buffer; 2],
    needs_init: bool,
}

const TILE_DIM: u32 = 128;

impl Blurs {
    pub fn init<A>(
        ctx: &gpu::Context<A>,
        shaders: &str,
        framebuffer_format: gpu::PixelFormat,
    ) -> Result<Self>
    where
        A: gpu::Adapter,
    {
        let shader_module = ctx.new_shader_module(gpu::ShaderModuleDescriptor::Source(shaders))?;
        let render_pp = ctx.new_render_pipeline(&gpu::RenderPipelineDescriptor {
            label: None,
            topology: gpu::PrimitiveTopology::Triangle,
            layout: gpu::BindingLayout::IndexedByOrder(&[
                gpu::BindingType::Sampler(gpu::SamplerBindingType::Filtering),
                gpu::BindingType::Texture { filterable: true },
            ]),
            vertex_stage: gpu::VertexStage {
                program: gpu::ProgrammableStage {
                    module: &shader_module,
                    entry_point: "vs_main",
                },
                buffers: None,
            },
            fragment_stage: Some(gpu::FragmentStage {
                program: gpu::ProgrammableStage {
                    module: &shader_module,
                    entry_point: "fs_main",
                },
                color_targets: &[gpu::ColorTarget {
                    format: framebuffer_format,
                    blend: None,
                }],
            }),
        })?;
        let blur_pp = ctx.new_compute_pipeline(&gpu::ComputePipelineDescriptor {
            label: Some("blur"),
            program: gpu::ProgrammableStage {
                module: &shader_module,
                entry_point: "blur",
            },
            layout: gpu::BindingLayout::IndexedByOrder(&[
                gpu::BindingType::Sampler(gpu::SamplerBindingType::Filtering),
                gpu::BindingType::Uniform,
                gpu::BindingType::Texture { filterable: true },
                gpu::BindingType::StorageTexture {
                    access: gpu::StorageTextureAccess::WriteOnly,
                    format: gpu::PixelFormat::RGBA8Unorm,
                },
                gpu::BindingType::Uniform,
            ]),
        })?;

        let img = image::open(&Path::new("/Users/armansito/Code/personal/blur-experiments/images/mandrill_512.png"))?;
        let img = img.into_rgba8();
        let (width, height) = img.dimensions();
        println!("img dims: {}x{}", width, height);
        let src_texture = ctx.new_texture(&gpu::TextureDescriptor {
            dimension: gpu::TextureDimension::make_2d(width, height),
            mip_level_count: 1,
            sample_count: 1,
            format: gpu::PixelFormat::RGBA8Unorm,
            usage: gpu::TextureUsage::Sample | gpu::TextureUsage::CopyDst,
        });
        ctx.upload_texture(&src_texture, img.as_bytes(), width * 4);

        let blur_textures = [
            ctx.new_texture(&gpu::TextureDescriptor {
                dimension: gpu::TextureDimension::make_2d(width, height),
                mip_level_count: 1,
                sample_count: 1,
                format: gpu::PixelFormat::RGBA8Unorm,
                usage: gpu::TextureUsage::Sample | gpu::TextureUsage::Storage,
            }),
            ctx.new_texture(&gpu::TextureDescriptor {
                dimension: gpu::TextureDimension::make_2d(width, height),
                mip_level_count: 1,
                sample_count: 1,
                format: gpu::PixelFormat::RGBA8Unorm,
                usage: gpu::TextureUsage::Sample | gpu::TextureUsage::Storage,
            }),
        ];

        let sampler = ctx.new_sampler(&gpu::SamplerDescriptor::new(
            gpu::AddressMode::Repeat,
            gpu::FilterMode::Linear,
            None,
        ));

        let filter_dim = 32;
        let uniforms = Uniforms {
            filter_dim,
            block_dim: TILE_DIM - (filter_dim - 1),
        };
        let uniforms_buffer = ctx.new_buffer(
            std::mem::size_of::<Uniforms>() as u64,
            gpu::BufferUsage::Uniform,
            gpu::AccessPattern::HostVisible,
        );
        let flip_buffers = [
            ctx.new_buffer(
                std::mem::size_of::<u32>() as u64,
                gpu::BufferUsage::Uniform,
                gpu::AccessPattern::HostVisible,
            ),
            ctx.new_buffer(
                std::mem::size_of::<u32>() as u64,
                gpu::BufferUsage::Uniform,
                gpu::AccessPattern::HostVisible,
            ),
        ];

        Ok(Blurs {
            render_pp,
            blur_pp,
            src_texture,
            blur_textures,
            sampler,
            uniforms,
            uniforms_buffer,
            flip_buffers,
            needs_init: true,
        })
    }

    pub fn render_frame<A, F>(
        &mut self,
        ctx: &mut gpu::Context<A>,
        target: &gpu::TextureHandle,
        present_drawable: F,
    ) -> Result<()>
    where
        A: gpu::Adapter,
        F: Fn() + Send + 'static,
    {
        let needs_init = self.needs_init;
        self.needs_init = false;

        let handlers = gpu::SubmitHandlers {
            scheduled: Some(Box::new(present_drawable)),
            completion: None,
        };
        let workgroup_count = gpu::WorkgroupExtent::new(
            (512 + self.uniforms.block_dim - 1) / self.uniforms.block_dim,
            512 / 4,
            1,
        );
        ctx.submit(
            &[
                Command::WriteBuffer {
                    buffer: &self.uniforms_buffer,
                    data: bytemuck::bytes_of(&self.uniforms),
                    offset: 0,
                },
                Command::WriteBuffer {
                    buffer: &self.flip_buffers[0],
                    data: bytemuck::bytes_of(&(0 as u32)),
                    offset: 0,
                },
                Command::WriteBuffer {
                    buffer: &self.flip_buffers[1],
                    data: bytemuck::bytes_of(&(1 as u32)),
                    offset: 0,
                },
                // First pass
                Command::ComputePass(ComputeParams {
                    pipeline: &self.blur_pp,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Sampler(&self.sampler),
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.uniforms_buffer,
                        },
                        Resource::Texture(&self.src_texture),
                        Resource::Texture(&self.blur_textures[0]),
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.flip_buffers[0],
                        },
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(32, 1, 1),
                        workgroup_count,
                    }],
                }),
                Command::ComputePass(ComputeParams {
                    pipeline: &self.blur_pp,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Sampler(&self.sampler),
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.uniforms_buffer,
                        },
                        Resource::Texture(&self.blur_textures[0]),
                        Resource::Texture(&self.blur_textures[1]),
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.flip_buffers[1],
                        },
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(32, 1, 1),
                        workgroup_count,
                    }],
                }),
                // Second pass
                Command::ComputePass(ComputeParams {
                    pipeline: &self.blur_pp,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Sampler(&self.sampler),
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.uniforms_buffer,
                        },
                        Resource::Texture(&self.blur_textures[1]),
                        Resource::Texture(&self.blur_textures[0]),
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.flip_buffers[0],
                        },
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(32, 1, 1),
                        workgroup_count,
                    }],
                }),
                Command::ComputePass(ComputeParams {
                    pipeline: &self.blur_pp,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Sampler(&self.sampler),
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.uniforms_buffer,
                        },
                        Resource::Texture(&self.blur_textures[0]),
                        Resource::Texture(&self.blur_textures[1]),
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.flip_buffers[1],
                        },
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(32, 1, 1),
                        workgroup_count,
                    }],
                }),
                Command::RenderPass(RenderParams {
                    target,
                    clear_color: Some([0., 0., 0., 1.]),
                    pipeline: Some(&self.render_pp),
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Sampler(&self.sampler),
                        Resource::Texture(&self.blur_textures[1]),
                    ]),
                    vertex_buffers: BindingList::Empty,
                    commands: &[RenderCommand::Draw {
                        vertex_count: 6,
                        instance_count: 1,
                        first_vertex: 0,
                        first_instance: 0,
                    }],
                }),
            ],
            Some(handlers),
        ).context("failed to render")
    }
}
