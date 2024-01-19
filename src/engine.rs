use {
    anyhow::{Context, Result},
    bytemuck::{Pod, Zeroable},
    gpu::command::*,
    image::EncodableLayout,
    std::path::Path,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    filter_dim: u32,
    block_dim: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FFTUniforms {
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

pub struct Blurs {
    render_pp: gpu::RenderPipelineState,
    blur_pp: gpu::ComputePipelineState,

    src_texture: gpu::TextureHandle,
    blur_textures: [gpu::TextureHandle; 2],
    sampler: gpu::SamplerHandle,

    uniforms: Uniforms,
    uniforms_buffer: gpu::Buffer,
    flip_buffers: [gpu::Buffer; 2],

    // FFT
    fft_clear: gpu::ComputePipelineState,
    fft_stage0: gpu::ComputePipelineState,
    fft_stage1: gpu::ComputePipelineState,
    fft_stage2: gpu::ComputePipelineState,
    fft_stage3: gpu::ComputePipelineState,
    fft_uniforms: FFTUniforms,
    fft_uniforms_buffer: gpu::Buffer,
    fft_real_part: gpu::TextureHandle,
    fft_imag_part: gpu::TextureHandle,
    fft_blur_output: gpu::TextureHandle,

    fft_blur_value: u32,

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
        let fft_module = ctx.new_shader_module(gpu::ShaderModuleDescriptor::Source(
            include_str!("../fft.wgsl"),
        ))?;
        let render_pp = ctx.new_render_pipeline(&gpu::RenderPipelineDescriptor {
            label: None,
            topology: gpu::PrimitiveTopology::Triangle,
            fill_mode: gpu::FillMode::Fill,
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
        let fft_layout = [
            gpu::BindingType::Uniform,
            gpu::BindingType::Texture { filterable: false },
            gpu::BindingType::StorageTexture {
                access: gpu::StorageTextureAccess::ReadWrite,
                format: gpu::PixelFormat::RGBA32Float,
            },
            gpu::BindingType::StorageTexture {
                access: gpu::StorageTextureAccess::ReadWrite,
                format: gpu::PixelFormat::RGBA32Float,
            },
        ];
        let fft_clear = ctx.new_compute_pipeline(&gpu::ComputePipelineDescriptor {
            label: Some("fft_clear"),
            program: gpu::ProgrammableStage {
                module: &fft_module,
                entry_point: "clear",
            },
            layout: gpu::BindingLayout::IndexedByOrder(&fft_layout),
        })?;
        let fft_stage0 = ctx.new_compute_pipeline(&gpu::ComputePipelineDescriptor {
            label: Some("fft_stage0"),
            program: gpu::ProgrammableStage {
                module: &fft_module,
                entry_point: "stage0",
            },
            layout: gpu::BindingLayout::IndexedByOrder(&fft_layout),
        })?;
        let fft_stage1 = ctx.new_compute_pipeline(&gpu::ComputePipelineDescriptor {
            label: Some("fft_stage1"),
            program: gpu::ProgrammableStage {
                module: &fft_module,
                entry_point: "stage1",
            },
            layout: gpu::BindingLayout::IndexedByOrder(&fft_layout),
        })?;
        let fft_stage2 = ctx.new_compute_pipeline(&gpu::ComputePipelineDescriptor {
            label: Some("fft_stage2"),
            program: gpu::ProgrammableStage {
                module: &fft_module,
                entry_point: "stage1_inverse",
            },
            layout: gpu::BindingLayout::IndexedByOrder(&fft_layout),
        })?;
        let fft_stage3 = ctx.new_compute_pipeline(&gpu::ComputePipelineDescriptor {
            label: Some("fft_stage3"),
            program: gpu::ProgrammableStage {
                module: &fft_module,
                entry_point: "stage_inverse_final",
            },
            layout: gpu::BindingLayout::IndexedByOrder(&[
                gpu::BindingType::Uniform,
                gpu::BindingType::StorageTexture {
                    access: gpu::StorageTextureAccess::WriteOnly,
                    format: gpu::PixelFormat::RGBA32Float,
                },
                gpu::BindingType::StorageTexture {
                    access: gpu::StorageTextureAccess::ReadWrite,
                    format: gpu::PixelFormat::RGBA32Float,
                },
                gpu::BindingType::StorageTexture {
                    access: gpu::StorageTextureAccess::ReadWrite,
                    format: gpu::PixelFormat::RGBA32Float,
                },
            ]),
        })?;

        let img = image::open(&Path::new(
            "/Users/armansito/Code/personal/blur-experiments/images/mandrill_1600.png",
        ))?;
        //let img = image::open(&Path::new("/Users/armansito/Code/personal/blur-experiments/images/bunny.png"))?;
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
        ctx.upload_texture(&src_texture, img.as_bytes(), width, height, width * 4);

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
            gpu::FilterMode::Nearest,
            None,
        ));

        let filter_dim = 64;
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

        // FFT stuff
        let fft_width = width.next_power_of_two();
        let fft_height = height.next_power_of_two();
        let fft_img_desc = gpu::TextureDescriptor {
            dimension: gpu::TextureDimension::make_2d(fft_width, fft_height),
            mip_level_count: 1,
            sample_count: 1,
            format: gpu::PixelFormat::RGBA32Float,
            usage: gpu::TextureUsage::Sample | gpu::TextureUsage::Storage,
        };
        let fft_blur_output = ctx.new_texture(&gpu::TextureDescriptor {
            dimension: gpu::TextureDimension::make_2d(width, height),
            mip_level_count: 1,
            sample_count: 1,
            format: gpu::PixelFormat::RGBA32Float,
            usage: gpu::TextureUsage::Sample | gpu::TextureUsage::Storage,
        });
        let fft_real_part = ctx.new_texture(&fft_img_desc);
        let fft_imag_part = ctx.new_texture(&fft_img_desc);

        let fft_uniforms = FFTUniforms {
            input_width: width,
            input_height: height,
            output_width: fft_width,
            output_height: fft_height,
            clz_width: fft_width.leading_zeros() + 1,
            clz_height: fft_height.leading_zeros() + 1,
            logtwo_width: 32 - fft_width.leading_zeros() - 1,
            logtwo_height: 32 - fft_height.leading_zeros() - 1,
            no_of_channels: 4,
            blur_value: 30,
        };
        let fft_uniforms_buffer = ctx.new_buffer(
            std::mem::size_of::<FFTUniforms>() as u64,
            gpu::BufferUsage::Uniform,
            gpu::AccessPattern::HostVisible,
        );

        Ok(Blurs {
            render_pp,
            blur_pp,
            src_texture,
            blur_textures,
            sampler,
            uniforms,
            uniforms_buffer,
            flip_buffers,
            fft_clear,
            fft_stage0,
            fft_stage1,
            fft_stage2,
            fft_stage3,
            fft_uniforms,
            fft_uniforms_buffer,
            fft_real_part,
            fft_imag_part,
            fft_blur_value: 30,
            fft_blur_output,
            needs_init: true,
        })
    }

    pub fn adjust_fft_filter_up(&mut self, val: u32) {
        self.fft_blur_value = self.fft_blur_value.saturating_add(val);
        self.uniforms.filter_dim = self.uniforms.filter_dim.saturating_add(val).min(TILE_DIM);
        self.uniforms.block_dim = TILE_DIM - (self.uniforms.filter_dim - 1);
    }

    pub fn adjust_fft_filter_down(&mut self, val: u32) {
        self.fft_blur_value = self.fft_blur_value.saturating_sub(val);
        self.uniforms.filter_dim = self.uniforms.filter_dim.saturating_sub(val).min(TILE_DIM);
        self.uniforms.block_dim = TILE_DIM - (self.uniforms.filter_dim - 1);
    }

    pub fn render_simple_blur<A, F>(
        &mut self,
        ctx: &mut gpu::Context<A>,
        target: &gpu::TextureHandle,
        present_drawable: F,
    ) -> Result<()>
    where
        A: gpu::Adapter,
        F: Fn() + Send + 'static,
    {
        let _needs_init = self.needs_init;
        self.needs_init = false;

        let handlers = gpu::SubmitHandlers {
            scheduled: Some(Box::new(present_drawable)),
            completion: None,
        };
        let workgroup_count = gpu::WorkgroupExtent::new(
            (self.fft_uniforms.input_width + self.uniforms.block_dim - 1) / self.uniforms.block_dim,
            self.fft_uniforms.input_height / 4,
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
        )
        .context("failed to render")
    }

    pub fn render_fft_blur<A, F>(
        &mut self,
        ctx: &mut gpu::Context<A>,
        target: &gpu::TextureHandle,
        present_drawable: F,
    ) -> Result<()>
    where
        A: gpu::Adapter,
        F: Fn() + Send + 'static,
    {
        let handlers = gpu::SubmitHandlers {
            scheduled: Some(Box::new(present_drawable)),
            completion: None,
        };
        let workgroup_count = gpu::WorkgroupExtent::new(1, 1, 1);
        self.fft_uniforms.blur_value = self.fft_blur_value;
        ctx.submit(
            &[
                Command::WriteBuffer {
                    buffer: &self.fft_uniforms_buffer,
                    data: bytemuck::bytes_of(&self.fft_uniforms),
                    offset: 0,
                },
                Command::ComputePass(ComputeParams {
                    pipeline: &self.fft_clear,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.fft_uniforms_buffer,
                        },
                        Resource::Texture(&self.src_texture),
                        Resource::Texture(&self.fft_real_part),
                        Resource::Texture(&self.fft_imag_part),
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(16, 16, 1),
                        workgroup_count: gpu::WorkgroupExtent::new(
                            self.fft_uniforms.output_width / 16,
                            self.fft_uniforms.output_height / 16,
                            1,
                        ),
                    }],
                }),
                Command::ComputePass(ComputeParams {
                    pipeline: &self.fft_stage0,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.fft_uniforms_buffer,
                        },
                        Resource::Texture(&self.src_texture),
                        Resource::Texture(&self.fft_real_part),
                        Resource::Texture(&self.fft_imag_part),
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(256, 1, 1),
                        workgroup_count: gpu::WorkgroupExtent::new(
                            self.fft_uniforms.output_width,
                            1,
                            1,
                        ),
                    }],
                }),
                Command::ComputePass(ComputeParams {
                    pipeline: &self.fft_stage1,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.fft_uniforms_buffer,
                        },
                        Resource::Texture(&self.src_texture),
                        Resource::Texture(&self.fft_real_part),
                        Resource::Texture(&self.fft_imag_part),
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(256, 1, 1),
                        workgroup_count: gpu::WorkgroupExtent::new(
                            self.fft_uniforms.output_height,
                            1,
                            1,
                        ),
                    }],
                }),
                Command::ComputePass(ComputeParams {
                    pipeline: &self.fft_stage2,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.fft_uniforms_buffer,
                        },
                        Resource::Texture(&self.src_texture),
                        Resource::Texture(&self.fft_real_part),
                        Resource::Texture(&self.fft_imag_part),
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(256, 1, 1),
                        workgroup_count: gpu::WorkgroupExtent::new(
                            self.fft_uniforms.output_height,
                            1,
                            1,
                        ),
                    }],
                }),
                Command::ComputePass(ComputeParams {
                    pipeline: &self.fft_stage3,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.fft_uniforms_buffer,
                        },
                        Resource::Texture(&self.fft_blur_output),
                        Resource::Texture(&self.fft_real_part),
                        Resource::Texture(&self.fft_imag_part),
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(256, 1, 1),
                        workgroup_count: gpu::WorkgroupExtent::new(
                            self.fft_uniforms.output_width,
                            1,
                            1,
                        ),
                    }],
                }),
                Command::RenderPass(RenderParams {
                    target,
                    clear_color: Some([0., 1., 0., 1.]),
                    pipeline: Some(&self.render_pp),
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Sampler(&self.sampler),
                        Resource::Texture(&self.fft_blur_output),
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
        )
        .context("failed to render")
    }
}
