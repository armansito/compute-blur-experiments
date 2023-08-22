use {
    bytemuck::{Pod, Zeroable},
    gpu::command::*,
    instant::Instant,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    particle_count: u32,
    time_delta_ms: u32,
    total_time_ms: u32,
}

const MAX_FRAMES_IN_FLIGHT: u32 = 3;
const PARTICLE_COUNT: u32 = 300000;

// TODO: query the context for the uniform buffer alignment (256 on macOS, 16 on iOS)
const UNIFORMS_STRIDE: u32 = (std::mem::size_of::<Uniforms>() as u32 + 255) & !255;

const G_CENTERS: [f32; 6] = [-3., 3., 3., 3., 0., -4.24];

pub struct Blurs {
    render_pp: gpu::RenderPipelineState,
    init_particles_pp: gpu::ComputePipelineState,
    simulate_pp: gpu::ComputePipelineState,

    uniforms: Uniforms,
    uniforms_buffer: gpu::Buffer,
    uniforms_buffer_idx: u32,
    uniforms_buffer_offset: u32,

    particles_buffers: [gpu::Buffer; 2],
    g_centers_buffers: [gpu::Buffer; 2],
    buffer_swap_idx: usize,

    semaphore: gpu::Semaphore,
    time: Instant,
    needs_init: bool,
}

impl Blurs {
    pub fn init<A>(
        ctx: &gpu::Context<A>,
        shaders: &str,
        framebuffer_format: gpu::PixelFormat,
    ) -> Result<Self, gpu::Error>
    where
        A: gpu::Adapter,
    {
        let shader_module = ctx.new_shader_module(gpu::ShaderModuleDescriptor::Source(shaders))?;
        let render_pp = ctx.new_render_pipeline(&gpu::RenderPipelineDescriptor {
            label: None,
            topology: gpu::PrimitiveTopology::Triangle,
            layout: gpu::BindingLayout::IndexedByOrder(&[gpu::BindingType::Uniform]),
            vertex_stage: gpu::VertexStage {
                program: gpu::ProgrammableStage {
                    module: &shader_module,
                    entry_point: "vs_main",
                },
                buffers: Some(&[gpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: gpu::VertexStepMode::Instance,
                    attributes: &[
                        gpu::VertexAttribute {
                            format: gpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        gpu::VertexAttribute {
                            format: gpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }]),
            },
            fragment_stage: Some(gpu::FragmentStage {
                program: gpu::ProgrammableStage {
                    module: &shader_module,
                    entry_point: "fs_main",
                },
                color_targets: &[gpu::ColorTarget {
                    format: framebuffer_format,
                    blend: Some(gpu::Blend {
                        color: gpu::BlendComponent {
                            op: gpu::BlendOp::Add,
                            src_factor: gpu::BlendFactor::SrcAlpha,
                            dst_factor: gpu::BlendFactor::One,
                        },
                        alpha: gpu::BlendComponent {
                            op: gpu::BlendOp::Add,
                            src_factor: gpu::BlendFactor::One,
                            dst_factor: gpu::BlendFactor::One,
                        },
                    }),
                }],
            }),
        })?;
        let init_particles_pp = ctx.new_compute_pipeline(&gpu::ComputePipelineDescriptor {
            label: Some("init_particles"),
            program: gpu::ProgrammableStage {
                module: &shader_module,
                entry_point: "init_particles",
            },
            layout: gpu::BindingLayout::IndexedByOrder(&[
                gpu::BindingType::Uniform,
                gpu::BindingType::Storage,
            ]),
        })?;
        let simulate_pp = ctx.new_compute_pipeline(&gpu::ComputePipelineDescriptor {
            label: Some("simulate"),
            program: gpu::ProgrammableStage {
                module: &shader_module,
                entry_point: "simulate",
            },
            layout: gpu::BindingLayout::IndexedByOrder(&[
                gpu::BindingType::Uniform,
                gpu::BindingType::ReadOnlyStorage,
                gpu::BindingType::Storage,
                gpu::BindingType::Storage,
            ]),
        })?;

        let uniforms = Uniforms {
            particle_count: PARTICLE_COUNT,
            time_delta_ms: 0,
            total_time_ms: 0,
        };
        let uniforms_buffer = ctx.new_buffer(
            (UNIFORMS_STRIDE * MAX_FRAMES_IN_FLIGHT) as u64,
            gpu::BufferUsage::Uniform,
            gpu::AccessPattern::HostVisible,
        );

        let particles_buffers = [
            ctx.new_buffer(
                (4 * std::mem::size_of::<f32>() * PARTICLE_COUNT as usize) as u64,
                gpu::BufferUsage::VertexStorage,
                gpu::AccessPattern::GpuOnly,
            ),
            ctx.new_buffer(
                (4 * std::mem::size_of::<f32>() * PARTICLE_COUNT as usize) as u64,
                gpu::BufferUsage::VertexStorage,
                gpu::AccessPattern::GpuOnly,
            ),
        ];
        let g_centers_buffers = [
            ctx.new_buffer(
                (G_CENTERS.len() * std::mem::size_of::<f32>()) as u64,
                gpu::BufferUsage::Storage,
                gpu::AccessPattern::HostVisible,
            ),
            ctx.new_buffer(
                (G_CENTERS.len() * std::mem::size_of::<f32>()) as u64,
                gpu::BufferUsage::Storage,
                gpu::AccessPattern::GpuOnly,
            ),
        ];

        Ok(Blurs {
            render_pp,
            init_particles_pp,
            simulate_pp,
            uniforms,
            uniforms_buffer,
            uniforms_buffer_idx: 0,
            uniforms_buffer_offset: 0,
            particles_buffers,
            g_centers_buffers,
            buffer_swap_idx: 0,
            semaphore: gpu::Semaphore::new(MAX_FRAMES_IN_FLIGHT),
            time: Instant::now(),
            needs_init: true,
        })
    }

    pub fn render_frame<A, F>(
        &mut self,
        ctx: &mut gpu::Context<A>,
        target: &gpu::TextureHandle,
        present_drawable: F,
    ) -> Result<(), gpu::Error>
    where
        A: gpu::Adapter,
        F: Fn() + Send + 'static,
    {
        self.semaphore.acquire();

        self.uniforms_buffer_offset = UNIFORMS_STRIDE * self.uniforms_buffer_idx;
        self.uniforms_buffer_idx = (self.uniforms_buffer_idx + 1) % MAX_FRAMES_IN_FLIGHT;

        let now = Instant::now();
        self.uniforms.time_delta_ms = (now - self.time).as_millis() as u32;
        self.uniforms.total_time_ms += self.uniforms.time_delta_ms;
        self.time = now;

        let needs_init = self.needs_init;
        self.needs_init = false;

        let prev_idx = self.buffer_swap_idx;
        let next_idx = (self.buffer_swap_idx + 1) % 2;
        self.buffer_swap_idx = next_idx;

        let bg_color = {
            let hue = ((self.uniforms.total_time_ms as f64 * 0.001) % 10.) * 72.;
            peniko::Color::hlc(hue, 80., 80.)
        };

        let sem = self.semaphore.clone();
        let handlers = gpu::SubmitHandlers {
            scheduled: Some(Box::new(present_drawable)),
            completion: Some(Box::new(move || sem.release())),
        };
        ctx.submit(
            &[
                Command::WriteBuffer {
                    buffer: &self.uniforms_buffer,
                    data: bytemuck::bytes_of(&self.uniforms),
                    offset: self.uniforms_buffer_offset,
                },
                Command::Optional(
                    Some(&Command::WriteBuffer {
                        buffer: &self.g_centers_buffers[0],
                        data: bytemuck::bytes_of(&G_CENTERS),
                        offset: 0,
                    })
                    .filter(|_| needs_init),
                ),
                Command::Optional(
                    Some(&Command::ComputePass(ComputeParams {
                        pipeline: &self.init_particles_pp,
                        bindings: BindingList::IndexedByOrder(&[
                            Resource::Buffer {
                                offset: self.uniforms_buffer_offset,
                                buffer: &self.uniforms_buffer,
                            },
                            Resource::Buffer {
                                offset: 0,
                                buffer: &self.particles_buffers[prev_idx],
                            },
                        ]),
                        commands: &[ComputeCommand::Dispatch {
                            workgroup_size: gpu::WorkgroupExtent::new(256, 1, 1),
                            workgroup_count: gpu::WorkgroupExtent::new(
                                (PARTICLE_COUNT + 255) / 256,
                                1,
                                1,
                            ),
                        }],
                    }))
                    .filter(|_| needs_init),
                ),
                Command::ComputePass(ComputeParams {
                    pipeline: &self.simulate_pp,
                    bindings: BindingList::IndexedByOrder(&[
                        Resource::Buffer {
                            offset: self.uniforms_buffer_offset,
                            buffer: &self.uniforms_buffer,
                        },
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.particles_buffers[prev_idx],
                        },
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.particles_buffers[next_idx],
                        },
                        Resource::Buffer {
                            offset: 0,
                            buffer: &self.g_centers_buffers[0],
                        },
                    ]),
                    commands: &[ComputeCommand::Dispatch {
                        workgroup_size: gpu::WorkgroupExtent::new(256, 1, 1),
                        workgroup_count: gpu::WorkgroupExtent::new(
                            (PARTICLE_COUNT + 255) / 256,
                            1,
                            1,
                        ),
                    }],
                }),
                Command::RenderPass(RenderParams {
                    target,
                    clear_color: Some([
                        bg_color.r as f32 * 0.3 / 255.,
                        bg_color.g as f32 * 0.3 / 255.,
                        bg_color.b as f32 * 0.3 / 255.,
                        1.,
                    ]),
                    pipeline: Some(&self.render_pp),
                    bindings: BindingList::IndexedByOrder(&[Resource::Buffer {
                        offset: self.uniforms_buffer_offset,
                        buffer: &self.uniforms_buffer,
                    }]),
                    vertex_buffers: BindingList::IndexedByOrder(&[Resource::Buffer {
                        offset: 0,
                        buffer: &self.particles_buffers[prev_idx],
                    }]),
                    commands: &[RenderCommand::Draw {
                        vertex_count: 3,
                        instance_count: PARTICLE_COUNT,
                        first_vertex: 0,
                        first_instance: 0,
                    }],
                }),
            ],
            Some(handlers),
        )
    }
}
