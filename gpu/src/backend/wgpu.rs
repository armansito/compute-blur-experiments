use crate::{
    backend::Adapter,
    command::{
        BindingList, Command, ComputeCommand, ComputeParams, RenderCommand, RenderParams, Resource,
    },
    pipeline::{
        BindingLayout, BindingType, BlendComponent, BlendFactor, BlendOp,
        ComputePipelineDescriptor, ComputePipelineHandle, ComputePipelineState, FillMode,
        PrimitiveTopology, RenderPipelineDescriptor, RenderPipelineHandle, RenderPipelineState,
        SamplerBindingType, ShaderModule, ShaderModuleDescriptor, StorageTextureAccess,
        VertexFormat, VertexStepMode,
    },
    resource::{
        AccessPattern, AddressMode, Buffer, BufferHandle, BufferUsage, CompareFunction, FilterMode,
        PixelFormat, SamplerBorderColor, SamplerDescriptor, SamplerHandle, TextureDescriptor,
        TextureDimension, TextureHandle, TextureUsage,
    },
    Context, Error, SubmitHandlers,
};
use std::{borrow::Cow, rc::Rc};

pub fn new_context(device: Rc<wgpu::Device>, queue: wgpu::Queue) -> Context<Backend> {
    Context {
        backend: Backend::new(device, queue),
    }
}

const STAGING_BUFFER_CHUNK_SIZE: u64 = 65536;

pub struct Backend {
    device: Rc<wgpu::Device>,
    queue: wgpu::Queue,
    staging_belt: wgpu::util::StagingBelt,
}

impl Backend {
    pub fn new(device: Rc<wgpu::Device>, queue: wgpu::Queue) -> Backend {
        Backend {
            device,
            queue,
            staging_belt: wgpu::util::StagingBelt::new(STAGING_BUFFER_CHUNK_SIZE),
        }
    }
}

impl Adapter for Backend {

    fn upload_texture(&self, texture: &TextureHandle, data: &[u8], width: u32, height: u32, bytes_per_row: u32) {
        let target = match &texture {
            TextureHandle::Wgpu { texture, view: _ } => texture,
            _ => panic!("expected a wgpu texture type"),
        };
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn submit_commands(
        &mut self,
        commands: &[Command],
        handlers: Option<SubmitHandlers>,
    ) -> Result<(), Error> {
        let inner = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let cmd_buf = {
            if let Some(handlers) = handlers {
                if let Some(handler) = handlers.scheduled {
                    handler();
                }
                if let Some(handler) = handlers.completion {
                    self.queue.on_submitted_work_done(handler);
                }
            }
            let mut encoder = CommandEncoder {
                backend: self,
                inner,
            };
            for cmd in commands.iter().filter_map(|cmd| match cmd {
                Command::Optional(cmd) => *cmd,
                others => Some(others),
            }) {
                match cmd {
                    Command::WriteBuffer {
                        buffer,
                        data,
                        offset,
                    } => encoder.write_buffer(buffer, *offset, data)?,
                    Command::RenderPass(params) => encoder.encode_draws(&params)?,
                    Command::ComputePass(params) => encoder.encode_dispatches(&params),
                    Command::Optional(_) => unreachable!(),
                }
            }
            encoder.inner.finish()
        };

        self.staging_belt.finish();
        self.queue.submit(Some(cmd_buf));
        self.staging_belt.recall();

        Ok(())
    }

    fn new_buffer(&self, size: u64, usage: BufferUsage, access: AccessPattern) -> Buffer {
        let desc = wgpu::BufferDescriptor {
            label: None,
            size,
            usage: usage_for_buffer(&usage),
            mapped_at_creation: false,
        };
        Buffer {
            handle: BufferHandle::Wgpu(self.device.create_buffer(&desc)),
            size,
            usage,
            access,
        }
    }

    fn new_buffer_with_data<T>(&self, _: &[T], _: BufferUsage, _: AccessPattern) -> Buffer {
        unimplemented!()
    }

    fn new_texture(&self, descriptor: &TextureDescriptor) -> TextureHandle {
        let (size, dimension) = match descriptor.dimension {
            TextureDimension::D1 {
                width,
                array_layers,
            } => (
                wgpu::Extent3d {
                    width,
                    height: 1,
                    depth_or_array_layers: array_layers,
                },
                wgpu::TextureDimension::D1,
            ),
            TextureDimension::D2 {
                width,
                height,
                array_layers,
            } => (
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: array_layers,
                },
                wgpu::TextureDimension::D2,
            ),
            TextureDimension::D3 {
                width,
                height,
                depth,
            } => (
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: depth,
                },
                wgpu::TextureDimension::D3,
            ),
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension,
            format: descriptor.format.into(),
            usage: (&descriptor.usage).into(),
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        TextureHandle::Wgpu { texture, view }
    }

    fn new_sampler(&self, descriptor: &SamplerDescriptor) -> SamplerHandle {
        let (mipmap_filter, lod_min_clamp, lod_max_clamp) = match descriptor.mip_filter {
            Some(mip) => (mip.filter.into(), mip.min_lod_clamp, mip.max_lod_clamp),
            None => (wgpu::FilterMode::Nearest, 0., 0.),
        };
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: descriptor.address_mode.0.into(),
            address_mode_v: descriptor.address_mode.1.into(),
            address_mode_w: descriptor.address_mode.2.into(),
            mag_filter: descriptor.mag_filter.into(),
            min_filter: descriptor.min_filter.into(),
            mipmap_filter,
            lod_min_clamp,
            lod_max_clamp,
            compare: descriptor.compare.map(wgpu::CompareFunction::from),
            anisotropy_clamp: descriptor.max_anisotropy,
            border_color: descriptor.border_color.map(wgpu::SamplerBorderColor::from),
        });
        SamplerHandle::Wgpu(sampler)
    }

    fn new_shader_module(
        &self,
        descriptor: &ShaderModuleDescriptor,
    ) -> Result<ShaderModule, Error> {
        let ShaderModuleDescriptor::Source(src) = descriptor;
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
            });
        Ok(ShaderModule::Wgpu(module))
    }

    fn new_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor,
    ) -> Result<RenderPipelineState, Error> {
        let mut vert_attrs = Vec::new();
        let mut attr_ranges = Vec::new();
        let vert_buffers = {
            let mut attr_start_idx = 0;
            descriptor
                .vertex_stage
                .buffers
                .map(|layouts| {
                    for layout in layouts {
                        for attr in layout.attributes {
                            vert_attrs.push(wgpu::VertexAttribute {
                                format: (&attr.format).into(),
                                offset: attr.offset.into(),
                                shader_location: attr.shader_location,
                            });
                        }
                        attr_ranges.push(std::ops::Range {
                            start: attr_start_idx,
                            end: vert_attrs.len(),
                        });
                        attr_start_idx = vert_attrs.len();
                    }
                    let mut result = Vec::with_capacity(layouts.len());
                    let mut layout_idx = 0;
                    for layout in layouts {
                        let attr_range = &attr_ranges[layout_idx];
                        result.push(wgpu::VertexBufferLayout {
                            array_stride: layout.array_stride.into(),
                            step_mode: match layout.step_mode {
                                VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
                                VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
                            },
                            attributes: &vert_attrs[attr_range.start..attr_range.end],
                        });
                        layout_idx += 1;
                    }
                    result
                })
                .unwrap_or_else(|| Vec::new())
        };
        let bindings = build_bind_group_layout_entries(
            &descriptor.layout,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
        let bind_group_layout = if bindings.is_empty() {
            None
        } else {
            let desc = wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &bindings,
            };
            Some(self.device.create_bind_group_layout(&desc))
        };
        let pipeline_layout = {
            let layouts = match bind_group_layout.as_ref() {
                Some(layout) => vec![layout],
                None => vec![],
            };
            let desc = wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &layouts,
                push_constant_ranges: &[],
            };
            self.device.create_pipeline_layout(&desc)
        };
        let mut desc = wgpu::RenderPipelineDescriptor {
            label: descriptor.label,
            layout: Some(&pipeline_layout),
            vertex: {
                let program = &descriptor.vertex_stage.program;
                wgpu::VertexState {
                    module: program.module.as_wgpu_ref(),
                    entry_point: program.entry_point,
                    buffers: &vert_buffers,
                }
            },
            primitive: wgpu::PrimitiveState {
                topology: descriptor.topology.into(),
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: descriptor.fill_mode.into(),
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview: None,
        };
        let mut color_targets = Vec::new();
        if let Some(fs) = &descriptor.fragment_stage {
            color_targets.reserve(fs.color_targets.len());
            for target in fs.color_targets.iter() {
                color_targets.push(Some(wgpu::ColorTargetState {
                    format: (*target).format.into(),
                    blend: (*target).blend.as_ref().map(|b| wgpu::BlendState {
                        color: (&b.color).into(),
                        alpha: (&b.alpha).into(),
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }));
            }
            desc.fragment = Some(wgpu::FragmentState {
                module: fs.program.module.as_wgpu_ref(),
                entry_point: fs.program.entry_point,
                targets: &color_targets,
            });
        }
        Ok(RenderPipelineState {
            handle: RenderPipelineHandle::Wgpu {
                inner: self.device.create_render_pipeline(&desc),
                bind_group_layout,
                pipeline_layout,
            },
            topology: descriptor.topology,
            fill_mode: descriptor.fill_mode,
        })
    }

    fn new_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<ComputePipelineState, Error> {
        let program = &descriptor.program;
        let bindings =
            build_bind_group_layout_entries(&descriptor.layout, wgpu::ShaderStages::COMPUTE);
        let bind_group_layout = {
            let desc = wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &bindings,
            };
            self.device.create_bind_group_layout(&desc)
        };
        let pipeline_layout = {
            let desc = wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            };
            self.device.create_pipeline_layout(&desc)
        };
        let desc = wgpu::ComputePipelineDescriptor {
            label: descriptor.label,
            layout: Some(&pipeline_layout),
            module: &program.module.as_wgpu_ref(),
            entry_point: program.entry_point,
        };
        Ok(ComputePipelineState {
            handle: ComputePipelineHandle::Wgpu {
                inner: self.device.create_compute_pipeline(&desc),
                bind_group_layout,
                pipeline_layout,
            },
        })
    }
}

struct CommandEncoder<'a> {
    backend: &'a mut Backend,
    inner: wgpu::CommandEncoder,
}

impl<'a> CommandEncoder<'a> {
    fn write_buffer(&mut self, buffer: &Buffer, offset: u32, data: &[u8]) -> Result<(), Error> {
        let len: u64 = data.len().try_into().unwrap();
        let mut view = self.backend.staging_belt.write_buffer(
            &mut self.inner,
            buffer.handle.as_wgpu_ref(),
            offset.into(),
            std::num::NonZeroU64::new(len).unwrap(),
            &self.backend.device,
        );
        view.copy_from_slice(data);
        Ok(())
    }

    fn encode_draws(&mut self, params: &RenderParams) -> Result<(), Error> {
        let target_view_ref = match &params.target {
            TextureHandle::WgpuView(view) => view,
            TextureHandle::Wgpu { texture: _, view } => view,
            _ => panic!("expected a wgpu texture type"),
        };
        let mut bind_group = None;
        let mut render_pass = self.inner.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view_ref,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: match params.clear_color {
                        None => wgpu::LoadOp::Load,
                        Some(color) => wgpu::LoadOp::Clear(wgpu::Color {
                            r: color[0].into(),
                            g: color[1].into(),
                            b: color[2].into(),
                            a: color[3].into(),
                        }),
                    },
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        match params.vertex_buffers {
            BindingList::Empty => {}
            BindingList::IndexedByOrder(bindings) => {
                for (idx, resource) in bindings.iter().enumerate() {
                    match *resource {
                        Resource::Buffer { offset, buffer } => match buffer.usage {
                            BufferUsage::Vertex | BufferUsage::VertexStorage => render_pass
                                .set_vertex_buffer(
                                    idx as u32,
                                    buffer.handle.as_wgpu_ref().slice(offset as u64..),
                                ),
                            _ => panic!("Invalid vertex buffer usage: {:?}", buffer.usage),
                        },
                        _ => panic!("Invalid vertex buffer resource type: {:?}", *resource),
                    }
                }
            }
        }

        if let Some(pipeline) = params.pipeline {
            let RenderPipelineHandle::Wgpu {
                inner: handle,
                bind_group_layout,
                pipeline_layout: _,
            } = &pipeline.handle
            else {
                panic!("cannot be converted to wgpu type: {:?}", pipeline.handle);
            };
            render_pass.set_pipeline(handle);
            bind_group = match &bind_group_layout {
                Some(layout) => create_bind_group(&self.backend.device, &layout, &params.bindings),
                None => None,
            };
        }

        if let Some(bind_group) = bind_group.as_ref() {
            render_pass.set_bind_group(0, &bind_group, &[]);
        }

        for cmd in params.commands {
            match cmd {
                &RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    render_pass.draw(
                        first_vertex..(first_vertex + vertex_count),
                        first_instance..(first_instance + instance_count.max(1)),
                    );
                }
                &RenderCommand::DrawIndexed {
                    index_buffer,
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    let index_buffer_offset =
                        first_index as u64 * std::mem::size_of::<u32>() as u64;
                    let index_buffer_len = index_count as u64 * std::mem::size_of::<u32>() as u64;
                    render_pass.set_index_buffer(
                        index_buffer
                            .handle
                            .as_wgpu_ref()
                            .slice(index_buffer_offset..index_buffer_offset + index_buffer_len),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(
                        0..index_count,
                        base_vertex,
                        first_instance..first_instance + instance_count,
                    );
                }
            }
        }

        drop(render_pass);

        Ok(())
    }

    fn encode_dispatches(&mut self, params: &ComputeParams) {
        let ComputePipelineHandle::Wgpu {
            inner: handle,
            bind_group_layout,
            pipeline_layout: _,
        } = &params.pipeline.handle
        else {
            panic!(
                "cannot be converted to wgpu type: {:?}",
                params.pipeline.handle
            );
        };
        let bind_group =
            create_bind_group(&self.backend.device, &bind_group_layout, &params.bindings);
        let mut compute_pass = self
            .inner
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_pipeline(handle);

        if let Some(bind_group) = bind_group.as_ref() {
            compute_pass.set_bind_group(0, &bind_group, &[]);
        }

        for cmd in params.commands {
            let ComputeCommand::Dispatch {
                workgroup_size: _,
                workgroup_count,
            } = cmd;
            compute_pass.dispatch_workgroups(
                workgroup_count.width,
                workgroup_count.height,
                workgroup_count.depth,
            );
        }
    }
}

trait AsWgpuRef<T> {
    fn as_wgpu_ref(&self) -> &T;
}

macro_rules! impl_as_wgpu_ref {
    ($type:tt, $wgpu_type:ty) => {
        impl AsWgpuRef<$wgpu_type> for $type {
            fn as_wgpu_ref(&self) -> &$wgpu_type {
                match self {
                    $type::Wgpu(handle) => &handle,
                    unsupported => panic!("cannot be converted to wgpu type: {:?}", unsupported),
                }
            }
        }
    };
}

impl_as_wgpu_ref!(BufferHandle, wgpu::Buffer);
impl_as_wgpu_ref!(SamplerHandle, wgpu::Sampler);
impl_as_wgpu_ref!(ShaderModule, wgpu::ShaderModule);

fn usage_for_buffer(usage: &BufferUsage) -> wgpu::BufferUsages {
    let mut usages = wgpu::BufferUsages::empty();
    match *usage {
        BufferUsage::Vertex => {
            usages.toggle(wgpu::BufferUsages::VERTEX);
            usages.toggle(wgpu::BufferUsages::COPY_DST);
        }
        BufferUsage::VertexStorage => {
            usages.toggle(wgpu::BufferUsages::VERTEX);
            usages.toggle(wgpu::BufferUsages::STORAGE);
            usages.toggle(wgpu::BufferUsages::COPY_DST);
        }
        BufferUsage::Index => {
            usages.toggle(wgpu::BufferUsages::INDEX);
            usages.toggle(wgpu::BufferUsages::COPY_DST);
        }
        BufferUsage::IndexStorage => {
            usages.toggle(wgpu::BufferUsages::INDEX);
            usages.toggle(wgpu::BufferUsages::STORAGE);
            usages.toggle(wgpu::BufferUsages::COPY_DST);
        }
        BufferUsage::Uniform => {
            usages.toggle(wgpu::BufferUsages::UNIFORM);
            usages.toggle(wgpu::BufferUsages::COPY_DST);
        }
        BufferUsage::Storage => {
            usages.toggle(wgpu::BufferUsages::STORAGE);
            usages.toggle(wgpu::BufferUsages::COPY_DST);
        }
    }
    usages
}

fn build_bind_group_layout_entries(
    layout: &BindingLayout,
    visibility: wgpu::ShaderStages,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    match &layout {
        BindingLayout::Empty => Vec::new(),
        BindingLayout::IndexedByOrder(entries) => entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| wgpu::BindGroupLayoutEntry {
                binding: idx as u32,
                visibility,
                ty: match *entry {
                    BindingType::Uniform => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingType::Storage | BindingType::ReadOnlyStorage => {
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: *entry == BindingType::ReadOnlyStorage,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        }
                    }
                    BindingType::StorageTexture { access, format } => {
                        wgpu::BindingType::StorageTexture {
                            access: access.into(),
                            format: format.into(),
                            view_dimension: wgpu::TextureViewDimension::D2,
                        }
                    }
                    BindingType::Texture { filterable } => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    BindingType::Sampler(s) => wgpu::BindingType::Sampler(s.into()),
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    }
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bindings: &BindingList,
) -> Option<wgpu::BindGroup> {
    match bindings {
        BindingList::IndexedByOrder(bindings) => {
            let entries = bindings
                .iter()
                .enumerate()
                .map(|(idx, resource)| wgpu::BindGroupEntry {
                    binding: idx as u32,
                    resource: match *resource {
                        Resource::Buffer { offset, buffer } => {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: buffer.handle.as_wgpu_ref(),
                                offset: offset.into(),
                                size: None,
                            })
                        }
                        Resource::Texture(handle) => {
                            let view = match handle {
                                TextureHandle::WgpuView(view) => view,
                                TextureHandle::Wgpu { texture: _, view } => view,
                                unsupported => {
                                    panic!("cannot be converted to wgpu type: {:?}", unsupported)
                                }
                            };
                            wgpu::BindingResource::TextureView(view)
                        }
                        Resource::Sampler(handle) => {
                            wgpu::BindingResource::Sampler(handle.as_wgpu_ref())
                        }
                    },
                })
                .collect::<Vec<_>>();
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &entries,
            }))
        }
        BindingList::Empty => None,
    }
}

impl From<&BlendComponent> for wgpu::BlendComponent {
    fn from(src: &BlendComponent) -> Self {
        wgpu::BlendComponent {
            src_factor: src.src_factor.into(),
            dst_factor: src.dst_factor.into(),
            operation: src.op.into(),
        }
    }
}

impl From<BlendFactor> for wgpu::BlendFactor {
    fn from(src: BlendFactor) -> Self {
        match src {
            BlendFactor::Zero => wgpu::BlendFactor::Zero,
            BlendFactor::One => wgpu::BlendFactor::One,
            BlendFactor::Src => wgpu::BlendFactor::Src,
            BlendFactor::OneMinusSrc => wgpu::BlendFactor::OneMinusSrc,
            BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
            BlendFactor::Dst => wgpu::BlendFactor::Dst,
            BlendFactor::OneMinusDst => wgpu::BlendFactor::OneMinusDst,
            BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
            BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
            BlendFactor::SrcAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
            BlendFactor::Constant => wgpu::BlendFactor::Constant,
            BlendFactor::OneMinusConstant => wgpu::BlendFactor::OneMinusConstant,
        }
    }
}

impl From<AddressMode> for wgpu::AddressMode {
    fn from(src: AddressMode) -> Self {
        match src {
            AddressMode::ClampToEdge => Self::ClampToEdge,
            AddressMode::Repeat => Self::Repeat,
            AddressMode::MirrorRepeat => Self::MirrorRepeat,
            AddressMode::ClampToBorder => Self::ClampToBorder,
        }
    }
}

impl From<FilterMode> for wgpu::FilterMode {
    fn from(src: FilterMode) -> Self {
        match src {
            FilterMode::Nearest => Self::Nearest,
            FilterMode::Linear => Self::Linear,
        }
    }
}

impl From<CompareFunction> for wgpu::CompareFunction {
    fn from(src: CompareFunction) -> Self {
        match src {
            CompareFunction::Never => Self::Never,
            CompareFunction::Less => Self::Less,
            CompareFunction::Equal => Self::Equal,
            CompareFunction::LessEqual => Self::LessEqual,
            CompareFunction::Greater => Self::Greater,
            CompareFunction::NotEqual => Self::NotEqual,
            CompareFunction::GreaterEqual => Self::GreaterEqual,
            CompareFunction::Always => Self::Always,
        }
    }
}

impl From<SamplerBindingType> for wgpu::SamplerBindingType {
    fn from(src: SamplerBindingType) -> Self {
        match src {
            SamplerBindingType::Filtering => Self::Filtering,
            SamplerBindingType::NonFiltering => Self::NonFiltering,
            SamplerBindingType::Comparison => Self::Comparison,
        }
    }
}

impl From<SamplerBorderColor> for wgpu::SamplerBorderColor {
    fn from(src: SamplerBorderColor) -> Self {
        match src {
            SamplerBorderColor::TransparentBlack => Self::TransparentBlack,
            SamplerBorderColor::OpaqueBlack => Self::OpaqueBlack,
            SamplerBorderColor::OpaqueWhite => Self::OpaqueWhite,
        }
    }
}

impl From<BlendOp> for wgpu::BlendOperation {
    fn from(src: BlendOp) -> Self {
        match src {
            BlendOp::Add => wgpu::BlendOperation::Add,
            BlendOp::Subtract => wgpu::BlendOperation::Subtract,
            BlendOp::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
            BlendOp::Min => wgpu::BlendOperation::Min,
            BlendOp::Max => wgpu::BlendOperation::Max,
        }
    }
}

impl From<PixelFormat> for wgpu::TextureFormat {
    fn from(src: PixelFormat) -> Self {
        match src {
            PixelFormat::BGRA8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            PixelFormat::RGBA8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            PixelFormat::BGRA8Unorm_sRGB => wgpu::TextureFormat::Bgra8UnormSrgb,
            PixelFormat::RGBA16Float => wgpu::TextureFormat::Rgba16Float,
            PixelFormat::RGBA32Float => wgpu::TextureFormat::Rgba32Float,
        }
    }
}

impl From<PrimitiveTopology> for wgpu::PrimitiveTopology {
    fn from(src: PrimitiveTopology) -> Self {
        match src {
            PrimitiveTopology::Point => wgpu::PrimitiveTopology::PointList,
            PrimitiveTopology::Line => wgpu::PrimitiveTopology::LineList,
            PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
            PrimitiveTopology::Triangle => wgpu::PrimitiveTopology::TriangleList,
            PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        }
    }
}

impl From<FillMode> for wgpu::PolygonMode {
    fn from(src: FillMode) -> Self {
        match src {
            FillMode::Fill => wgpu::PolygonMode::Fill,
            FillMode::Lines => wgpu::PolygonMode::Line,
        }
    }
}

impl From<StorageTextureAccess> for wgpu::StorageTextureAccess {
    fn from(src: StorageTextureAccess) -> Self {
        match src {
            StorageTextureAccess::ReadOnly => Self::ReadOnly,
            StorageTextureAccess::WriteOnly => Self::WriteOnly,
            StorageTextureAccess::ReadWrite => Self::ReadWrite,
        }
    }
}

impl From<&TextureUsage> for wgpu::TextureUsages {
    fn from(src: &TextureUsage) -> Self {
        let mut dst = wgpu::TextureUsages::empty();
        if src.contains(TextureUsage::CopySrc) {
            dst |= wgpu::TextureUsages::COPY_SRC;
        }
        if src.contains(TextureUsage::CopyDst) {
            dst |= wgpu::TextureUsages::COPY_DST;
        }
        if src.contains(TextureUsage::Sample) {
            dst |= wgpu::TextureUsages::TEXTURE_BINDING;
        }
        if src.contains(TextureUsage::Storage) {
            dst |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        if src.contains(TextureUsage::RenderTarget) {
            dst |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        dst
    }
}

impl From<&VertexFormat> for wgpu::VertexFormat {
    fn from(src: &VertexFormat) -> Self {
        match src {
            VertexFormat::Float32 => wgpu::VertexFormat::Float32,
            VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
            VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
            VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        }
    }
}
