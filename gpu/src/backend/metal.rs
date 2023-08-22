use crate::{
    backend::Adapter,
    command::{
        BindingList, Command, ComputeCommand, ComputeParams, RenderCommand, RenderParams, Resource,
    },
    pipeline::{
        BlendFactor, BlendOp, ComputePipelineDescriptor, ComputePipelineHandle,
        ComputePipelineState, PrimitiveTopology, RenderPipelineDescriptor, RenderPipelineHandle,
        RenderPipelineState, ShaderModule, ShaderModuleDescriptor, VertexFormat, VertexStepMode,
    },
    resource::{
        AccessPattern, AddressMode, Buffer, BufferHandle, BufferUsage, CompareFunction, FilterMode,
        PixelFormat, SamplerBorderColor, SamplerDescriptor, SamplerHandle, TextureDescriptor,
        TextureDimension, TextureHandle, TextureUsage,
    },
    Context, Error, SubmitHandlers,
};

pub fn new_context(device: &mtl::Device) -> Context<Backend> {
    Context {
        backend: Backend::new(device),
    }
}

pub struct Backend {
    device: mtl::Device,
    queue: mtl::CommandQueue,
}

impl Backend {
    pub fn new(device: &mtl::Device) -> Backend {
        Backend {
            device: device.clone(),
            queue: device.new_command_queue(),
        }
    }
}

impl Adapter for Backend {
    fn submit_commands(
        &mut self,
        commands: &[Command],
        handlers: Option<SubmitHandlers>,
    ) -> Result<(), Error> {
        let cmd_buf = CommandBuffer {
            backend: self,
            inner: self.queue.new_command_buffer().to_owned(),
        };

        if let Some(handlers) = handlers {
            if let Some(handler) = handlers.scheduled {
                cmd_buf.add_scheduled_handler(handler);
            }
            if let Some(handler) = handlers.completion {
                cmd_buf.add_completion_handler(handler);
            }
        }

        for cmd in commands.iter().filter_map(|cmd| match cmd {
            Command::Optional(cmd) => *cmd,
            others => Some(others),
        }) {
            match cmd {
                Command::WriteBuffer {
                    buffer,
                    data,
                    offset,
                } => cmd_buf.write_buffer(buffer, *offset, data)?,
                Command::RenderPass(params) => cmd_buf.encode_draws(&params)?,
                Command::ComputePass(params) => cmd_buf.encode_dispatches(&params),
                Command::Optional(_) => unreachable!(),
            }
        }

        cmd_buf.commit();
        Ok(())
    }

    fn new_buffer(&self, size: u64, usage: BufferUsage, access: AccessPattern) -> Buffer {
        let options = resource_options_for_buffer(&self.device, &access);
        Buffer {
            handle: BufferHandle::Metal(self.device.new_buffer(size.into(), options)),
            size,
            usage,
            access,
        }
    }

    fn new_buffer_with_data<T>(
        &self,
        data: &[T],
        usage: BufferUsage,
        access: AccessPattern,
    ) -> Buffer {
        let options = resource_options_for_buffer(&self.device, &access);
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        Buffer {
            handle: BufferHandle::Metal(self.device.new_buffer_with_data(
                data.as_ptr() as *const _,
                size,
                options,
            )),
            size,
            usage,
            access,
        }
    }

    fn new_texture(&self, descriptor: &TextureDescriptor) -> TextureHandle {
        let desc = mtl::TextureDescriptor::new();
        let (type_, w, h, d, array_len) = match descriptor.dimension {
            TextureDimension::D1 {
                width,
                array_layers,
            } => (
                if array_layers > 1 {
                    mtl::MTLTextureType::D1Array
                } else {
                    mtl::MTLTextureType::D1
                },
                width,
                1,
                1,
                array_layers,
            ),
            TextureDimension::D2 {
                width,
                height,
                array_layers,
            } => (
                if descriptor.sample_count > 1 {
                    if array_layers > 1 {
                        mtl::MTLTextureType::D2MultisampleArray
                    } else {
                        mtl::MTLTextureType::D2Multisample
                    }
                } else if array_layers > 1 {
                    mtl::MTLTextureType::D2Array
                } else {
                    mtl::MTLTextureType::D2
                },
                width,
                height,
                1,
                array_layers,
            ),
            TextureDimension::D3 {
                width,
                height,
                depth,
            } => (mtl::MTLTextureType::D3, width, height, depth, 1),
        };
        desc.set_texture_type(type_);
        desc.set_width(w.into());
        desc.set_height(h.into());
        desc.set_depth(d.into());
        desc.set_array_length(array_len.into());
        desc.set_mipmap_level_count(descriptor.mip_level_count.into());
        desc.set_sample_count(descriptor.sample_count.into());
        desc.set_pixel_format(descriptor.format.into());
        desc.set_usage(descriptor.usage.into());
        // TODO: set the storage mode based on GPU capabilities and a client-provided hint.
        desc.set_storage_mode(mtl::MTLStorageMode::Private);

        TextureHandle::Metal(self.device.new_texture(&desc))
    }

    fn new_sampler(&self, descriptor: &SamplerDescriptor) -> SamplerHandle {
        let desc = mtl::SamplerDescriptor::new();
        desc.set_address_mode_s(descriptor.address_mode.0.into());
        desc.set_address_mode_t(descriptor.address_mode.1.into());
        desc.set_address_mode_r(descriptor.address_mode.2.into());
        desc.set_min_filter(descriptor.min_filter.into());
        desc.set_mag_filter(descriptor.min_filter.into());
        desc.set_max_anisotropy(descriptor.max_anisotropy.into());
        match descriptor.mip_filter {
            Some(mip) => {
                desc.set_mip_filter(mip.filter.into());
                desc.set_lod_min_clamp(mip.min_lod_clamp);
                desc.set_lod_max_clamp(mip.max_lod_clamp);
            }
            None => {
                desc.set_mip_filter(mtl::MTLSamplerMipFilter::NotMipmapped);
            }
        }
        if let Some(compare) = descriptor.compare {
            desc.set_compare_function(compare.into());
        }
        if let Some(border) = descriptor.border_color {
            desc.set_border_color(border.into());
        }

        SamplerHandle::Metal(self.device.new_sampler(&desc))
    }

    fn new_shader_module(
        &self,
        descriptor: &ShaderModuleDescriptor,
    ) -> Result<ShaderModule, Error> {
        let ShaderModuleDescriptor::Source(src) = descriptor;
        let opts = mtl::CompileOptions::new();
        self.device
            .new_library_with_source(src, &opts)
            .map(ShaderModule::Metal)
            .map_err(Error::ShaderCompilation)
    }

    fn new_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor,
    ) -> Result<RenderPipelineState, Error> {
        let desc = mtl::RenderPipelineDescriptor::new();
        {
            let program = &descriptor.vertex_stage.program;
            let library = program.module.as_metal_ref();
            let vert_func = library
                .get_function(program.entry_point, None)
                .map_err(Error::ModuleEntryPoint)?;
            desc.set_vertex_function(Some(&vert_func));
        }
        if let Some(vert_buffers) = &descriptor.vertex_stage.buffers {
            let vert_desc = mtl::VertexDescriptor::new();
            let mut buffer_idx = 0;
            for buffer in vert_buffers.iter() {
                let layout_desc = mtl::VertexBufferLayoutDescriptor::new();
                layout_desc.set_stride(buffer.array_stride.into());
                layout_desc.set_step_function(match buffer.step_mode {
                    VertexStepMode::Vertex => mtl::MTLVertexStepFunction::PerVertex,
                    VertexStepMode::Instance => mtl::MTLVertexStepFunction::PerInstance,
                });
                for attr in buffer.attributes {
                    let attr_desc = mtl::VertexAttributeDescriptor::new();
                    attr_desc.set_format(match attr.format {
                        VertexFormat::Float32 => mtl::MTLVertexFormat::Float,
                        VertexFormat::Float32x2 => mtl::MTLVertexFormat::Float2,
                        VertexFormat::Float32x3 => mtl::MTLVertexFormat::Float3,
                        VertexFormat::Float32x4 => mtl::MTLVertexFormat::Float4,
                    });
                    attr_desc.set_offset(attr.offset.into());
                    attr_desc.set_buffer_index(buffer_idx);
                    vert_desc
                        .attributes()
                        .set_object_at(attr.shader_location.into(), Some(&attr_desc));
                }
                vert_desc
                    .layouts()
                    .set_object_at(buffer_idx, Some(&layout_desc));
                buffer_idx += 1;
            }
            desc.set_vertex_descriptor(Some(&vert_desc));
        }
        if let Some(frag_stage) = &descriptor.fragment_stage {
            let program = &frag_stage.program;
            let library = program.module.as_metal_ref();
            let frag_func = library
                .get_function(program.entry_point, None)
                .map_err(Error::ModuleEntryPoint)?;
            desc.set_fragment_function(Some(&frag_func));

            let mut idx = 0;
            for target in frag_stage.color_targets.iter() {
                let attachment =
                    desc.color_attachments()
                        .object_at(idx)
                        .ok_or(Error::PipelineCreation(format!(
                            "invalid color attachment index: {}",
                            idx
                        )))?;
                attachment.set_pixel_format((*target).format.into());
                if let Some(blend) = &(*target).blend {
                    attachment.set_blending_enabled(true);
                    attachment.set_rgb_blend_operation(blend.color.op.into());
                    attachment.set_alpha_blend_operation(blend.alpha.op.into());
                    attachment.set_source_rgb_blend_factor(blend.color.src_factor.into());
                    attachment.set_source_alpha_blend_factor(blend.alpha.src_factor.into());
                    attachment.set_destination_rgb_blend_factor(blend.color.dst_factor.into());
                    attachment.set_destination_alpha_blend_factor(blend.alpha.dst_factor.into());
                }
                idx += 1;
            }
        }
        self.device
            .new_render_pipeline_state(&desc)
            .map(|handle| RenderPipelineState {
                topology: descriptor.topology,
                handle: RenderPipelineHandle::Metal(handle),
            })
            .map_err(Error::PipelineCreation)
    }

    fn new_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<ComputePipelineState, Error> {
        let desc = mtl::ComputePipelineDescriptor::new();
        desc.set_thread_group_size_is_multiple_of_thread_execution_width(true);
        if let Some(label) = descriptor.label {
            desc.set_label(label);
        }
        let program = &descriptor.program;
        let library = program.module.as_metal_ref();
        let compute_func = library
            .get_function(program.entry_point, None)
            .map_err(Error::ModuleEntryPoint)?;
        desc.set_compute_function(Some(&compute_func));
        self.device
            .new_compute_pipeline_state(&desc)
            .map(ComputePipelineHandle::Metal)
            .map(|handle| ComputePipelineState { handle })
            .map_err(Error::PipelineCreation)
    }
}

struct CommandBuffer<'a> {
    backend: &'a Backend,
    inner: mtl::CommandBuffer,
}

impl<'a> CommandBuffer<'a> {
    fn add_completion_handler(&self, func: Box<dyn Fn()>) {
        let block = block::ConcreteBlock::new(move |_| func()).copy();
        self.inner.add_completed_handler(&block);
    }

    fn add_scheduled_handler(&self, func: Box<dyn Fn()>) {
        let block = block::ConcreteBlock::new(move |_| func()).copy();
        self.inner.add_scheduled_handler(&block);
    }

    fn write_buffer(&self, buffer: &Buffer, offset: u32, data: &[u8]) -> Result<(), Error> {
        if let AccessPattern::GpuOnly = buffer.access {
            return Err(Error::InvalidBufferUsage);
        }

        if buffer.size < offset as u64 + data.len() as u64 {
            return Err(Error::InvalidBufferSize(buffer.size));
        }

        let buf = buffer.handle.as_metal_ref();
        let ptr = buf.contents() as *mut u8;
        unsafe {
            std::ptr::copy(data.as_ptr(), ptr.offset(offset as isize), data.len());
        }

        if !self.backend.device.has_unified_memory() {
            buf.did_modify_range(mtl::NSRange::new(offset as u64, data.len() as u64));
        }

        Ok(())
    }

    fn encode_draws(&self, params: &RenderParams) -> Result<(), Error> {
        let descriptor = mtl::RenderPassDescriptor::new();

        let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
        color_attachment.set_store_action(mtl::MTLStoreAction::Store);
        match params.clear_color {
            Some(color) => {
                color_attachment.set_load_action(mtl::MTLLoadAction::Clear);
                color_attachment.set_clear_color(mtl::MTLClearColor::new(
                    color[0].into(),
                    color[1].into(),
                    color[2].into(),
                    color[3].into(),
                ));
            }
            None => {
                color_attachment.set_load_action(mtl::MTLLoadAction::Load);
            }
        };
        color_attachment.set_texture(Some(params.target.as_metal_ref()));

        let encoder = self.inner.new_render_command_encoder(&descriptor);

        encoder.set_cull_mode(mtl::MTLCullMode::Back);
        encoder.set_front_facing_winding(mtl::MTLWinding::CounterClockwise);

        // Process the pipeline and extract primitive state for the draw command.
        let topology = match params.pipeline {
            Some(pipeline) => {
                encoder.set_render_pipeline_state(pipeline.handle.as_metal_ref());
                Some(pipeline.topology)
            }
            None => None,
        };

        let mut buffer_idx = 0;
        match params.vertex_buffers {
            BindingList::Empty => {}
            BindingList::IndexedByOrder(buffers) => {
                for resource in buffers {
                    match *resource {
                        Resource::Buffer { offset, buffer } => {
                            let idx = buffer_idx;
                            buffer_idx += 1;
                            encoder.set_vertex_buffer(
                                idx,
                                Some(buffer.handle.as_metal_ref()),
                                offset.into(),
                            )
                        }
                        _ => panic!("Invalid vertex buffer resource type: {:?}", *resource),
                    }
                }
            }
        }
        match params.bindings {
            BindingList::Empty => {}
            BindingList::IndexedByOrder(bindings) => {
                let mut texture_idx = 0;
                let mut sampler_idx = 0;
                for resource in bindings {
                    match *resource {
                        Resource::Buffer { offset, buffer } => {
                            let idx = buffer_idx;
                            buffer_idx += 1;
                            encoder.set_vertex_buffer(
                                idx,
                                Some(buffer.handle.as_metal_ref()),
                                offset.into(),
                            );
                            encoder.set_fragment_buffer(
                                idx,
                                Some(buffer.handle.as_metal_ref()),
                                offset.into(),
                            );
                        }
                        Resource::Texture(handle) => {
                            let idx = texture_idx;
                            texture_idx += 1;
                            encoder.set_vertex_texture(idx, Some(handle.as_metal_ref()));
                            encoder.set_fragment_texture(idx, Some(handle.as_metal_ref()));
                        }
                        Resource::Sampler(handle) => {
                            let idx = sampler_idx;
                            sampler_idx += 1;
                            encoder.set_vertex_sampler_state(idx, Some(handle.as_metal_ref()));
                            encoder.set_fragment_sampler_state(idx, Some(handle.as_metal_ref()));
                        }
                    }
                }
            }
        }

        if !params.commands.is_empty() {
            let primitive: mtl::MTLPrimitiveType = match &topology {
                None => {
                    return Err(Error::InvalidRenderCommand(
                        "draw requires a pipeline".to_string(),
                    ))
                }
                Some(t) => (*t).into(),
            };
            for cmd in params.commands {
                match cmd {
                    &RenderCommand::Draw {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    } => {
                        if instance_count == 0 {
                            encoder.draw_primitives(
                                primitive,
                                first_vertex.into(),
                                vertex_count.into(),
                            );
                        } else {
                            encoder.draw_primitives_instanced_base_instance(
                                primitive,
                                first_vertex.into(),
                                vertex_count.into(),
                                instance_count.into(),
                                first_instance.into(),
                            );
                        }
                    }
                    &RenderCommand::DrawIndexed {
                        index_count: _,
                        instance_count: _,
                        first_index: _,
                        base_vertex: _,
                        first_instance: _,
                    } => unimplemented!(),
                    /*} => {
                        /* TODO: need to store an index buffer handle somewhere in the render pass
                         * descriptor to pass the to the API call.
                        if instance_count == 0 {
                            encoder.draw_indexed_primitives(
                                primitive,
                                index_count.into(),
                                mtl::MTLIndexType::UInt16,  // TODO: make this configurable via
                                                            // primitive state

                        }*/
                    }*/
                }
            }
        }

        encoder.end_encoding();

        Ok(())
    }

    fn encode_dispatches(&self, params: &ComputeParams) {
        let encoder = self.inner.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(params.pipeline.handle.as_metal_ref());
        match params.bindings {
            BindingList::Empty => {}
            BindingList::IndexedByOrder(bindings) => {
                let mut buffer_idx = 0;
                let mut texture_idx = 0;
                let mut sampler_idx = 0;
                for resource in bindings {
                    match *resource {
                        Resource::Buffer { offset, buffer } => {
                            let idx = buffer_idx;
                            buffer_idx += 1;
                            encoder.set_buffer(
                                idx,
                                Some(buffer.handle.as_metal_ref()),
                                offset.into(),
                            )
                        }
                        Resource::Texture(handle) => {
                            let idx = texture_idx;
                            texture_idx += 1;
                            encoder.set_texture(idx, Some(handle.as_metal_ref()))
                        }
                        Resource::Sampler(handle) => {
                            let idx = sampler_idx;
                            sampler_idx += 1;
                            encoder.set_sampler_state(idx, Some(handle.as_metal_ref()))
                        }
                    }
                }
            }
        }
        for cmd in params.commands {
            let ComputeCommand::Dispatch {
                workgroup_size,
                workgroup_count,
            } = cmd;
            encoder.dispatch_thread_groups(
                mtl::MTLSize::new(
                    workgroup_count.width.into(),
                    workgroup_count.height.into(),
                    workgroup_count.depth.into(),
                ),
                mtl::MTLSize::new(
                    workgroup_size.width.into(),
                    workgroup_size.height.into(),
                    workgroup_size.depth.into(),
                ),
            );
        }
        encoder.end_encoding()
    }

    fn commit(&self) {
        self.inner.commit()
    }
}

impl From<PixelFormat> for mtl::MTLPixelFormat {
    fn from(src: PixelFormat) -> Self {
        match src {
            PixelFormat::BGRA8Unorm => mtl::MTLPixelFormat::BGRA8Unorm,
            PixelFormat::BGRA8Unorm_sRGB => mtl::MTLPixelFormat::BGRA8Unorm_sRGB,
            PixelFormat::RGBA16Float => mtl::MTLPixelFormat::RGBA16Float,
            PixelFormat::RGBA32Float => mtl::MTLPixelFormat::RGBA32Float,
        }
    }
}

impl From<TextureUsage> for mtl::MTLTextureUsage {
    fn from(src: TextureUsage) -> Self {
        let mut dst = mtl::MTLTextureUsage::Unknown;
        if src.contains(TextureUsage::Sample) {
            dst |= mtl::MTLTextureUsage::ShaderRead;
        }
        if src.contains(TextureUsage::Storage) {
            dst |= mtl::MTLTextureUsage::ShaderWrite;
        }
        if src.contains(TextureUsage::RenderTarget) {
            dst |= mtl::MTLTextureUsage::RenderTarget;
        }
        dst
    }
}

impl From<PrimitiveTopology> for mtl::MTLPrimitiveType {
    fn from(src: PrimitiveTopology) -> Self {
        match src {
            PrimitiveTopology::Point => mtl::MTLPrimitiveType::Point,
            PrimitiveTopology::Line => mtl::MTLPrimitiveType::Line,
            PrimitiveTopology::LineStrip => mtl::MTLPrimitiveType::LineStrip,
            PrimitiveTopology::Triangle => mtl::MTLPrimitiveType::Triangle,
            PrimitiveTopology::TriangleStrip => mtl::MTLPrimitiveType::TriangleStrip,
        }
    }
}

impl From<BlendOp> for mtl::MTLBlendOperation {
    fn from(src: BlendOp) -> Self {
        match src {
            BlendOp::Add => mtl::MTLBlendOperation::Add,
            BlendOp::Subtract => mtl::MTLBlendOperation::Subtract,
            BlendOp::ReverseSubtract => mtl::MTLBlendOperation::ReverseSubtract,
            BlendOp::Min => mtl::MTLBlendOperation::Min,
            BlendOp::Max => mtl::MTLBlendOperation::Max,
        }
    }
}

impl From<BlendFactor> for mtl::MTLBlendFactor {
    fn from(src: BlendFactor) -> Self {
        match src {
            BlendFactor::Zero => mtl::MTLBlendFactor::Zero,
            BlendFactor::One => mtl::MTLBlendFactor::One,
            BlendFactor::Src => mtl::MTLBlendFactor::SourceColor,
            BlendFactor::OneMinusSrc => mtl::MTLBlendFactor::OneMinusSourceColor,
            BlendFactor::SrcAlpha => mtl::MTLBlendFactor::SourceAlpha,
            BlendFactor::OneMinusSrcAlpha => mtl::MTLBlendFactor::OneMinusSourceAlpha,
            BlendFactor::Dst => mtl::MTLBlendFactor::DestinationColor,
            BlendFactor::OneMinusDst => mtl::MTLBlendFactor::OneMinusDestinationColor,
            BlendFactor::DstAlpha => mtl::MTLBlendFactor::DestinationAlpha,
            BlendFactor::OneMinusDstAlpha => mtl::MTLBlendFactor::OneMinusDestinationAlpha,
            BlendFactor::SrcAlphaSaturated => mtl::MTLBlendFactor::SourceAlphaSaturated,
            BlendFactor::Constant => mtl::MTLBlendFactor::BlendColor,
            BlendFactor::OneMinusConstant => mtl::MTLBlendFactor::OneMinusBlendColor,
        }
    }
}

impl From<AddressMode> for mtl::MTLSamplerAddressMode {
    fn from(src: AddressMode) -> Self {
        match src {
            AddressMode::ClampToEdge => Self::ClampToEdge,
            AddressMode::Repeat => Self::Repeat,
            AddressMode::MirrorRepeat => Self::MirrorRepeat,
            AddressMode::ClampToBorder => Self::ClampToBorderColor,
        }
    }
}

impl From<FilterMode> for mtl::MTLSamplerMinMagFilter {
    fn from(src: FilterMode) -> Self {
        match src {
            FilterMode::Nearest => Self::Nearest,
            FilterMode::Linear => Self::Linear,
        }
    }
}

impl From<FilterMode> for mtl::MTLSamplerMipFilter {
    fn from(src: FilterMode) -> Self {
        match src {
            FilterMode::Nearest => Self::Nearest,
            FilterMode::Linear => Self::Linear,
        }
    }
}

impl From<CompareFunction> for mtl::MTLCompareFunction {
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

impl From<SamplerBorderColor> for mtl::MTLSamplerBorderColor {
    fn from(src: SamplerBorderColor) -> Self {
        match src {
            SamplerBorderColor::TransparentBlack => Self::TransparentBlack,
            SamplerBorderColor::OpaqueBlack => Self::OpaqueBlack,
            SamplerBorderColor::OpaqueWhite => Self::OpaqueWhite,
        }
    }
}

trait AsMetalRef<T> {
    fn as_metal_ref(&self) -> &T;
}

macro_rules! impl_as_metal_ref {
    ($type:tt, $metal_type:ty) => {
        impl AsMetalRef<$metal_type> for $type {
            fn as_metal_ref(&self) -> &$metal_type {
                match self {
                    $type::Metal(handle) => &handle,
                    unsupported => panic!("cannot be converted to metal type: {:?}", unsupported),
                }
            }
        }
    };
}

impl_as_metal_ref!(BufferHandle, mtl::BufferRef);
impl_as_metal_ref!(SamplerHandle, mtl::SamplerStateRef);
impl_as_metal_ref!(TextureHandle, mtl::TextureRef);
impl_as_metal_ref!(ShaderModule, mtl::LibraryRef);
impl_as_metal_ref!(RenderPipelineHandle, mtl::RenderPipelineStateRef);
impl_as_metal_ref!(ComputePipelineHandle, mtl::ComputePipelineStateRef);

fn resource_options_for_buffer(
    device: &mtl::Device,
    access: &AccessPattern,
) -> mtl::MTLResourceOptions {
    match *access {
        AccessPattern::HostVisible => {
            if device.has_unified_memory() {
                mtl::MTLResourceOptions::StorageModeShared
            } else {
                mtl::MTLResourceOptions::StorageModeManaged
            }
        }
        AccessPattern::GpuOnly => mtl::MTLResourceOptions::StorageModePrivate,
    }
}
