use common::Example;
use {
    anyhow::{Context, Result},
    cocoa::{appkit::NSView, base::id as cocoa_id},
    core_graphics_types::geometry::CGSize,
    mtl::{Device, MTLPixelFormat, MetalLayer},
    objc::{rc::autoreleasepool, runtime::YES},
    winit::{
        event::{Event, WindowEvent},
        event_loop::ControlFlow,
        platform::macos::WindowExtMacOS,
    },
};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 960;

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::new();
    let size = winit::dpi::LogicalSize::new(WIDTH, HEIGHT);

    let device = Device::system_default().context("A Metal GPU device not found")?;
    let layer = MetalLayer::new();
    layer.set_device(&device);
    layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    layer.set_presents_with_transaction(false);

    let window = winit::window::WindowBuilder::new()
        .with_inner_size(size)
        .with_title("Metal Renderer".to_string())
        .build(&event_loop)?;

    unsafe {
        let view = window.ns_view() as cocoa_id;
        view.setWantsLayer(YES);
        view.setLayer(std::mem::transmute(layer.as_ref()));
    }

    let draw_size = window.inner_size();
    layer.set_drawable_size(CGSize::new(draw_size.width as f64, draw_size.height as f64));

    let mut ctx = gpu::metal_context(&device);
    let mut example = Example::init(
        &ctx,
        include_str!("../shaders.metal"),
        gpu::PixelFormat::BGRA8Unorm,
    )?;

    event_loop.run(move |event, _, control_flow| {
        autoreleasepool(|| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(size) => {
                        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
                    }
                    _ => (),
                },
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    let drawable = match layer.next_drawable() {
                        Some(drawable) => drawable.to_owned(),
                        None => return,
                    };
                    let render_target = drawable.texture().to_owned();
                    if let Err(e) = example.render_frame(
                        &mut ctx,
                        &gpu::TextureHandle::Metal(render_target),
                        move || drawable.present(),
                    ) {
                        println!("Failed to submit GPU commands: {}", e);
                    }
                }
                _ => (),
            }
        });
    });
}
