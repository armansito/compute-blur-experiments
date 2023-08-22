use {
    anyhow::{Context, Result},
    winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::Window,
    },
};

mod engine;
use crate::engine::Blurs;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 960;

async fn run(event_loop: EventLoop<()>, window: Window) -> Result<()> {
    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window) }?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .context("Failed to find an appropriate adapter")?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .context("Failed to initialize device")?;

    // Set up the swapchain
    let size = window.inner_size();
    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities
        .formats
        .into_iter()
        .find(|it| {
            matches!(
                it,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            )
        })
        .expect("supported swapchain formats: Rgba8Unorm or Bgra8Unorm");
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    let device = std::rc::Rc::new(device);
    let mut ctx = gpu::wgpu_context(device.clone(), queue);
    let mut example = Blurs::init(
        &ctx,
        include_str!("../shaders.wgsl"),
        swapchain_format.into(),
    )?;

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources. `event_loop.run` never returns,
        // therefore we must do this to ensure the resources are properly cleaned up.
        let _ = (&instance, &adapter);
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(size) => {
                    config.width = size.width;
                    config.height = size.height;
                    surface.configure(&device, &config);
                    // On macos the window needs to be redrawn manually after resizing
                    window.request_redraw();
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let render_target = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(e) = example.render_frame(
                    &mut ctx,
                    &gpu::TextureHandle::WgpuView(render_target),
                    move || {},
                ) {
                    println!("Failed to submit GPU commands: {}", e);
                }
                frame.present();
            }
            _ => (),
        }
    });
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new();
    let size = winit::dpi::LogicalSize::new(WIDTH, HEIGHT);
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(size)
        .with_title("wgpu Blurs".to_string())
        .build(&event_loop)?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window))?;
    }
    // TODO: wasm32
    Ok(())
}
