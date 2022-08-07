use pollster::FutureExt;
use polystrip::graph::RenderGraph;
use polystrip::{PolystripDevice, RenderPassTarget, TextureHandle};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() {
    let el = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Polystrip Basic Rendergraph")
        .build(&el)
        .unwrap();

    let (wgpu_device, mut window) = unsafe {
        PolystripDevice::new_from_env_with_window(
            &window,
            window.inner_size().into(),
            wgpu::PresentMode::AutoVsync,
        )
        .block_on()
    };

    let mut renderer = wgpu_device.create_renderer();
    let pipeline = create_pipeline(&wgpu_device.device);

    el.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(new_size),
            ..
        } => {
            window.resize(new_size.into());
        }
        Event::MainEventsCleared => {
            let mut graph = RenderGraph::new(&mut renderer);
            graph
                .add_node()
                .with_passthrough(&pipeline)
                .build_renderpass(
                    RenderPassTarget::new()
                        .with_color(TextureHandle::RENDER_TARGET, wgpu::Color::BLACK),
                    |pass, [], [], (pipeline,)| {
                        pass.set_pipeline(pipeline);
                        pass.draw(0..3, 0..1);
                    },
                );
            graph.execute(&mut window);
        }
        _ => {}
    })
}

const SHADER_SOURCE: &str = "\
@vertex
fn vert(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(vertex_index >> 1u) * 2.0 - 1.0;
    let y = f32(vertex_index & 1u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn frag(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let hue = acos(dot(position.xy, vec2<f32>(1.0, 1.0)) / (sqrt(2.0) * length(position.xy)));
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(vec3<f32>(hue, hue, hue) + K.xyz) * 6.0 - K.www) - K.xxx;
    //TODO: Re-vectorise this clamping when possible?
    return vec4<f32>(clamp(p.x, 0.0, 1.0), clamp(p.y, 0.0, 1.0), clamp(p.z, 0.0, 1.0), 1.0);
}
";

fn create_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "vert",
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: "frag",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
    })
}
