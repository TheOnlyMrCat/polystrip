use pollster::FutureExt;
use polystrip::{graph::RenderGraph, PolystripDevice, RenderPassTarget, Renderer};
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let el = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Slime")
        .with_resizable(false)
        .build(&el)
        .unwrap();

    let (width, height) = window.inner_size().into();
    let (wgpu_device, window) = unsafe {
        PolystripDevice::new_from_env_with_window(
            &window,
            (width, height),
            wgpu::PresentMode::AutoVsync,
        )
        .block_on()
    };

    let agent_count = 1000 * 128;
    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let mut renderer = Renderer::new(wgpu_device.device.clone(), wgpu_device.queue);
    let canvas = wgpu_device.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Primary Canvas"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
    });
    let canvas_view = canvas.create_view(&Default::default());
    let canvas_primary = renderer.insert_texture(canvas, canvas_view);
    let canvas = wgpu_device.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Swap Canvas"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
    });
    let canvas_view = canvas.create_view(&Default::default());
    let canvas_swap = renderer.insert_texture(canvas, canvas_view);

    let sampler = renderer.insert_sampler(
        wgpu_device
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default()),
    );

    let render_pipeline = renderer
        .add_render_pipeline_from_wgsl(include_str!("vertex.wgsl"))
        .build();

    let agent_pipeline = renderer
        .add_compute_pipeline_from_wgsl(include_str!("agents.wgsl"))
        .build();

    let canvas_pipeline = renderer
        .add_compute_pipeline_from_wgsl(include_str!("canvas.wgsl"))
        .build();

    let agent_buffer = renderer.insert_buffer(wgpu_device.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Agent buffer"),
            contents: bytemuck::cast_slice(&random_agents_outwards_circle(
                agent_count,
                (width as f32 / 2.0, height as f32 / 2.0),
                80.0,
            )),
            usage: wgpu::BufferUsages::STORAGE,
        },
    ));

    let time_buffer = renderer.insert_buffer(wgpu_device.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Time"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    ));

    el.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::MainEventsCleared => {
            let mut graph = RenderGraph::new(&mut renderer);
            let surface_texture = window.get_current_texture().unwrap();
            let surface_view = graph.add_temporary_texture_view(
                surface_texture.texture.create_view(&Default::default()),
            );
            graph
                .add_compute_node(canvas_pipeline)
                .with_bind_group((canvas_swap, canvas_primary))
                .build(|pass, [], ()| {
                    pass.dispatch_workgroups(width / 8, height / 8, 1);
                });
            graph
                .add_node()
                .with_external_output()
                .build(|encoder, renderer, [], [], [], ()| {
                    encoder.copy_texture_to_texture(
                        renderer
                            .get_texture(canvas_primary)
                            .0
                            .unwrap()
                            .as_image_copy(),
                        renderer.get_texture(canvas_swap).0.unwrap().as_image_copy(),
                        texture_size,
                    );
                });
            graph
                .add_compute_node(agent_pipeline)
                .with_bind_group((agent_buffer, time_buffer))
                .with_bind_group((canvas_primary, canvas_swap))
                .build(|pass, [], ()| {
                    pass.dispatch_workgroups((agent_count / 128) as _, 1, 1);
                });
            graph
                .add_render_node(render_pipeline)
                .with_bind_group((canvas_primary, sampler))
                .build(
                    RenderPassTarget::new().with_color(surface_view, wgpu::Color::BLACK),
                    |pass, [], ()| {
                        pass.draw(0..3, 0..1);
                    },
                );
            graph.execute();
            surface_texture.present();
        }
        _ => {}
    })
}

fn random_agents_outwards_circle(
    count: usize,
    (centre_x, centre_y): (f32, f32),
    radius: f32,
) -> Vec<[f32; 4]> {
    // Random within a circle, facing outwards
    use rand::Rng;
    let mut random = vec![0f32; count * 2];
    rand::thread_rng().fill(&mut random[..]);
    (0..count)
        .map(|i| {
            let dir = random[i * 2] * std::f32::consts::TAU;
            let rad = random[i * 2 + 1];
            [
                dir.cos() * rad * radius + centre_x,
                dir.sin() * rad * radius + centre_y,
                dir * std::f32::consts::TAU,
                0.0,
            ]
        })
        .collect()
}
