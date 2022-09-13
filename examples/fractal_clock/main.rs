use std::sync::Arc;

use pollster::FutureExt;
use polystrip::graph::RenderGraph;
use polystrip::{
    ComputePipeline, Handle, PolystripDevice, RenderPassTarget, RenderPipeline, Renderer,
};
use time::{OffsetDateTime, UtcOffset};
use wgpu::util::DeviceExt;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() {
    let el = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Polystrip Fractal Clock")
        .build(&el)
        .unwrap();
    let window_size = window.inner_size();

    let (wgpu_device, mut window) = unsafe {
        PolystripDevice::new_from_env_with_window(
            &window,
            window_size.into(),
            wgpu::PresentMode::AutoVsync,
        )
        .block_on()
    };

    let mut depth = 4;
    let mut renderer = wgpu_device.create_renderer();
    let mut pipelines = Pipelines::new(
        &mut renderer,
        wgpu_device.device.clone(),
        wgpu_device.queue.clone(),
        window_size.into(),
        depth,
    );

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
            pipelines.resize(new_size.into(), &renderer);
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { input, .. },
            ..
        } => {
            if input.state == ElementState::Pressed {
                match input.virtual_keycode {
                    Some(VirtualKeyCode::Up) => depth += 1,
                    Some(VirtualKeyCode::Down) if depth > 1 => depth -= 1,
                    _ => (),
                }
                pipelines.set_depth(depth, &mut renderer);
            }
        }

        Event::MainEventsCleared => {
            let mut graph = RenderGraph::new(&mut renderer);
            let surface_texture = window.get_current_texture().unwrap();
            let surface_view = graph.add_temporary_texture_view(
                surface_texture.texture.create_view(&Default::default()),
            );
            pipelines.render_to(&mut graph, surface_view);
            graph.execute();
            surface_texture.present();
        }
        _ => {}
    })
}

struct Pipelines {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    index_buffer: Handle<wgpu::Buffer>,
    camera_buffer: Handle<wgpu::Buffer>,
    clock_buffer: Handle<wgpu::Buffer>,

    fractal_render_pipeline: Handle<RenderPipeline>,
    clock_render_pipeline: Handle<RenderPipeline>,
    vertices_compute_pipeline: Handle<ComputePipeline>,
    indices_compute_pipeline: Handle<ComputePipeline>,

    width: u32,
    height: u32,
    fractal_depth: u32,
}

impl Pipelines {
    fn new(
        renderer: &mut Renderer,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        (width, height): (u32, u32),
        fractal_depth: u32,
    ) -> Self {
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FC Index Buffer"),
            size: vertex_count_for_depth(fractal_depth) * vertex_size() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let index_buffer = renderer.insert_buffer(index_buffer);

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FC Camera Buffer"),
            size: std::mem::size_of::<[f64; 16]>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let camera_buffer = renderer.insert_buffer(camera_buffer);

        let clock_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FC Clock Vertex Buffer"),
            size: std::mem::size_of::<ClockRay>() as u64 * 75,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let clock_buffer = renderer.insert_buffer(clock_buffer);
        renderer.write_buffer(
            clock_buffer,
            std::mem::size_of::<ClockRay>() as u64 * 3,
            bytemuck::cast_slice(
                &(0..12)
                    .flat_map(|i| {
                        std::iter::once(ClockRay {
                            start: 0.45,
                            end: 0.5,
                            direction: (i as f32 * std::f32::consts::TAU / 12.0),
                            half_thickness: 0.004784689,
                        })
                        .chain((0..5).map(move |j| ClockRay {
                            start: 0.475,
                            end: 0.5,
                            direction: (i as f32 * std::f32::consts::TAU / 12.0)
                                + (j as f32 * std::f32::consts::TAU / 60.0),
                            half_thickness: 0.0023923445,
                        }))
                    })
                    .collect::<Vec<_>>(),
            ),
        );

        let fractal_render_pipeline = renderer
            .add_render_pipeline_from_wgsl(include_str!("fractal.wgsl"))
            .with_primitive_topology(wgpu::PrimitiveTopology::LineList)
            .with_depth_stencil(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
            .with_msaa(4)
            .build();

        let clock_render_pipeline = renderer
            .add_render_pipeline_from_wgsl(include_str!("clock.wgsl"))
            .with_vertex_step_mode(wgpu::VertexStepMode::Instance)
            .with_primitive_topology(wgpu::PrimitiveTopology::TriangleStrip)
            .with_depth_stencil(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
            .with_msaa(4)
            .build();

        let vertices_compute_pipeline = renderer
            .add_compute_pipeline_from_wgsl(include_str!("vertices.wgsl"))
            .build();
        let indices_compute_pipeline = renderer
            .add_compute_pipeline_from_wgsl(include_str!("indices.wgsl"))
            .build();

        let mut this = Self {
            device,
            queue,
            index_buffer,
            camera_buffer,
            clock_buffer,
            fractal_render_pipeline,
            clock_render_pipeline,
            vertices_compute_pipeline,
            indices_compute_pipeline,
            width,
            height,
            fractal_depth,
        };

        this.prepare_indices(renderer);
        this.resize((width, height), renderer);
        this
    }

    fn prepare_indices(&self, renderer: &mut Renderer) {
        let workgroup_size = workgroup_size_for_depth(self.fractal_depth) as u32;
        let workgroup_y = workgroup_size / 256 + 1;
        let workgroup_x = workgroup_size % 256 + 1;

        let indices_compute_pipeline = renderer.get_compute_pipeline(self.indices_compute_pipeline);
        //TODO: More offsets required? Consider push constants if on native?
        let null_offset = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Null offset"),
                contents: &0u32.to_ne_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let null_offset_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Null offset bind group"),
            layout: renderer.get_bind_group_layout(indices_compute_pipeline.bind_group_layouts[1]),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(null_offset.as_entire_buffer_binding()),
            }],
        });
        let index_bind_group_layout = indices_compute_pipeline.bind_group_layouts[0];
        let index_bind_group_handle =
            renderer.add_bind_group(index_bind_group_layout, (self.index_buffer,));
        let index_bind_group = renderer.get_bind_group(index_bind_group_handle);

        let indices_compute_pipeline = renderer.get_compute_pipeline(self.indices_compute_pipeline);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FC Indices Compute Encoder"),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FC Indices Compute Pass"),
        });
        pass.set_pipeline(&indices_compute_pipeline.pipeline);
        pass.set_bind_group(0, index_bind_group, &[]);
        pass.set_bind_group(1, &null_offset_bind_group, &[]);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        drop(pass);

        self.queue.submit([encoder.finish()]);
    }

    pub fn set_depth(&mut self, fractal_depth: u32, renderer: &mut Renderer) {
        let fractal_depth = fractal_depth.max(1);
        self.fractal_depth = fractal_depth;

        let index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FC Index Buffer"),
            size: vertex_count_for_depth(fractal_depth) * vertex_size() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.index_buffer = renderer.insert_buffer(index_buffer);

        self.prepare_indices(renderer);
    }

    pub fn resize(&mut self, (width, height): (u32, u32), renderer: &Renderer) {
        self.width = width;
        self.height = height;

        let matrix = if width > height {
            let aspect = width as f32 / height as f32;
            [
                [1.0 / (1.35 * aspect), 0.0, 0.0, 0.0],
                [0.0, 1.0 / 1.35, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        } else {
            let aspect = height as f32 / width as f32;
            [
                [1.0 / 1.35, 0.0, 0.0, 0.0],
                [0.0, 1.0 / (1.35 * aspect), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        };
        renderer.write_buffer(self.camera_buffer, 0, bytemuck::cast_slice(&matrix));
    }

    fn render_to<'node>(
        &'node self,
        graph: &mut RenderGraph<'_, 'node>,
        target: Handle<wgpu::TextureView>,
    ) {
        let (hr, min, sec, ms) = OffsetDateTime::now_utc()
            .to_offset(local_timezone())
            .to_hms_milli();

        let second_frac = (sec as f32 / 60.0) + (ms as f32 / 60_000.0);
        let minute_frac = (min as f32 / 60.0) + (second_frac / 60.0);
        let hour_frac = (hr as f32 / 12.0) + ((minute_frac / 60.0) * (60.0 / 12.0));

        // Function adapted from https://github.com/HackerPoet/FractalClock
        let colour_time = ms as f32 / 1000.0 + sec as f32 + min as f32 * 60.0 + hr as f32 * 3600.0;
        let r1 = (colour_time * 0.017).sin() * 0.5 + 0.5;
        let r2 = (colour_time * -0.011).sin() * 0.5 + 0.5;
        let r3 = (colour_time * 0.003).sin() * 0.5 + 0.5;

        let mut colours = Vec::with_capacity(self.fractal_depth as usize);
        for i in 1..=self.fractal_depth {
            let a = (self.fractal_depth - i) as f32 / self.fractal_depth as f32;
            let h = r2 + 0.5 * a;
            let s = 0.5 + 0.5 * r3 - 0.5 * (1.0 - a);
            let v = (0.3 * 0.5 * r1).max(0.5);
            if i == self.fractal_depth {
                let [r, g, b] = rgb_from_hsl((h, 1.0, 0.5));
                colours.push([r, g, b, 0.5]);
            } else {
                let [r, g, b] = rgb_from_hsl((h, s, v));
                colours.push([r, g, b, 1.0]);
            }
        }

        let colour_buffer = graph.add_intermediate_buffer(wgpu::BufferDescriptor {
            label: Some("FC Colour Buffer"),
            size: std::mem::size_of::<[f64; 4]>() as u64 * self.fractal_depth as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        graph
            .renderer()
            .write_buffer(colour_buffer, 0, bytemuck::cast_slice(&colours));

        graph.renderer().write_buffer(
            self.clock_buffer,
            0,
            bytemuck::cast_slice(&[
                ClockRay {
                    start: 0.0,
                    end: 0.175,
                    direction: hour_frac * std::f32::consts::TAU,
                    half_thickness: 0.02085,
                },
                ClockRay {
                    start: 0.0,
                    end: 0.4,
                    direction: minute_frac * std::f32::consts::TAU,
                    half_thickness: 0.0095,
                },
                ClockRay {
                    start: 0.0,
                    end: 0.4,
                    direction: second_frac * std::f32::consts::TAU,
                    half_thickness: 0.005,
                },
            ]),
        );

        let vertex_buffer = graph.add_intermediate_buffer(wgpu::BufferDescriptor {
            label: Some("FC Vertex Buffer"),
            size: vertex_count_for_depth(self.fractal_depth) * vertex_size() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        for i in 0..self.fractal_depth {
            let uniform_buffer = graph.add_intermediate_buffer(wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<[f32; 4]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            graph.renderer().write_buffer(
                uniform_buffer,
                0,
                bytemuck::cast_slice(&[
                    i as f32,
                    second_frac * std::f32::consts::TAU,
                    minute_frac * std::f32::consts::TAU,
                    hour_frac * std::f32::consts::TAU,
                ]),
            );
            graph
                .add_compute_node(self.vertices_compute_pipeline)
                .with_bind_group((vertex_buffer,))
                .with_bind_group((uniform_buffer,))
                .build(move |pass, [], ()| {
                    let workgroup_size = workgroup_size_for_depth((i + 1) as _) as u32;
                    let workgroup_y = workgroup_size / 256 + 1;
                    let workgroup_x = workgroup_size % 256 + 1;

                    pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
                })
        }

        let depth_texture = graph.add_intermediate_texture(wgpu::TextureDescriptor {
            label: Some("FC Depth Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 4,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let resolve_texture = graph.add_intermediate_texture(wgpu::TextureDescriptor {
            label: Some("FC MSAA Resolve Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 4,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        if self.fractal_depth > 1 {
            graph
                .add_render_node(self.fractal_render_pipeline)
                .with_buffer(vertex_buffer)
                .with_buffer(self.index_buffer)
                .with_bind_group((self.camera_buffer,))
                .with_bind_group((colour_buffer,))
                .build(
                    RenderPassTarget::new()
                        .with_msaa_color(
                            resolve_texture,
                            target,
                            wgpu::Color {
                                r: 0.01,
                                g: 0.01,
                                b: 0.01,
                                a: 1.0,
                            },
                        )
                        .with_depth(depth_texture, 1.0),
                    |pass, [vertex_buffer, index_buffer], ()| {
                        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(
                            6..(vertex_count_for_depth(self.fractal_depth) * 2 - 2) as _,
                            0,
                            0..1,
                        );
                    },
                );
        }

        graph
            .add_render_node(self.clock_render_pipeline)
            .with_buffer(self.clock_buffer)
            .with_bind_group((self.camera_buffer,))
            .build(
                RenderPassTarget::new()
                    .with_msaa_color(
                        resolve_texture,
                        target,
                        wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.01,
                            a: 1.0,
                        },
                    )
                    .with_depth(depth_texture, 1.0),
                |pass, [clock_buffer], ()| {
                    pass.set_vertex_buffer(0, clock_buffer.slice(..));
                    pass.draw(0..4, 0..75);
                },
            )
    }
}

fn vertex_size() -> usize {
    std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<f32>() * 2
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ClockRay {
    start: f32,
    end: f32,
    direction: f32,
    half_thickness: f32,
}

pub fn workgroup_size_for_depth(depth: u32) -> u64 {
    3_u64.pow(depth - 1)
}

pub fn vertex_count_for_depth(depth: u32) -> u64 {
    3_u64.pow(depth + 1) / 2 // OEIS A003462
}

/// All ranges in 0-1, rgb is linear.
pub fn rgb_from_hsv((h, s, v): (f32, f32, f32)) -> [f32; 3] {
    let h = (h.fract() + 1.0).fract(); // wrap
    let s = s.clamp(0.0, 1.0);

    let f = h * 6.0 - (h * 6.0).floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    match (h * 6.0).floor() as i32 % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        5 => [v, p, q],
        _ => unreachable!(),
    }
}

pub fn rgb_from_hsl((h, s, l): (f32, f32, f32)) -> [f32; 3] {
    let a = s * l.min(1.0 - l);
    let f = |n: f32| {
        let k = (n + h / (1.0 / 12.0)) % 12.0;
        l - a * (k - 3.0).min(9.0 - k).min(1.0).max(-1.0)
    };
    [f(0.0), f(8.0), f(4.0)]
}

#[cfg(not(unix))]
fn local_timezone() -> UtcOffset {
    UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC)
}

#[cfg(unix)]
fn local_timezone() -> UtcOffset {
    // Workaround for `time` not providing local time on unix systems.
    let mut time = 0;
    let tm = unsafe {
        // SAFETY: We know our app isn't multithreaded, so we can just use these functions.
        libc::time(&mut time);
        *libc::localtime(&time)
    };
    UtcOffset::from_whole_seconds(tm.tm_gmtoff as i32).unwrap_or(UtcOffset::UTC)
}
