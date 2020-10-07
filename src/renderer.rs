use std::convert::TryFrom;

use crate::vertex::{Vertex, Shape};

use raw_window_handle::HasRawWindowHandle;

/// An accelerated 2D renderer.
/// 
/// A renderer can be created for any window compatible with `raw_window_handle`. The size of this window must be updated
/// in the event loop, and specified on creation. For example, using `winit`:
/// ```no_run
/// # use winit::event::{Event, WindowEvent};
/// # use polystrip::prelude::Renderer;
/// # let event_loop = winit::event_loop::EventLoop::new();
/// # let window = winit::window::Window::new(&event_loop).unwrap();
/// let window_size = window.inner_size().to_logical(window.scale_factor());
/// let mut renderer = Renderer::new(&window, (window_size.width, window_size.height));
/// 
/// event_loop.run(move |event, _, control_flow| {
///     match event {
///         Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
///             let window_size = new_size.to_logical(window.scale_factor());
///             renderer.resize((window_size.width, window_size.height));
///         },
///         // --snip--
/// #       _ => {}
///     }
/// });
/// ```
pub struct Renderer {
	surface: wgpu::Surface,
	device: wgpu::Device,
	queue: wgpu::Queue,
	sc_desc: wgpu::SwapChainDescriptor,
	swap_chain: wgpu::SwapChain,
	render_pipeline: wgpu::RenderPipeline,

	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
}

//TODO: Builder pattern, to allow for more configuration?
impl Renderer {
	/// Creates a new renderer, initialising the wgpu backend. This method just calls and blocks on [`new_async`](#method.new_async)
	/// 
	/// # Arguments
	/// * `window`: A valid window compatible with `raw_window_handle`.
	/// * `size`: The size of the window in pixels, in the order (width, height). For window implementations which
	///           differentiate between physical and logical size, this refers to the logical size
	pub fn new(window: &impl HasRawWindowHandle, size: (u32, u32)) -> Renderer {
		futures::executor::block_on(Renderer::new_async(window, size))
	}

	/// Creates a new renderer asynchronously, initialising the wgpu backend. This method assumes the raw window handle
	/// was created legitimately. Technically, that's my problem, but if you're not making your window properly, I'm not
	/// going to take responsibility for the resulting crash.
	///
	/// # Arguments
	/// * `window`: A valid window compatible with `raw_window_handle`.
	/// * `size`: The size of the window in pixels, in the order (width, height). For window implementations which
	///           differentiate between physical and logical size, this refers to the logical size
	pub async fn new_async(window: &impl HasRawWindowHandle, size: (u32, u32)) -> Renderer {
		let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
		let surface = unsafe { instance.create_surface(window) };

		let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::Default,
			compatible_surface: Some(&surface),
		}).await.unwrap();

		let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(), //TODO
                shader_validation: true,
            },
            None, // Trace path
		).await.unwrap();
		
		let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::Fifo,
        };
		let swap_chain = device.create_swap_chain(&surface, &sc_desc);

		let vs_module = device.create_shader_module(wgpu::include_spirv!("spirv/shader.vert.spv"));
		let fs_module = device.create_shader_module(wgpu::include_spirv!("spirv/shader.frag.spv"));
		
		let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("polystrip_vertex_buffer"),
			size: (1024 * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress, //TODO: Figure out how big this should be
			usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
			mapped_at_creation: false,
		});

		let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("polystrip_index_buffer"),
			size: (1024 * std::mem::size_of::<u16>()) as wgpu::BufferAddress,
			usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::INDEX,
			mapped_at_creation: false,
		});

		let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("polystrip_render_pipeline_layout"),
			bind_group_layouts: &[],
			push_constant_ranges: &[],
		});

		let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("polystrip_render_pipeline"),
			layout: Some(&render_pipeline_layout),
			vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
				wgpu::ColorStateDescriptor {
					format: sc_desc.format,
					color_blend: wgpu::BlendDescriptor::REPLACE,
					alpha_blend: wgpu::BlendDescriptor::REPLACE,
					write_mask: wgpu::ColorWrite::ALL,
				}
			],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[Vertex::desc()],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
		});

		Renderer {
			surface, device, queue, sc_desc, swap_chain, render_pipeline,
			vertex_buffer, index_buffer,
		}
	}

	/// Returns a frame to draw on. The returned [`Frame`](struct.Frame.html) will draw its contents to this
	/// renderer when it is dropped. See its docs for information on how to draw.
	/// 
	/// The renderer cannot be modified or create another frame while there is already a frame alive, due to the
	/// lifetime bounds on this method.
	pub fn begin_frame<'a>(&'a mut self) -> Frame<'a> {
		Frame {
			renderer: self,
			shapes: vec![],
		}
	}

	/// This function should be called in your event loop whenever the window gets resized.
	/// 
	/// # Arguments
	/// * `size`: The size of the window in pixels, in the order (width, height). For window implementations which
	///           differentiate between physical and logical size, this refers to the logical size
	pub fn resize(&mut self, size: (u32, u32)) {
		self.sc_desc.width = size.0;
		self.sc_desc.height = size.1;
		self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
	}	
}

/// The data necessary for a single frame to be rendered. Stores [`Shape`](../vertex/enum.Shape.html)s and renders them
/// when this struct is dropped.
pub struct Frame<'a> {
	renderer: &'a mut Renderer,
	shapes: Vec<Shape>,
}

impl Frame<'_> {
	/// Queues up the passed [`Shape`](../vertex/enum.Shape.html) for rendering. Shapes are rendered in the order they are
	/// queued in.
	pub fn draw_shape(&mut self, shape: Shape) {
		self.shapes.push(shape);
	}

	/// Reserve space for at least `additional` more shapes to be drawn. This increases the capacity of the internal `Vec`
	/// used to store shapes before rendering. See [`Vec`'s documentation](https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.reserve)
	/// for more details.
	pub fn reserve(&mut self, additional: usize) {
		self.shapes.reserve(additional);
	}
}

impl Drop for Frame<'_> {
	fn drop(&mut self) {
		let frame: wgpu::SwapChainTexture = self.renderer.swap_chain.get_current_frame().expect("Couldn't get the next frame").output.into();
		let mut encoder: wgpu::CommandEncoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("polystrip_render_encoder"),
		}).into();

		//TODO: Merge these two iterations into one which produces two vectors?
		let vertex_data = self.shapes.iter().enumerate().flat_map(|(i, shape)| {
			let depth = i as f32;
			match shape {
				Shape::Colored { vertices, .. } => vertices.iter().map(move |v| Vertex::from_color(*v, depth)),
				Shape::Textured { vertices, texture_index, .. } => todo!(),
			}
		}).collect::<Vec<_>>();
		self.renderer.queue.write_buffer(&self.renderer.vertex_buffer, 0, bytemuck::cast_slice(&vertex_data));

		let mut index_offset: u16 = 0;
		let mut index_data = self.shapes.iter().flat_map(|shape| {
			match shape {
				Shape::Colored { indices, vertices, .. } => {
					let indices = indices.iter()
						.flat_map(|&indices| vec![indices.0 + index_offset, indices.1 + index_offset, indices.2 + index_offset].into_iter())
						.collect::<Vec<_>>();
					index_offset += u16::try_from(vertices.len()).unwrap();
					indices
				},
				Shape::Textured { indices, vertices, .. } => {
					let indices = indices.iter()
						.flat_map(|&indices| vec![indices.0 + index_offset, indices.1 + index_offset, indices.2 + index_offset].into_iter())
						.collect::<Vec<_>>();
					index_offset += u16::try_from(vertices.len()).unwrap();
					indices
				},
			}
		}).collect::<Vec<u16>>();
		let index_len = index_data.len();
		if index_len % 2 == 1 {
			index_data.push(0);
		}
		self.renderer.queue.write_buffer(&self.renderer.index_buffer, 0, bytemuck::cast_slice(&index_data));

		let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			color_attachments: &[
				wgpu::RenderPassColorAttachmentDescriptor {
					attachment: &frame.view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color {
							r: 0.0,
							g: 0.0,
							b: 0.0,
							a: 1.0,
						}),
						store: true,
					}
				}
			],
			depth_stencil_attachment: None,
		});

		render_pass.set_pipeline(&self.renderer.render_pipeline);

		render_pass.set_vertex_buffer(0, self.renderer.vertex_buffer.slice(..));
		render_pass.set_index_buffer(self.renderer.index_buffer.slice(..));
		render_pass.draw_indexed(0..u32::try_from(index_len).unwrap(), 0, 0..1);

		// Vertex and uniform buffers and rendering

		std::mem::drop(render_pass);

		self.renderer.queue.submit(std::iter::once(encoder.finish()));
	}
}