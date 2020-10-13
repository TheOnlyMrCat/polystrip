//! The core rendering context structures

use std::convert::TryFrom;

use crate::data::Color;
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
	pub(crate) device: wgpu::Device, //TODO: Encapsulation?
	pub(crate) queue: wgpu::Queue,
	sc_desc: wgpu::SwapChainDescriptor,
	swap_chain: wgpu::SwapChain,
	render_pipeline: wgpu::RenderPipeline,

	pub(crate) texture_bind_group_layout: wgpu::BindGroupLayout,
	texture_bind_group: wgpu::BindGroup,
	pub(crate) textures: Vec<wgpu::Texture>,
	pub(crate) texture_views: Vec<wgpu::TextureView>,
	sampler: wgpu::Sampler,

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
				features: wgpu::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
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

		let texture_bind_group_layout = device.create_bind_group_layout(
			&wgpu::BindGroupLayoutDescriptor {
				entries: &[
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStage::FRAGMENT,
						ty: wgpu::BindingType::Sampler {
							comparison: false,
						},
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStage::FRAGMENT,
						ty: wgpu::BindingType::SampledTexture {
							multisampled: false,
							dimension: wgpu::TextureViewDimension::D2,
							component_type: wgpu::TextureComponentType::Uint,
						},
						count: Some(std::num::NonZeroU32::new(8).unwrap()), //* Make sure this matches the array size in shader.frag
					},
				],
				label: Some("polystrip_texture_bind_group_layout"),
			}
		);

		let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: None,
			address_mode_u: wgpu::AddressMode::Repeat,
			address_mode_v: wgpu::AddressMode::Repeat,
			address_mode_w: wgpu::AddressMode::Repeat,
			mag_filter: wgpu::FilterMode::Nearest,
			min_filter: wgpu::FilterMode::Nearest,
			mipmap_filter: wgpu::FilterMode::Nearest,
			..Default::default()
		});

		let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("polystrip_texture_bind_group"),
			layout: &texture_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Sampler(&sampler),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureViewArray(&[]),
				}
			]
		});
		
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

		let vs_module = device.create_shader_module(wgpu::include_spirv!("spirv/shader.vert.spv"));
		let fs_module = device.create_shader_module(wgpu::include_spirv!("spirv/shader.frag.spv"));

		let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("polystrip_render_pipeline_layout"),
			bind_group_layouts: &[&texture_bind_group_layout],
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
			texture_bind_group_layout, sampler, texture_bind_group,
			textures: Vec::new(), texture_views: Vec::new(),
			vertex_buffer, index_buffer,
		}
	}
	
	/// Renders a frame. 
	pub fn render_frame(&mut self, frame: Frame) {
		let swap_chain_frame: wgpu::SwapChainTexture = self.swap_chain.get_current_frame().expect("Couldn't get the next frame").output.into();
		let mut encoder: wgpu::CommandEncoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("polystrip_render_encoder"),
		}).into();

		//TODO: Merge these two iterations into one which produces two vectors?
		let vertex_data = frame.shapes.iter().enumerate().flat_map(|(i, shape)| {
			let depth = i as f32;
			match shape {
				Shape::Colored { vertices, .. } => vertices.iter().map(move |&v| Vertex::from_color(v, depth)).collect::<Vec<_>>(),
				Shape::Textured { vertices, texture_index, .. } => vertices.iter().map(move |&v| Vertex::from_texture(v, *texture_index, depth)).collect::<Vec<_>>(),
			}
		}).collect::<Vec<_>>();
		self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertex_data));

		let mut index_offset: u16 = 0;
		let mut index_data = frame.shapes.iter().flat_map(|shape| {
			match shape {
				Shape::Colored { indices, vertices, .. } => {
					let indices = indices.iter()
						.flatten()
						.map(|&index| index + index_offset)
						.collect::<Vec<_>>();
					index_offset += u16::try_from(vertices.len()).unwrap();
					indices
				},
				Shape::Textured { indices, vertices, .. } => {
					let indices = indices.iter()
						.flatten()
						.map(|&index| index + index_offset)
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
		self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&index_data));

		let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			color_attachments: &[
				wgpu::RenderPassColorAttachmentDescriptor {
					attachment: &swap_chain_frame.view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: match frame.clear_color {
							Some(c) => wgpu::LoadOp::Clear(wgpu::Color {
								//TODO: Convert srgb properly
								r: f64::from(c.r) / 255.0,
								g: f64::from(c.g) / 255.0,
								b: f64::from(c.b) / 255.0,
								a: 1.0,
							}),
							None => wgpu::LoadOp::Load,
						},
						store: true,
					}
				}
			],
			depth_stencil_attachment: None,
		});

		render_pass.set_pipeline(&self.render_pipeline);

		render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
		render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
		render_pass.set_index_buffer(self.index_buffer.slice(..));
		render_pass.draw_indexed(0..u32::try_from(index_len).unwrap(), 0, 0..1);

		std::mem::drop(render_pass);

		self.queue.submit(std::iter::once(encoder.finish()));
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

	pub(crate) fn recreate_bind_group(&mut self) {
		self.texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("polystrip_texture_bind_group"),
			layout: &self.texture_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::Sampler(&self.sampler),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureViewArray(&self.texture_views),
				}
			]
		});
	}
}

/// The data necessary for a frame to be rendered. Stores [`Shape`](../vertex/enum.Shape.html)s and gets passed to
/// [`Renderer`](struct.Renderer.html) to be rendered.
pub struct Frame {
	shapes: Vec<Shape>,
	clear_color: Option<Color>,
}

impl Frame {
	/// Creates a new frame with no shapes and no clear colour.
	pub fn new() -> Frame {
		Frame {
			shapes: Vec::new(),
			clear_color: None,
		}
	}

	/// Queues up the passed [`Shape`](../vertex/enum.Shape.html) for rendering. Shapes are rendered in the order they are
	/// queued in.
	pub fn push_shape(&mut self, shape: Shape) {
		self.shapes.push(shape);
	}

	/// Sets the clear color of the frame. The frame is cleared before any shapes are drawn.
	/// 
	/// Any shapes drawn before calling this method will still be drawn.
	pub fn set_clear(&mut self, color: Color) {
		self.clear_color = Some(color);
	}

	/// Resets the clear color of the frame, meaning the previous frame is underlayed beneath this one. This is the default.
	pub fn reset_clear(&mut self) {
		self.clear_color = None;
	}

	/// Reserve space for at least `additional` more shapes to be drawn. This increases the capacity of the internal `Vec`
	/// used to store shapes before rendering. See [`Vec`'s documentation](https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.reserve)
	/// for more details.
	pub fn reserve(&mut self, additional: usize) {
		self.shapes.reserve(additional);
	}
}