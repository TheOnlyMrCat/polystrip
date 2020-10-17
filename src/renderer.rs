//! The core rendering context structures

use std::convert::TryFrom;

use crate::data::Color;
use crate::vertex::*;

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
	colour_render_pipeline: wgpu::RenderPipeline,
	texture_render_pipeline: wgpu::RenderPipeline,

	pub(crate) texture_bind_group_layout: wgpu::BindGroupLayout,

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
		//MARK: New Renderer
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

		let texture_bind_group_layout = device.create_bind_group_layout(
			&wgpu::BindGroupLayoutDescriptor {
				entries: &[
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStage::FRAGMENT,
						ty: wgpu::BindingType::SampledTexture {
							multisampled: false,
							dimension: wgpu::TextureViewDimension::D2,
							component_type: wgpu::TextureComponentType::Uint,
						},
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStage::FRAGMENT,
						ty: wgpu::BindingType::Sampler {
							comparison: false,
						},
						count: None,
					},
				],
				label: Some("polystrip_texture_bind_group_layout"),
			}
		);
		
		let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("polystrip_vertex_buffer"),
			size: (1024 * std::mem::size_of::<TextureVertex>()) as wgpu::BufferAddress, //TODO: Figure out how big this should be
			usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
			mapped_at_creation: false,
		});

		let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("polystrip_index_buffer"),
			size: (1024 * std::mem::size_of::<u16>()) as wgpu::BufferAddress,
			usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::INDEX,
			mapped_at_creation: false,
		});

		let colour_vs_module = device.create_shader_module(wgpu::include_spirv!("spirv/coloured.vert.spv"));
		let colour_fs_module = device.create_shader_module(wgpu::include_spirv!("spirv/coloured.frag.spv"));

		let colour_render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("polystrip_render_pipeline_layout_color"),
			bind_group_layouts: &[],
			push_constant_ranges: &[],
		});

		let colour_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("polystrip_render_pipeline_color"),
			layout: Some(&colour_render_pipeline_layout),
			vertex_stage: wgpu::ProgrammableStageDescriptor {
				module: &colour_vs_module,
				entry_point: "main",
			},
			fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
				module: &colour_fs_module,
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
					color_blend: wgpu::BlendDescriptor::REPLACE, //TODO: Choose suitable descriptors
					alpha_blend: wgpu::BlendDescriptor::REPLACE,
					write_mask: wgpu::ColorWrite::ALL,
				}
			],
			depth_stencil_state: None,
			vertex_state: wgpu::VertexStateDescriptor {
				index_format: wgpu::IndexFormat::Uint16,
				vertex_buffers: &[ColorVertex::desc()],
			},
			sample_count: 1,
			sample_mask: !0,
			alpha_to_coverage_enabled: false,
		});

		let texture_vs_module = device.create_shader_module(wgpu::include_spirv!("spirv/textured.vert.spv"));
		let texture_fs_module = device.create_shader_module(wgpu::include_spirv!("spirv/textured.frag.spv"));

		let texture_render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("polystrip_render_pipeline_layout_texture"),
			bind_group_layouts: &[&texture_bind_group_layout],
			push_constant_ranges: &[],
		});

		let texture_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("polystrip_render_pipeline_texture"),
			layout: Some(&texture_render_pipeline_layout),
			vertex_stage: wgpu::ProgrammableStageDescriptor {
				module: &texture_vs_module,
				entry_point: "main",
			},
			fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
				module: &texture_fs_module,
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
					color_blend: wgpu::BlendDescriptor::REPLACE, //TODO: As above
					alpha_blend: wgpu::BlendDescriptor::REPLACE,
					write_mask: wgpu::ColorWrite::ALL,
				}
			],
			depth_stencil_state: None,
			vertex_state: wgpu::VertexStateDescriptor {
				index_format: wgpu::IndexFormat::Uint16,
				vertex_buffers: &[TextureVertex::desc()],
			},
			sample_count: 1,
			sample_mask: !0,
			alpha_to_coverage_enabled: false,
		});

		Renderer {
			surface, device, queue, sc_desc, swap_chain, colour_render_pipeline, texture_render_pipeline,
			texture_bind_group_layout,
			vertex_buffer, index_buffer,
		}
	}
	
	/// Renders a frame. 
	pub fn render_frame(&mut self, frame: Frame) {
		//MARK: Render frame
		let swap_chain_frame: wgpu::SwapChainTexture = self.swap_chain.get_current_frame().expect("Couldn't get the next frame").output.into();
		
		let mut is_first_set = true;
		for set in &frame.shape_sets {
			let mut encoder: wgpu::CommandEncoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("polystrip_render_encoder"),
			}).into();
	
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments: &[
					wgpu::RenderPassColorAttachmentDescriptor {
						attachment: &swap_chain_frame.view,
						resolve_target: None,
						ops: wgpu::Operations {
							load: 
								if is_first_set {
									match frame.clear_color {
										Some(c) => wgpu::LoadOp::Clear(wgpu::Color {
											//TODO: Convert srgb properly
											r: f64::from(c.r) / 255.0,
											g: f64::from(c.g) / 255.0,
											b: f64::from(c.b) / 255.0,
											a: 1.0,
										}),
										None => wgpu::LoadOp::Load,
									}
								} else {
									wgpu::LoadOp::Load
								},
							store: true,
						}
					}
				],
				depth_stencil_attachment: None,
			});
			is_first_set = false;

			// * This match block is expected to set the render pipeline and write the vertex and index buffers
			let index_count;
			match &set {
				ShapeSet::SingleColored(shape) => {
					self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&shape.vertices));
					let mut index_data = shape.indices.iter().flatten().copied().collect::<Vec<_>>();
					index_count = index_data.len();
					if index_count % 2 == 1 {
						index_data.push(0); // Align the data to u32 for the upcoming buffer write
					}
					self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&index_data));

					render_pass.set_pipeline(&self.colour_render_pipeline);
				},
				ShapeSet::SingleTextured(shape, texture) => {
					// ! Duplicated code from above branch
					self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&shape.vertices));
					let mut index_data = shape.indices.iter().flatten().copied().collect::<Vec<_>>();
					index_count = index_data.len();
					if index_count % 2 == 1 {
						index_data.push(0); // Align the data to u32 for the upcoming buffer write
					}
					self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&index_data));
					// ! End of duplicated code

					render_pass.set_pipeline(&self.texture_render_pipeline);
					render_pass.set_bind_group(0, &texture.bind_group, &[]);
				},
				ShapeSet::MultiColored(shapes) => {
					//TODO: Merge these two iterations into one which produces two vectors?
					let vertex_data = shapes.iter().flat_map(|shape| shape.vertices.iter()).copied().collect::<Vec<_>>();
					self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertex_data));

					let mut index_offset: u16 = 0;
					let mut index_data = shapes.iter().flat_map(|shape| {
						let indices = shape.indices.iter()
							.flatten()
							.map(|&index| index + index_offset)
							.collect::<Vec<_>>();
						index_offset += u16::try_from(shape.vertices.len()).unwrap();
						indices
					}).collect::<Vec<u16>>();
					index_count = index_data.len();
					if index_count % 2 == 1 {
						index_data.push(0); // Align the data to u32 for the upcoming buffer write
					}
					self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&index_data));

					render_pass.set_pipeline(&self.colour_render_pipeline);
				},
				ShapeSet::MultiTextured(shapes, texture) => {
					// ! Duplicated code from above branch
					let vertex_data = shapes.iter().flat_map(|shape| shape.vertices.iter()).copied().collect::<Vec<_>>();
					self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertex_data));

					let mut index_offset: u16 = 0;
					let mut index_data = shapes.iter().flat_map(|shape| {
						let indices = shape.indices.iter()
							.flatten()
							.map(|&index| index + index_offset)
							.collect::<Vec<_>>();
						index_offset += u16::try_from(shape.vertices.len()).unwrap();
						indices
					}).collect::<Vec<u16>>();
					index_count = index_data.len();
					if index_count % 2 == 1 {
						index_data.push(0); // Align the data to u32 for the upcoming buffer write
					}
					self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&index_data));
					// ! End of duplicated code

					render_pass.set_pipeline(&self.texture_render_pipeline);
					render_pass.set_bind_group(0, &texture.bind_group, &[]);
				}
			};
	
			render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
			render_pass.set_index_buffer(self.index_buffer.slice(..));
			render_pass.draw_indexed(0..index_count as u32, 0, 0..1);

			std::mem::drop(render_pass);
	
			self.queue.submit(std::iter::once(encoder.finish()));
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

/// The data necessary for a frame to be rendered. Stores [`ShapeSet`](../vertex/enum.ShapeSet.html)s and gets passed to
/// [`Renderer`](struct.Renderer.html) to be rendered.
pub struct Frame<'a> {
	shape_sets: Vec<ShapeSet<'a>>,
	clear_color: Option<Color>,
}

//MARK: Frame API
impl<'a> Frame<'a> {
	/// Creates a new frame with no shape sets and no clear colour.
	pub fn new() -> Frame<'a> {
		Frame {
			shape_sets: Vec::new(),
			clear_color: None,
		}
	}

	/// Queues the passed [`ColoredShape`](../vertex/struct.ColoredShape.html) for rendering. Shapes are rendered in the order
	/// they are queued in.
	pub fn push_colored(&mut self, shape: ColoredShape) {
		self.shape_sets.push(ShapeSet::SingleColored(shape));
	}

	/// Queues the passed [`TexturedShape`](../vertex/struct.TexturedShape.html) for rendering. The shape will be rendered with
	/// the passed texture. Shapes are rendered in the order they are queued in.
	pub fn push_textured(&mut self, shape: TexturedShape, texture: &'a crate::texture::Texture) {
		self.shape_sets.push(ShapeSet::SingleTextured(shape, texture));
	}

	/// Queues the passed [`ShapeSet`](../vertex/enum.ShapeSet.html) for rendering. Shapes and shape sets are rendered in the
	/// order they are queued in.
	pub fn push_shape_set(&mut self, set: ShapeSet<'a>) {
		self.shape_sets.push(set);
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

	/// Reserve space for at least `additional` more shape sets to be drawn. This increases the capacity of the internal `Vec`
	/// used to store shapes before rendering. See [`Vec`'s documentation](https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.reserve)
	/// for more details.
	pub fn reserve(&mut self, additional: usize) {
		self.shape_sets.reserve(additional);
	}
}