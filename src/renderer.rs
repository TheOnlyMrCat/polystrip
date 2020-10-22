//! The core rendering context structures

use std::convert::TryFrom;

use crate::data::{GpuPos, Color};
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

	width: u32,
	height: u32,
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
					color_blend: wgpu::BlendDescriptor {
						src_factor: wgpu::BlendFactor::SrcAlpha,
						dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
						operation: wgpu::BlendOperation::Add,
					},
					alpha_blend: wgpu::BlendDescriptor {
						src_factor: wgpu::BlendFactor::One,
						dst_factor: wgpu::BlendFactor::Zero,
						operation: wgpu::BlendOperation::Add,
					},
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
					color_blend: wgpu::BlendDescriptor {
						src_factor: wgpu::BlendFactor::SrcAlpha,
						dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
						operation: wgpu::BlendOperation::Add,
					},
					alpha_blend: wgpu::BlendDescriptor {
						src_factor: wgpu::BlendFactor::One,
						dst_factor: wgpu::BlendFactor::Zero,
						operation: wgpu::BlendOperation::Add,
					},
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

			width: size.0, height: size.1,
		}
	}

	/// Returns the next `Frame`, which can be drawn to and will present on drop. This `Renderer` is borrowed mutably while the
	/// frame is alive. Any operations on this renderer must be done through the `Frame`, which implements `Dever<Target = Renderer>`.
	pub fn get_next_frame(&mut self) -> Frame<'_> {
		Frame {
			swap_chain_frame: self.swap_chain.get_current_frame().unwrap(),
			renderer: self,
		}
	}
	
	/// Resizes the internal swapchain
	/// 
	/// For correctness, call this method in your window's event loop whenever the window gets resized
	/// 
	/// # Arguments
	/// * `size`: The size of the window in pixels, in the order (width, height). For window implementations which
	///           differentiate between physical and logical size, this refers to the logical size
	pub fn resize(&mut self, size: (u32, u32)) {
		self.sc_desc.width = size.0;
		self.sc_desc.height = size.1;
		self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
	}

	/// Converts pixel coordinates to Gpu coordinates
	pub fn pixel(&self, x: i32, y: i32) -> GpuPos {
		GpuPos {
			x: (x * 2) as f32 / self.width as f32 - 1.0,
			y: -((y * 2) as f32 / self.height as f32 - 1.0),
		}
	}
}

/// A frame to be drawn to. The frame gets presented on drop.
/// 
/// Since a `Frame` borrows the [`Renderer`](struct.Renderer.html) it was created for, any functions which would normally
/// be called on a `&Renderer` must be called on the `Frame`, which implements `Deref<Target = Renderer>`.
/// 
/// More methods are implemented in the [`FrameGeometryExt`](../geometry/trait.FrameGeometryExt.html) trait.
pub struct Frame<'a> {
	renderer: &'a mut Renderer,
	swap_chain_frame: wgpu::SwapChainFrame,
}

//MARK: Frame API
impl<'a> Frame<'a> {
	/// Draws a [`ColoredShape`](../vertex/struct.ColoredShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	pub fn draw_colored(&mut self, shape: ColoredShape) {
		let mut encoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("polystrip_render_encoder"),
		});

		let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			color_attachments: &[
				wgpu::RenderPassColorAttachmentDescriptor {
					attachment: &self.swap_chain_frame.output.view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Load,
						store: true,
					}
				}
			],
			depth_stencil_attachment: None,
		});

		self.renderer.queue.write_buffer(&self.renderer.vertex_buffer, 0, bytemuck::cast_slice(&shape.vertices));
		let mut index_data = shape.indices.iter().flatten().copied().collect::<Vec<_>>();
		let index_count = index_data.len();
		if index_count % 2 == 1 {
			index_data.push(0); // Align the data to u32 for the upcoming buffer write
		}
		self.renderer.queue.write_buffer(&self.renderer.index_buffer, 0, bytemuck::cast_slice(&index_data));

		render_pass.set_pipeline(&self.renderer.colour_render_pipeline);
		render_pass.set_vertex_buffer(0, self.renderer.vertex_buffer.slice(..));
		render_pass.set_index_buffer(self.renderer.index_buffer.slice(..));
		render_pass.draw_indexed(0..index_count as u32, 0, 0..1);

		std::mem::drop(render_pass);

		self.renderer.queue.submit(std::iter::once(encoder.finish()));
	}

	/// Draws a [`TexturedShape`](../vertex/struct.TexturedShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	/// 
	/// # Arguments
	/// * `shape`: The `TexturedShape` to be rendered. 
	/// * `texture`: The `Texture` to be drawn to the geometry of the shape.
	pub fn draw_textured(&mut self, shape: TexturedShape, texture: &'a crate::texture::Texture) {
		let mut encoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("polystrip_render_encoder"),
		});

		let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			color_attachments: &[
				wgpu::RenderPassColorAttachmentDescriptor {
					attachment: &self.swap_chain_frame.output.view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Load,
						store: true,
					}
				}
			],
			depth_stencil_attachment: None,
		});

		self.renderer.queue.write_buffer(&self.renderer.vertex_buffer, 0, bytemuck::cast_slice(&shape.vertices));
		let mut index_data = shape.indices.iter().flatten().copied().collect::<Vec<_>>();
		let index_count = index_data.len();
		if index_count % 2 == 1 {
			index_data.push(0); // Align the data to u32 for the upcoming buffer write
		}
		self.renderer.queue.write_buffer(&self.renderer.index_buffer, 0, bytemuck::cast_slice(&index_data));

		render_pass.set_pipeline(&self.renderer.texture_render_pipeline);
		render_pass.set_bind_group(0, &texture.bind_group, &[]);
		render_pass.set_vertex_buffer(0, self.renderer.vertex_buffer.slice(..));
		render_pass.set_index_buffer(self.renderer.index_buffer.slice(..));
		render_pass.draw_indexed(0..index_count as u32, 0, 0..1);

		std::mem::drop(render_pass);

		self.renderer.queue.submit(std::iter::once(encoder.finish()));
	}

	/// Draws a [`ShapeSet`](../vertex/enum.ShapeSet.html). All shapes in the set will be drawn in front of shapes drawn before
	/// the set. The render order of shapes in the set is unspecified.
	pub fn draw_shape_set(&mut self, set: ShapeSet<'a>) {
		let mut encoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("polystrip_render_encoder"),
		});

		let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			color_attachments: &[
				wgpu::RenderPassColorAttachmentDescriptor {
					attachment: &self.swap_chain_frame.output.view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Load,
						store: true,
					}
				}
			],
			depth_stencil_attachment: None,
		});

		let index_count;
		match set {
			ShapeSet::Colored(shapes) => {
				//TODO: Merge these two iterations into one which produces two vectors?
				let vertex_data = shapes.iter().flat_map(|shape| shape.vertices.iter()).copied().collect::<Vec<_>>();
				self.renderer.queue.write_buffer(&self.renderer.vertex_buffer, 0, bytemuck::cast_slice(&vertex_data));

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
				self.renderer.queue.write_buffer(&self.renderer.index_buffer, 0, bytemuck::cast_slice(&index_data));

				render_pass.set_pipeline(&self.renderer.colour_render_pipeline);
			},
			ShapeSet::Textured(shapes, texture) => {
				// ! Duplicated code from above branch
				let vertex_data = shapes.iter().flat_map(|shape| shape.vertices.iter()).copied().collect::<Vec<_>>();
				self.renderer.queue.write_buffer(&self.renderer.vertex_buffer, 0, bytemuck::cast_slice(&vertex_data));

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
				self.renderer.queue.write_buffer(&self.renderer.index_buffer, 0, bytemuck::cast_slice(&index_data));
				// ! End of duplicated code

				render_pass.set_pipeline(&self.renderer.texture_render_pipeline);
				render_pass.set_bind_group(0, &texture.bind_group, &[]);
			}
		}

		render_pass.set_vertex_buffer(0, self.renderer.vertex_buffer.slice(..));
		render_pass.set_index_buffer(self.renderer.index_buffer.slice(..));
		render_pass.draw_indexed(0..index_count as u32, 0, 0..1);

		std::mem::drop(render_pass);

		self.renderer.queue.submit(std::iter::once(encoder.finish()));
	}

	/// Clears the entire frame with the specified color, setting every pixel to its value.
	pub fn clear(&mut self, color: Color) {
		let mut encoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("polystrip_render_encoder"),
		});

		let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			color_attachments: &[
				wgpu::RenderPassColorAttachmentDescriptor {
					attachment: &self.swap_chain_frame.output.view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color {
							//TODO: Convert sRGB properly
							r: color.r as f64 / 255.0,
							g: color.g as f64 / 255.0,
							b: color.b as f64 / 255.0,
							a: color.a as f64 / 255.0,
						}),
						store: true,
					}
				}
			],
			depth_stencil_attachment: None,
		});

		std::mem::drop(render_pass);

		self.renderer.queue.submit(std::iter::once(encoder.finish()));
	}
}

impl<'a> std::ops::Deref for Frame<'a> {
	type Target = Renderer;

	fn deref(&self) -> &Renderer {
		&self.renderer
	}
}