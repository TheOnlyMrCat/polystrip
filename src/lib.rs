//! Polystrip is an accelerated 2D graphics library with similar capabilities to SDL's graphics system, which
//! it intends to be a replacement for.
//! 
//! # The `Renderer`
//! The [`Renderer`](renderer/struct.Renderer.html) is the core of `polystrip`. It can be built on top of any window compatible with `raw_window_handle`,
//! but must have its size managed manually. A `Renderer` allows one to create [`Frame`](renderer/struct.Frame.html)s,
//! which can be drawn onto and will present themselves when they are dropped.
//! 
//! # Example with `winit`
//! ```no_run
//! # use winit::event::{Event, WindowEvent};
//! # use polystrip::prelude::Renderer;
//! let event_loop = winit::event_loop::EventLoop::new();
//! let window = winit::window::Window::new(&event_loop).unwrap();
//! 
//! let window_size = window.inner_size().to_logical(window.scale_factor());
//! let mut renderer = Renderer::new(&window, (window_size.width, window_size.height));
//! 
//! event_loop.run(move |event, _, control_flow| {
//!     match event {
//!         Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
//!             let window_size = new_size.to_logical(window.scale_factor());
//!             renderer.resize((window_size.width, window_size.height));
//!         },
//!         Event::MainEventsCleared => {
//!             let mut frame = renderer.get_next_frame();
//!             // Render in here
//!         },
//!         _ => {}
//!     }
//! });
//! ```

pub(crate) mod backend;
pub mod data;
pub mod vertex;
pub mod texture;
pub use texture::Texture;

use std::cell::Cell;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use crate::data::{GpuPos, Color};
use crate::vertex::*;

use raw_window_handle::HasRawWindowHandle;

use gfx_hal::prelude::*;

use align_data::{include_aligned, Align32};
static COLOURED_VERT_SPV: &[u8] = include_aligned!(Align32, "../spirv/coloured.vert.spv");
static COLOURED_FRAG_SPV: &[u8] = include_aligned!(Align32, "../spirv/coloured.frag.spv");
static TEXTURED_VERT_SPV: &[u8] = include_aligned!(Align32, "../spirv/textured.vert.spv");
static TEXTURED_FRAG_SPV: &[u8] = include_aligned!(Align32, "../spirv/textured.frag.spv");

pub struct RendererContext {
	pub instance: backend::Instance,
	pub gpu: gfx_hal::adapter::Gpu<backend::Backend>,
	pub adapter: gfx_hal::adapter::Adapter<backend::Backend>,

	pub colour_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	pub texture_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	pub texture_graphics_pipeline_layout: ManuallyDrop<backend::PipelineLayout>,
	pub render_pass: ManuallyDrop<backend::RenderPass>,

	pub command_pool: ManuallyDrop<backend::CommandPool>,
	pub command_buffer: ManuallyDrop<backend::CommandBuffer>,

	pub render_semaphore: ManuallyDrop<backend::Semaphore>,

	pub texture_descriptor_set_layout: ManuallyDrop<backend::DescriptorSetLayout>,
	pub descriptor_pool: ManuallyDrop<backend::DescriptorPool>,
	
	pub extent: Cell<gfx_hal::window::Extent2D>,
}

impl RendererContext {
	fn new(format: gfx_hal::format::Format, extent: gfx_hal::window::Extent2D) -> RendererContext {
		//Note: Keep up-to-date.         X0.X4.X0_XX
		const POLYSTRIP_VERSION: u32 = 0x00_04_00_00;
		let instance = backend::Instance::create("polystrip", POLYSTRIP_VERSION).unwrap();

		let adapter = instance.enumerate_adapters().into_iter()
			.find(|adapter| {
				adapter.queue_families.iter()
					.any(|family| family.queue_type().supports_graphics())
			})
			.unwrap();

		let gpu = unsafe {
			adapter.physical_device.open(
				&[(
					adapter.queue_families.iter()
						.find(|family| family.queue_type().supports_graphics()).unwrap(),
					&[0.9]
				)],
				gfx_hal::Features::empty()
			).unwrap()		
		};

		let (command_pool, command_buffer) = unsafe {
			let mut command_pool = gpu.device.create_command_pool(gpu.queue_groups[0].family, gfx_hal::pool::CommandPoolCreateFlags::empty()).unwrap();
			let command_buffer = command_pool.allocate_one(gfx_hal::command::Level::Primary);
			(command_pool, command_buffer)
		};

		let texture_descriptor_set_layout = unsafe { gpu.device.create_descriptor_set_layout(
			&[
				gfx_hal::pso::DescriptorSetLayoutBinding {
					binding: 0,
					ty: gfx_hal::pso::DescriptorType::Image {
						ty: gfx_hal::pso::ImageDescriptorType::Sampled {
							with_sampler: true,
						},
					},
					count: 1,
					stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
					immutable_samplers: false,
				},
				gfx_hal::pso::DescriptorSetLayoutBinding {
					binding: 1,
					ty: gfx_hal::pso::DescriptorType::Sampler,
					count: 1,
					stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
					immutable_samplers: false,
				}
			],
			&[]
		)}.unwrap();
		let descriptor_pool = unsafe { gpu.device.create_descriptor_pool(
			1024,
			&[
				gfx_hal::pso::DescriptorRangeDesc {
					ty: gfx_hal::pso::DescriptorType::Image {
						ty: gfx_hal::pso::ImageDescriptorType::Sampled {
							with_sampler: true,
						},
					},
					count: 1,
				},
				gfx_hal::pso::DescriptorRangeDesc {
					ty: gfx_hal::pso::DescriptorType::Sampler,
					count: 1,
				},
			],
			gfx_hal::pso::DescriptorPoolCreateFlags::empty(),
		)}.unwrap();

		let main_pass = unsafe { gpu.device.create_render_pass(
			&[gfx_hal::pass::Attachment {
				format: Some(format),
				samples: 1,
				ops: gfx_hal::pass::AttachmentOps::DONT_CARE, //TODO: Potential customisation
				stencil_ops: gfx_hal::pass::AttachmentOps::DONT_CARE,
				layouts: gfx_hal::image::Layout::Undefined..gfx_hal::image::Layout::Present,
			}],
			&[gfx_hal::pass::SubpassDesc {
				colors: &[(0, gfx_hal::image::Layout::ColorAttachmentOptimal)],
				depth_stencil: None,
				inputs: &[],
				resolves: &[],
				preserves: &[],
			}],
			&[]
		)}.unwrap();

		let colour_vs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(COLOURED_VERT_SPV)) }.unwrap();
		let colour_fs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(COLOURED_FRAG_SPV)) }.unwrap();

		let colour_graphics_pipeline_layout = unsafe { gpu.device.create_pipeline_layout(&[], &[]) }.unwrap();
		let colour_graphics_pipeline = unsafe { gpu.device.create_graphics_pipeline(&gfx_hal::pso::GraphicsPipelineDesc {
			primitive_assembler: gfx_hal::pso::PrimitiveAssemblerDesc::Vertex {
				buffers: &[gfx_hal::pso::VertexBufferDesc {
					binding: 0,
					stride: std::mem::size_of::<ColorVertex>() as u32,
					rate: gfx_hal::pso::VertexInputRate::Vertex,
				}],
				attributes: ColorVertex::desc(),
				input_assembler: gfx_hal::pso::InputAssemblerDesc {
					primitive: gfx_hal::pso::Primitive::TriangleList,
					with_adjacency: false,
					restart_index: None,
				},
				vertex: gfx_hal::pso::EntryPoint {
					entry: "main",
					module: &colour_vs_module,
					specialization: gfx_hal::pso::Specialization {
						constants: std::borrow::Cow::Borrowed(&[gfx_hal::pso::SpecializationConstant {
							id: 0,
							range: 0..1
						}]),
						data: std::borrow::Cow::Borrowed(&[cfg!(any(feature = "metal", feature = "dx12")) as u8]),
					}
				},
				tessellation: None,
				geometry: None,
			},
			rasterizer: gfx_hal::pso::Rasterizer {
				polygon_mode: gfx_hal::pso::PolygonMode::Fill,
				cull_face: gfx_hal::pso::Face::BACK,
				front_face: gfx_hal::pso::FrontFace::CounterClockwise,
				depth_clamping: false,
				depth_bias: None,
				conservative: false,
				line_width: gfx_hal::pso::State::Dynamic,
			},
			fragment: Some(gfx_hal::pso::EntryPoint {
				entry: "main",
				module: &colour_fs_module,
				specialization: gfx_hal::pso::Specialization {
					constants: std::borrow::Cow::Borrowed(&[]),
					data: std::borrow::Cow::Borrowed(&[]),
				}
			}),
			blender: gfx_hal::pso::BlendDesc {
				logic_op: None,
				targets: vec![gfx_hal::pso::ColorBlendDesc {
					mask: gfx_hal::pso::ColorMask::ALL,
					blend: Some(gfx_hal::pso::BlendState {
						color: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						},
						alpha: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						}
					})
				}]
			},
			depth_stencil: gfx_hal::pso::DepthStencilDesc {
				depth: None,
				depth_bounds: false,
				stencil: None,
			},
			multisampling: None,
			baked_states: gfx_hal::pso::BakedStates {
				viewport: None,
				scissor: None,
				blend_color: None,
				depth_bounds: None,
			},
			layout: &colour_graphics_pipeline_layout,
			subpass: gfx_hal::pass::Subpass {
				index: 0,
				main_pass: &main_pass,
			},
			flags: gfx_hal::pso::PipelineCreationFlags::empty(),
			parent: gfx_hal::pso::BasePipeline::None,
		}, None) }.unwrap();

		let texture_vs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(TEXTURED_VERT_SPV)) }.unwrap();
		let texture_fs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(TEXTURED_FRAG_SPV)) }.unwrap();

		let texture_graphics_pipeline_layout = unsafe { gpu.device.create_pipeline_layout(vec![&texture_descriptor_set_layout], &[]) }.unwrap();
		let texture_graphics_pipeline = unsafe { gpu.device.create_graphics_pipeline(&gfx_hal::pso::GraphicsPipelineDesc {
			primitive_assembler: gfx_hal::pso::PrimitiveAssemblerDesc::Vertex {
				buffers: &[gfx_hal::pso::VertexBufferDesc {
					binding: 0,
					stride: std::mem::size_of::<TextureVertex>() as u32,
					rate: gfx_hal::pso::VertexInputRate::Vertex,
				}],
				attributes: TextureVertex::desc(),
				input_assembler: gfx_hal::pso::InputAssemblerDesc {
					primitive: gfx_hal::pso::Primitive::TriangleList,
					with_adjacency: false,
					restart_index: None,
				},
				vertex: gfx_hal::pso::EntryPoint {
					entry: "main",
					module: &texture_vs_module,
					specialization: gfx_hal::pso::Specialization {
						constants: std::borrow::Cow::Borrowed(&[gfx_hal::pso::SpecializationConstant {
							id: 0,
							range: 0..1
						}]),
						data: std::borrow::Cow::Borrowed(&[cfg!(any(feature = "metal", feature = "dx12")) as u8]),
					}
				},
				tessellation: None,
				geometry: None,
			},
			rasterizer: gfx_hal::pso::Rasterizer {
				polygon_mode: gfx_hal::pso::PolygonMode::Fill,
				cull_face: gfx_hal::pso::Face::BACK,
				front_face: gfx_hal::pso::FrontFace::CounterClockwise,
				depth_clamping: false,
				depth_bias: None,
				conservative: false,
				line_width: gfx_hal::pso::State::Dynamic,
			},
			fragment: Some(gfx_hal::pso::EntryPoint {
				entry: "main",
				module: &texture_fs_module,
				specialization: gfx_hal::pso::Specialization {
					constants: std::borrow::Cow::Borrowed(&[]),
					data: std::borrow::Cow::Borrowed(&[]),
				}
			}),
			blender: gfx_hal::pso::BlendDesc {
				logic_op: None,
				targets: vec![gfx_hal::pso::ColorBlendDesc {
					mask: gfx_hal::pso::ColorMask::ALL,
					blend: Some(gfx_hal::pso::BlendState {
						color: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						},
						alpha: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						}
					})
				}]
			},
			depth_stencil: gfx_hal::pso::DepthStencilDesc {
				depth: None,
				depth_bounds: false,
				stencil: None,
			},
			multisampling: None,
			baked_states: gfx_hal::pso::BakedStates {
				viewport: None,
				scissor: None,
				blend_color: None,
				depth_bounds: None,
			},
			layout: &texture_graphics_pipeline_layout,
			subpass: gfx_hal::pass::Subpass {
				index: 0,
				main_pass: &main_pass,
			},
			flags: gfx_hal::pso::PipelineCreationFlags::empty(),
			parent: gfx_hal::pso::BasePipeline::None,
		}, None) }.unwrap();

		let render_semaphore = gpu.device.create_semaphore().unwrap();

		RendererContext {
			instance, gpu, adapter,

			colour_graphics_pipeline: ManuallyDrop::new(colour_graphics_pipeline),
			texture_graphics_pipeline: ManuallyDrop::new(texture_graphics_pipeline),
			texture_graphics_pipeline_layout: ManuallyDrop::new(texture_graphics_pipeline_layout),
			render_pass: ManuallyDrop::new(main_pass),

			command_pool: ManuallyDrop::new(command_pool),
			command_buffer: ManuallyDrop::new(command_buffer),

			render_semaphore: ManuallyDrop::new(render_semaphore),

			texture_descriptor_set_layout: ManuallyDrop::new(texture_descriptor_set_layout),
			descriptor_pool: ManuallyDrop::new(descriptor_pool),

			extent: Cell::new(extent),
		}
	}
}

impl Drop for RendererContext {
	fn drop(&mut self) {
		unsafe {
			let mut command_pool = ManuallyDrop::take(&mut self.command_pool);
			command_pool.free(std::iter::once(ManuallyDrop::take(&mut self.command_buffer)));
			self.gpu.device.destroy_command_pool(command_pool);

			self.gpu.device.destroy_semaphore(ManuallyDrop::take(&mut self.render_semaphore));

			self.gpu.device.destroy_descriptor_set_layout(ManuallyDrop::take(&mut self.texture_descriptor_set_layout));
			self.gpu.device.destroy_descriptor_pool(ManuallyDrop::take(&mut self.descriptor_pool));

			self.gpu.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.colour_graphics_pipeline));
			self.gpu.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.texture_graphics_pipeline));
			self.gpu.device.destroy_pipeline_layout(ManuallyDrop::take(&mut self.texture_graphics_pipeline_layout));
			
			self.gpu.device.destroy_render_pass(ManuallyDrop::take(&mut self.render_pass));
		}
	}
}

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
	context: Rc<RendererContext>,
	surface: backend::Surface,
	swapchain_config: gfx_hal::window::SwapchainConfig,
}

//TODO: Builder pattern, to allow for more configuration?
impl Renderer {
	/// Creates a new renderer, initialising the `gfx_hal` backend. This method assumes the raw window handle
	/// was created legitimately. *Technically*, that's my problem, but if you're not making your window properly, I'm not
	/// going to take responsibility for the resulting crash. (The only way I'd be able to deal with it anyway would be to
	/// mark this method `unsafe`)
	/// 
	/// # Arguments
	/// * `window`: A valid window compatible with `raw_window_handle`.
	/// * `size`: The size of the window in pixels, in the order (width, height). For window implementations which
	///           differentiate between physical and logical size, this refers to the logical size
	pub fn new(window: &impl HasRawWindowHandle, (width, height): (u32, u32)) -> Renderer {
		//MARK: New Renderer
		let swapchain_config = gfx_hal::window::SwapchainConfig::new(width, height, gfx_hal::format::Format::Bgra8Srgb, 2);

		let mut context = RendererContext::new(swapchain_config.format, swapchain_config.extent);

		let mut surface = unsafe { context.instance.create_surface(window).unwrap() };
		
		unsafe { surface.configure_swapchain(&context.gpu.device, swapchain_config.clone()).unwrap(); }

		Renderer {
			context: Rc::new(context),

			surface, swapchain_config,
		}
	}

	/// Returns the next `Frame`, which can be drawn to and will present on drop. This `Renderer` is borrowed mutably while the
	/// frame is alive. Any operations on this renderer must be done through the `Frame`, which implements `Deref<Target = Renderer>`.
	pub fn next_frame(&mut self) -> Frame<'_> {
		self.generate_frame(self.acquire_image(), None)
	}

	pub fn next_frame_clear(&mut self, clear_color: Color) -> Frame<'_> {
		self.generate_frame(self.acquire_image(), Some(clear_color))
	}

	fn acquire_image(&mut self) -> backend::SwapchainImage {
		match unsafe { self.surface.acquire_image(1_000_000 /* 1 ms */) } {
			Ok((image, _)) => image,
			Err(gfx_hal::window::AcquireError::OutOfDate) => {
				unsafe { self.surface.configure_swapchain(&self.context.gpu.device, self.swapchain_config.clone()) }.unwrap();
				match unsafe { self.surface.acquire_image(0) } {
					Ok((image, _)) => image,
					Err(e) => panic!("{}", e),
				}
			},
			Err(e) => panic!("{}", e),
		}
	}

	fn generate_frame(&mut self, image: backend::SwapchainImage, clear_colour: Option<Color>) -> Frame<'_> {
		use std::borrow::Borrow;

		let viewport = gfx_hal::pso::Viewport {
			rect: gfx_hal::pso::Rect {
				x: 0,
				y: 0,
				w: self.swapchain_config.extent.width as i16,
				h: self.swapchain_config.extent.height as i16,
			},
			depth: 0.0..1.0,
		};
		let framebuffer = unsafe { self.context.gpu.device.create_framebuffer(&self.context.render_pass, vec![image.borrow()], self.swapchain_config.extent.to_extent()) }.unwrap();

		let clear_colour_linear = clear_colour.map(|clear_colour| gfx_hal::command::ClearColor {
			float32: [
				(clear_colour.r as f32).powi(2) / 65_025.0,
				(clear_colour.g as f32).powi(2) / 65_025.0,
				(clear_colour.b as f32).powi(2) / 65_025.0,
				clear_colour.a as f32 / 255.0,
			]
		});

		unsafe {
			self.context.command_buffer.reset(false);

			self.context.command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
					
			self.context.command_buffer.set_viewports(0, &[viewport.clone()]);
			self.context.command_buffer.set_scissors(0, &[viewport.rect]);

			self.context.command_buffer.begin_render_pass(
				&self.context.render_pass,
				&framebuffer,
				viewport.rect,
				if let Some(c) = clear_colour_linear {
					&[gfx_hal::command::ClearValue { color: c }] as &[gfx_hal::command::ClearValue]
				} else {
					&[] as _
				},
				gfx_hal::command::SubpassContents::Inline
			);

			if let Some(c) = clear_colour_linear {
				self.context.command_buffer.clear_attachments(
					&[gfx_hal::command::AttachmentClear::Color { index: 0, value: c }],
					&[gfx_hal::pso::ClearRect { rect: viewport.rect, layers: 0..1 }]
				);
			}
		}

		Frame {
			renderer: self,
			swap_chain_frame: ManuallyDrop::new(image),
			framebuffer: ManuallyDrop::new(framebuffer),
			viewport,
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
		self.swapchain_config.extent.width = size.0;
		self.swapchain_config.extent.height = size.1;
		self.context.extent.set(self.swapchain_config.extent);
		unsafe { self.surface.configure_swapchain(&self.context.gpu.device, self.swapchain_config.clone()) }.unwrap();
	}

	/// Gets the width of the internal swapchain, which is updated every time [`resize`](#method.resize) is called
	pub fn width(&self) -> u32 {
		self.swapchain_config.extent.width
	}

	/// Gets the height of the internal swapchain, which is updated every time [`resize`](#method.resize) is called
	pub fn height(&self) -> u32 {
		self.swapchain_config.extent.height
	}

	/// Converts pixel coordinates to Gpu coordinates
	pub fn pixel(&self, x: i32, y: i32) -> GpuPos {
		GpuPos {
			x: (x * 2) as f32 / self.swapchain_config.extent.width as f32 - 1.0,
			y: -((y * 2) as f32 / self.swapchain_config.extent.height as f32 - 1.0),
		}
	}
}

impl Drop for Renderer {
	fn drop(&mut self) {
		unsafe {
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
	swap_chain_frame: std::mem::ManuallyDrop<<backend::Surface as gfx_hal::window::PresentationSurface<backend::Backend>>::SwapchainImage>,
	framebuffer: std::mem::ManuallyDrop<backend::Framebuffer>,
	viewport: gfx_hal::pso::Viewport,
}

//MARK: Frame API
impl<'a> Frame<'a> {
	/// Draws a [`ColoredShape`](../vertex/struct.ColoredShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	pub fn draw_colored(&mut self, shape: &'a ColoredShape) {
		unsafe {
			self.renderer.context.command_buffer.bind_vertex_buffers(0, vec![(&shape.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			self.renderer.context.command_buffer.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
				buffer: &shape.index_buffer,
				range: gfx_hal::buffer::SubRange::WHOLE,
				index_type: gfx_hal::IndexType::U16,
			});

			self.renderer.context.command_buffer.begin_render_pass(&self.renderer.context.render_pass, &self.framebuffer, self.viewport.rect, &[], gfx_hal::command::SubpassContents::Inline);
			self.renderer.context.command_buffer.bind_graphics_pipeline(&self.renderer.context.colour_graphics_pipeline);
			self.renderer.context.command_buffer.draw_indexed(0..shape.index_count, 0, 0..1);
		}
	}

	/// Draws a [`TexturedShape`](../vertex/struct.TexturedShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	/// 
	/// # Arguments
	/// * `shape`: The `TexturedShape` to be rendered. 
	/// * `texture`: The `Texture` to be drawn to the geometry of the shape.
	pub fn draw_textured(&mut self, shape: &'a TexturedShape, texture: &'a Texture) {
		unsafe {
			self.renderer.context.command_buffer.bind_vertex_buffers(0, vec![(&shape.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			self.renderer.context.command_buffer.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
				buffer: &shape.index_buffer,
				range: gfx_hal::buffer::SubRange::WHOLE,
				index_type: gfx_hal::IndexType::U16,
			});
			
			self.renderer.context.command_buffer.begin_render_pass(&self.renderer.context.render_pass, &self.framebuffer, self.viewport.rect, &[], gfx_hal::command::SubpassContents::Inline);
			self.renderer.context.command_buffer.bind_graphics_pipeline(&self.renderer.context.texture_graphics_pipeline);
			self.renderer.context.command_buffer.bind_graphics_descriptor_sets(&self.renderer.context.texture_graphics_pipeline_layout, 0, vec![&texture.descriptor_set], &[0]);
			self.renderer.context.command_buffer.draw_indexed(0..shape.index_count as u32, 0, 0..1);
		}
	}

	/// Converts pixel coordinates to Gpu coordinates
	/// 
	/// This is a copy of the same function in `Renderer`.
	pub fn pixel(&self, x: i32, y: i32) -> GpuPos {
		GpuPos {
			x: (x * 2) as f32 / self.viewport.rect.w as f32 - 1.0,
			y: -((y * 2) as f32 / self.viewport.rect.h as f32 - 1.0),
		}
	}
}

impl<'a> Drop for Frame<'a> {
	fn drop(&mut self) {
		if !std::thread::panicking() {
			self.renderer.context.command_buffer.end_render_pass();
			self.renderer.context.command_buffer.finish();

			self.renderer.context.gpu.queue_groups[0].queues[0].submit(
				gfx_hal::queue::Submission {
					command_buffers: vec![&*self.renderer.context.command_buffer],
					wait_semaphores: vec![],
					signal_semaphores: vec![&*self.renderer.context.render_semaphore],
				},
				None
			);

			unsafe {
				self.renderer.context.gpu.queue_groups[0].queues[0].present(&mut self.renderer.surface, ManuallyDrop::take(&mut self.swap_chain_frame), None).unwrap();
			}
		} else {
			unsafe {
				ManuallyDrop::drop(&mut self.swap_chain_frame);
			}
		}
		unsafe {
			self.renderer.context.gpu.device.destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));
		}
	}
}

impl<'a> std::ops::Deref for Frame<'a> {
	type Target = Renderer;

	fn deref(&self) -> &Renderer {
		&self.renderer
	}
}