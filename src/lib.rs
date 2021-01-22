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
//! # use polystrip::Renderer;
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
//!             let mut frame = renderer.next_frame();
//!             // Render in here
//!         },
//!         _ => {}
//!     }
//! });
//! ```

pub(crate) mod backend;
pub mod data;
pub mod geometry;
pub mod vertex;

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::rc::Rc;

use gpu_alloc::{GpuAllocator, MemoryBlock, Request, UsageFlags};
use gpu_alloc_gfx::GfxMemoryDevice;

use crate::data::{GpuPos, Color};
use crate::vertex::*;

use raw_window_handle::HasRawWindowHandle;

use gfx_hal::prelude::*;

use align_data::{include_aligned, Align32};
static COLOURED_VERT_SPV: &[u8] = include_aligned!(Align32, "spirv/coloured.vert.spv");
static COLOURED_FRAG_SPV: &[u8] = include_aligned!(Align32, "spirv/coloured.frag.spv");
static TEXTURED_VERT_SPV: &[u8] = include_aligned!(Align32, "spirv/textured.vert.spv");
static TEXTURED_FRAG_SPV: &[u8] = include_aligned!(Align32, "spirv/textured.frag.spv");

pub struct RendererContext {
	pub instance: backend::Instance,
	pub device: backend::Device,
	pub queue_groups: RefCell<Vec<gfx_hal::queue::family::QueueGroup<backend::Backend>>>,
	pub adapter: gfx_hal::adapter::Adapter<backend::Backend>,

	pub colour_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	pub texture_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	pub texture_graphics_pipeline_layout: ManuallyDrop<backend::PipelineLayout>,
	pub render_pass: ManuallyDrop<backend::RenderPass>,

	pub command_pool: RefCell<ManuallyDrop<backend::CommandPool>>,
	pub command_buffer: RefCell<ManuallyDrop<backend::CommandBuffer>>,

	pub render_semaphore: ManuallyDrop<backend::Semaphore>,

	pub texture_descriptor_set_layout: ManuallyDrop<backend::DescriptorSetLayout>,
	pub descriptor_pool: RefCell<ManuallyDrop<backend::DescriptorPool>>,
	
	pub extent: Cell<gfx_hal::window::Extent2D>,

	pub allocator: RefCell<GpuAllocator<std::sync::Arc<backend::Memory>>>,
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
					count: 1024,
				},
				gfx_hal::pso::DescriptorRangeDesc {
					ty: gfx_hal::pso::DescriptorType::Sampler,
					count: 1024,
				},
			],
			gfx_hal::pso::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
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

		let alloc_props = gpu_alloc_gfx::gfx_device_properties(&adapter);
		let allocator = GpuAllocator::new(
			gpu_alloc::Config::i_am_prototyping(), //TODO: Choose sensible defaults
			alloc_props,
		);

		RendererContext {
			instance, adapter,
			device: gpu.device,
			queue_groups: RefCell::new(gpu.queue_groups),

			colour_graphics_pipeline: ManuallyDrop::new(colour_graphics_pipeline),
			texture_graphics_pipeline: ManuallyDrop::new(texture_graphics_pipeline),
			texture_graphics_pipeline_layout: ManuallyDrop::new(texture_graphics_pipeline_layout),
			render_pass: ManuallyDrop::new(main_pass),

			command_pool: RefCell::new(ManuallyDrop::new(command_pool)),
			command_buffer: RefCell::new(ManuallyDrop::new(command_buffer)),

			render_semaphore: ManuallyDrop::new(render_semaphore),

			texture_descriptor_set_layout: ManuallyDrop::new(texture_descriptor_set_layout),
			descriptor_pool: RefCell::new(ManuallyDrop::new(descriptor_pool)),

			extent: Cell::new(extent),

			allocator: RefCell::new(allocator),
		}
	}
}

impl Drop for RendererContext {
	fn drop(&mut self) {
		unsafe {
			self.allocator.get_mut().cleanup(GfxMemoryDevice::wrap(&self.device));

			let mut command_pool = ManuallyDrop::take(self.command_pool.get_mut());
			command_pool.free(std::iter::once(ManuallyDrop::take(self.command_buffer.get_mut())));
			self.device.destroy_command_pool(command_pool);

			self.device.destroy_semaphore(ManuallyDrop::take(&mut self.render_semaphore));

			self.device.destroy_descriptor_set_layout(ManuallyDrop::take(&mut self.texture_descriptor_set_layout));
			self.device.destroy_descriptor_pool(ManuallyDrop::take(self.descriptor_pool.get_mut()));

			self.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.colour_graphics_pipeline));
			self.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.texture_graphics_pipeline));
			self.device.destroy_pipeline_layout(ManuallyDrop::take(&mut self.texture_graphics_pipeline_layout));
			
			self.device.destroy_render_pass(ManuallyDrop::take(&mut self.render_pass));
		}
	}
}

/// An accelerated 2D renderer.
/// 
/// A renderer can be created for any window compatible with `raw_window_handle`. The size of this window must be updated
/// in the event loop, and specified on creation. For example, using `winit`:
/// ```no_run
/// # use winit::event::{Event, WindowEvent};
/// # use polystrip::Renderer;
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
	surface: ManuallyDrop<backend::Surface>,
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
		let swapchain_config = gfx_hal::window::SwapchainConfig::new(width, height, gfx_hal::format::Format::Bgra8Srgb, 2);
		let context = RendererContext::new(swapchain_config.format, swapchain_config.extent);

		let mut surface = unsafe { context.instance.create_surface(window).unwrap() };
		unsafe { surface.configure_swapchain(&context.device, swapchain_config.clone()).unwrap(); }

		Renderer {
			context: Rc::new(context),
			surface: ManuallyDrop::new(surface),
			swapchain_config,
		}
	}

	/// Returns the next `Frame`, which can be drawn to and will present on drop. This `Renderer` is borrowed mutably while the
	/// frame is alive.
	pub fn next_frame(&mut self) -> Frame<'_> {
		let image = self.acquire_image();
		self.generate_frame(image, None)
	}

	pub fn next_frame_clear(&mut self, clear_color: Color) -> Frame<'_> {
		let image = self.acquire_image();
		self.generate_frame(image, Some(clear_color))
	}


	fn acquire_image(&mut self) -> backend::SwapchainImage {
		match unsafe { self.surface.acquire_image(1_000_000 /* 1 ms */) } {
			Ok((image, _)) => image,
			Err(gfx_hal::window::AcquireError::OutOfDate) => {
				unsafe { self.surface.configure_swapchain(&self.context.device, self.swapchain_config.clone()) }.unwrap();
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
		let framebuffer = unsafe { self.context.device.create_framebuffer(&self.context.render_pass, vec![image.borrow()], self.swapchain_config.extent.to_extent()) }.unwrap();

		let clear_colour_linear = clear_colour.map(|clear_colour| gfx_hal::command::ClearColor {
			float32: [
				(clear_colour.r as f32).powi(2) / 65_025.0,
				(clear_colour.g as f32).powi(2) / 65_025.0,
				(clear_colour.b as f32).powi(2) / 65_025.0,
				clear_colour.a as f32 / 255.0,
			]
		});

		let mut command_buffer = self.context.command_buffer.borrow_mut();

		unsafe {
			command_buffer.reset(false);

			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
					
			command_buffer.set_viewports(0, &[viewport.clone()]);
			command_buffer.set_scissors(0, &[viewport.rect]);

			if let Some(c) = clear_colour_linear {
				command_buffer.begin_render_pass(
					&self.context.render_pass,
					&framebuffer,
					viewport.rect,
					&[gfx_hal::command::ClearValue { color: c }] as &[gfx_hal::command::ClearValue],
					gfx_hal::command::SubpassContents::Inline
				);

				command_buffer.clear_attachments(
					&[gfx_hal::command::AttachmentClear::Color { index: 0, value: c }],
					&[gfx_hal::pso::ClearRect { rect: viewport.rect, layers: 0..1 }]
				);
			} else {
				command_buffer.begin_render_pass(
					&self.context.render_pass,
					&framebuffer,
					viewport.rect,
					&[],
					gfx_hal::command::SubpassContents::Inline
				);
			}
		}

		std::mem::drop(command_buffer);

		Frame {
			renderer: self,
			swap_chain_frame: ManuallyDrop::new(image),
			framebuffer: ManuallyDrop::new(framebuffer),
			viewport,
			allocations: Vec::new(),
		}
	}

	/// Create a new texture from the given rgba data, associated with this `Renderer`.
	/// 
	/// # Arguments
	/// * `data`: A reference to a byte array containing the pixel data. The data must be formatted to `Rgba8` in
	///           the sRGB color space, in row-major order.
	/// * `size`: The size of the texture, in pixels, in (width, height) order.
	pub fn texture_from_rgba(&self, data: &[u8], (width, height): (u32, u32)) -> Texture {
		let context = self.context.clone();

		let descriptor_set = unsafe { context.descriptor_pool.borrow_mut().allocate_set(&self.context.texture_descriptor_set_layout) }.unwrap();
		let mut command_buffer = unsafe { context.command_pool.borrow_mut().allocate_one(gfx_hal::command::Level::Primary) };
		let memory_device = GfxMemoryDevice::wrap(&context.device);

		let mut image = unsafe { context.device.create_image(
			gfx_hal::image::Kind::D2(width, height, 1, 1),
			1,
			gfx_hal::format::Format::Rgba8Srgb,
			gfx_hal::image::Tiling::Optimal,
			gfx_hal::image::Usage::TRANSFER_DST | gfx_hal::image::Usage::SAMPLED,
			gfx_hal::image::ViewCapabilities::empty(),
		)}.unwrap();
		let img_req = unsafe { context.device.get_image_requirements(&image) };
		let img_block = unsafe { context.allocator.borrow_mut().alloc(
			memory_device,
			Request {
				size: img_req.size,
				align_mask: img_req.alignment,
				memory_types: img_req.type_mask,
				usage: UsageFlags::FAST_DEVICE_ACCESS,
			}
		)}.unwrap();

		unsafe {
			context.device.bind_image_memory(&img_block.memory(), img_block.offset(), &mut image).unwrap();
		}

		let mut buffer = unsafe { context.device.create_buffer(img_req.size, gfx_hal::buffer::Usage::TRANSFER_SRC) }.unwrap();
		let buf_req = unsafe { context.device.get_buffer_requirements(&buffer) };
		let buf_block = unsafe { context.allocator.borrow_mut().alloc(
			memory_device,
			Request {
				size: buf_req.size,
				align_mask: buf_req.alignment,
				memory_types: buf_req.type_mask,
				usage: UsageFlags::UPLOAD | UsageFlags::TRANSIENT,
			}
		)}.unwrap();

		unsafe {
			buf_block.write_bytes(memory_device, 0, data).unwrap();
			context.device.bind_buffer_memory(&buf_block.memory(), buf_block.offset(), &mut buffer).unwrap();

			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TOP_OF_PIPE..gfx_hal::pso::PipelineStage::TRANSFER,
				gfx_hal::memory::Dependencies::empty(),
				&[gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::empty(), gfx_hal::image::Layout::Undefined)
						..
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal),
					target: &image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.copy_buffer_to_image(
				&buffer,
				&image,
				gfx_hal::image::Layout::TransferDstOptimal,
				&[gfx_hal::command::BufferImageCopy {
					buffer_offset: 0,
					buffer_width: width,
					buffer_height: height,
					image_layers: gfx_hal::image::SubresourceLayers {
						aspects: gfx_hal::format::Aspects::COLOR,
						level: 0,
						layers: 0..1,
					},
					image_offset: gfx_hal::image::Offset::ZERO,
					image_extent: gfx_hal::image::Extent {
						width, height,
						depth: 1,
					}
				}]
			);
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TRANSFER..gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
				gfx_hal::memory::Dependencies::empty(),
				&[gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal)
						..
						(gfx_hal::image::Access::SHADER_READ, gfx_hal::image::Layout::ShaderReadOnlyOptimal),
					target: &image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.finish();

			let fence = context.device.create_fence(false).unwrap();
			context.queue_groups.borrow_mut()[0].queues[0].submit_without_semaphores(&[command_buffer], Some(&fence));
			context.device.wait_for_fence(&fence, u64::MAX).unwrap();

			context.device.destroy_fence(fence);
		}
		
		unsafe {
			context.allocator.borrow_mut().dealloc(
				GfxMemoryDevice::wrap(&context.device),
				buf_block
			);
			context.device.destroy_buffer(buffer);
		}

		let view = unsafe { context.device.create_image_view(
			&image,
			gfx_hal::image::ViewKind::D2,
			gfx_hal::format::Format::Rgba8Srgb,
			gfx_hal::format::Swizzle::NO,
			gfx_hal::image::SubresourceRange {
				aspects: gfx_hal::format::Aspects::COLOR,
				level_start: 0,
				level_count: None,
				layer_start: 0,
				layer_count: None,
			},
		)}.unwrap();

		let sampler = unsafe { context.device.create_sampler(&gfx_hal::image::SamplerDesc::new(gfx_hal::image::Filter::Nearest, gfx_hal::image::WrapMode::Tile)) }.unwrap();

		unsafe {
			context.device.write_descriptor_sets(vec![gfx_hal::pso::DescriptorSetWrite {
				set: &descriptor_set,
				binding: 0,
				array_offset: 0,
				descriptors: &[
					gfx_hal::pso::Descriptor::Image(&view, gfx_hal::image::Layout::General),
					gfx_hal::pso::Descriptor::Sampler(&sampler),
				]
			}])
		}

		Texture {
			context,
			image: ManuallyDrop::new(image),
			view: ManuallyDrop::new(view),
			sampler: ManuallyDrop::new(sampler),
			descriptor_set: ManuallyDrop::new(descriptor_set),
			memory_block: ManuallyDrop::new(img_block),
			width, height,
		}
	}
	
	/// Resizes the internal swapchain
	/// 
	/// Call this method in your window's event loop whenever the window gets resized
	/// 
	/// # Arguments
	/// * `size`: The size of the window in pixels, in the order (width, height). For window implementations which
	///           differentiate between physical and logical size, this refers to the logical size
	pub fn resize(&mut self, size: (u32, u32)) {
		self.swapchain_config.extent.width = size.0;
		self.swapchain_config.extent.height = size.1;
		self.context.extent.set(self.swapchain_config.extent);
		unsafe { self.surface.configure_swapchain(&self.context.device, self.swapchain_config.clone()) }.unwrap();
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
			self.context.instance.destroy_surface(ManuallyDrop::take(&mut self.surface));
		}
	}
}

/// A frame to be drawn to. The frame gets presented on drop.
/// 
/// More methods are implemented in the [`FrameGeometryExt`](geometry/trait.FrameGeometryExt.html) trait.
pub struct Frame<'a> {
	renderer: &'a mut Renderer,
	swap_chain_frame: ManuallyDrop<<backend::Surface as gfx_hal::window::PresentationSurface<backend::Backend>>::SwapchainImage>,
	framebuffer: ManuallyDrop<backend::Framebuffer>,
	viewport: gfx_hal::pso::Viewport,
	allocations: Vec<MemoryBlock<std::sync::Arc<backend::Memory>>>,
}

//MARK: Frame API
impl<'a> Frame<'a> {
	fn create_staging_buffers(&mut self, vertices: &[u8], indices: &[u8]) -> (backend::Buffer, backend::Buffer) {
		let mut vertex_buffer = unsafe {
			self.renderer.context.device.create_buffer(vertices.len() as u64, gfx_hal::buffer::Usage::VERTEX)
		}.unwrap();
		let mut index_buffer = unsafe {
			self.renderer.context.device.create_buffer(indices.len() as u64, gfx_hal::buffer::Usage::INDEX)
		}.unwrap();
		let vertex_mem_req = unsafe { self.renderer.context.device.get_buffer_requirements(&vertex_buffer) };
		let index_mem_req = unsafe { self.renderer.context.device.get_buffer_requirements(&index_buffer) };

		let memory_device = GfxMemoryDevice::wrap(&self.renderer.context.device);
		let vertex_block = unsafe {
			self.renderer.context.allocator.borrow_mut().alloc(
				memory_device,
				Request {
					size: vertex_mem_req.size,
					align_mask: vertex_mem_req.alignment,
					memory_types: vertex_mem_req.type_mask,
					usage: UsageFlags::UPLOAD | UsageFlags::TRANSIENT, // Implies host-visible
				}
			)
		}.unwrap();
		let index_block = unsafe {
			self.renderer.context.allocator.borrow_mut().alloc(
				memory_device,
				Request {
					size: index_mem_req.size,
					align_mask: index_mem_req.alignment,
					memory_types: index_mem_req.type_mask,
					usage: UsageFlags::UPLOAD | UsageFlags::TRANSIENT,
				}
			)
		}.unwrap();
		unsafe {
			vertex_block.write_bytes(memory_device, 0, vertices).unwrap();
			index_block.write_bytes(memory_device, 0, indices).unwrap();
			self.renderer.context.device.bind_buffer_memory(&vertex_block.memory(), vertex_block.offset(), &mut vertex_buffer).unwrap();
			self.renderer.context.device.bind_buffer_memory(&index_block.memory(), index_block.offset(), &mut index_buffer).unwrap();
		}
		self.allocations.push(vertex_block);
		self.allocations.push(index_block);
		(vertex_buffer, index_buffer)
	}

	/// Draws a [`ColoredShape`](vertex/struct.ColoredShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	pub fn draw_colored(&mut self, shape: ColoredShape<'_>) {
		let (vertex_buffer, index_buffer) = self.create_staging_buffers(bytemuck::cast_slice(shape.vertices), bytemuck::cast_slice(shape.indices));
		let mut command_buffer = self.renderer.context.command_buffer.borrow_mut();

		unsafe {
			command_buffer.bind_vertex_buffers(0, vec![(&vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			command_buffer.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
				buffer: &index_buffer,
				range: gfx_hal::buffer::SubRange::WHOLE,
				index_type: gfx_hal::IndexType::U16,
			});

			command_buffer.bind_graphics_pipeline(&self.renderer.context.colour_graphics_pipeline);
			command_buffer.draw_indexed(0..shape.indices.len() as u32 * 3, 0, 0..1);
		}
	}

	/// Draws a [`TexturedShape`](vertex/struct.TexturedShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	/// 
	/// # Arguments
	/// * `shape`: The `TexturedShape` to be rendered. 
	/// * `texture`: The `Texture` to be drawn to the geometry of the shape.
	pub fn draw_textured(&mut self, shape: TexturedShape<'_>, texture: &'a Texture) {
		if !Rc::ptr_eq(&self.renderer.context, &texture.context) {
			panic!("Texture was not made with renderer that made this frame");
		}

		let (vertex_buffer, index_buffer) = self.create_staging_buffers(bytemuck::cast_slice(shape.vertices), bytemuck::cast_slice(shape.indices));
		let mut command_buffer = self.renderer.context.command_buffer.borrow_mut();

		unsafe {
			command_buffer.bind_vertex_buffers(0, vec![(&vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			command_buffer.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
				buffer: &index_buffer,
				range: gfx_hal::buffer::SubRange::WHOLE,
				index_type: gfx_hal::IndexType::U16,
			});
			
			command_buffer.bind_graphics_pipeline(&self.renderer.context.texture_graphics_pipeline);
			command_buffer.bind_graphics_descriptor_sets(&self.renderer.context.texture_graphics_pipeline_layout, 0, vec![&*texture.descriptor_set], &[0]);
			command_buffer.draw_indexed(0..shape.indices.len() as u32 * 3, 0, 0..1);
		}
	}

	/// Converts pixel coordinates to Gpu coordinates
	pub fn pixel(&self, x: i32, y: i32) -> GpuPos {
		GpuPos {
			x: (x * 2) as f32 / self.viewport.rect.w as f32 - 1.0,
			y: -((y * 2) as f32 / self.viewport.rect.h as f32 - 1.0),
		}
	}
}

impl<'a> Drop for Frame<'a> {
	fn drop(&mut self) {
		unsafe {		
			let mut command_buffer = self.renderer.context.command_buffer.borrow_mut();
			command_buffer.end_render_pass();
			command_buffer.finish();
		}

		for block in self.allocations.drain(..) {
			unsafe {
				self.renderer.context.allocator.borrow_mut().dealloc(GfxMemoryDevice::wrap(&self.renderer.context.device), block);
			}
		}

		if !std::thread::panicking() {
			unsafe {
				let mut queue_groups = self.renderer.context.queue_groups.borrow_mut();
				queue_groups[0].queues[0].submit(
					gfx_hal::queue::Submission {
						command_buffers: vec![&**self.renderer.context.command_buffer.borrow()],
						wait_semaphores: vec![],
						signal_semaphores: vec![&*self.renderer.context.render_semaphore],
					},
					None
				);

				queue_groups[0].queues[0].present(&mut self.renderer.surface, ManuallyDrop::take(&mut self.swap_chain_frame), None).unwrap();
			}
		} else {
			unsafe {
				ManuallyDrop::drop(&mut self.swap_chain_frame);
			}
		}
		unsafe {
			self.renderer.context.device.destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));
		}
	}
}

/// A texture which can be copied to and rendered by a [`Frame`](struct.Frame.html).
/// 
/// It can be used only with the [`Renderer`](struct.Renderer.html) which created it.
pub struct Texture {
	context: Rc<RendererContext>,
	image: ManuallyDrop<backend::Image>,
	view: ManuallyDrop<backend::ImageView>,
	sampler: ManuallyDrop<backend::Sampler>,
	descriptor_set: ManuallyDrop<backend::DescriptorSet>,
	memory_block: ManuallyDrop<MemoryBlock<std::sync::Arc<backend::Memory>>>,
	width: u32,
	height: u32,
}

impl Texture {
	/// Get the dimensions of this texture, in (width, height) order.
	pub fn dimensions(&self) -> (u32, u32) {
		(self.width, self.height)
	}

	pub fn width(&self) -> u32 {
		self.width
	}

	pub fn height(&self) -> u32 {
		self.height
	}

	/// Converts pixel coordinates to Gpu coordinates
	pub fn pixel(&self, x: i32, y: i32) -> GpuPos {
		GpuPos {
			x: x as f32 / self.width as f32,
			y: y as f32 / self.height as f32,
		}
	}
}

impl Drop for Texture {
	fn drop(&mut self) {
		unsafe {
			self.context.descriptor_pool.borrow_mut().free(std::iter::once(ManuallyDrop::take(&mut self.descriptor_set)));
		}
		
		unsafe {
			self.context.device.destroy_sampler(ManuallyDrop::take(&mut self.sampler));
			self.context.device.destroy_image_view(ManuallyDrop::take(&mut self.view));
			self.context.device.destroy_image(ManuallyDrop::take(&mut self.image));
			self.context.allocator.borrow_mut().dealloc(GfxMemoryDevice::wrap(&self.context.device), ManuallyDrop::take(&mut self.memory_block));
		}
	}
}