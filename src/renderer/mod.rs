//! The core rendering context structures

pub(crate) mod backend;

use std::convert::TryFrom;

use crate::data::{GpuPos, Color};
use crate::vertex::*;

use raw_window_handle::HasRawWindowHandle;

use gfx_hal::prelude::*;

#[macro_use]
mod alignment {
	#[repr(C)] // guarantee 'bytes' comes after '_align'
    pub struct AlignedAs<Align, Bytes: ?Sized> {
        pub _align: [Align; 0],
        pub bytes: Bytes,
    }

    macro_rules! include_bytes_align_as {
        ($align_ty:ty, $path:literal) => {
            {  // const block expression to encapsulate the static
                use $crate::renderer::alignment::AlignedAs;
                
                // this assignment is made possible by CoerceUnsized
                static ALIGNED: &AlignedAs::<$align_ty, [u8]> = &AlignedAs {
                    _align: [],
                    bytes: *include_bytes!($path),
                };
    
                &ALIGNED.bytes
            }
        };
    }
}

static COLOURED_VERT_SPV: &'static [u8] = include_bytes_align_as!(u32, "../spirv/coloured.vert.spv");
static COLOURED_FRAG_SPV: &'static [u8] = include_bytes_align_as!(u32, "../spirv/coloured.frag.spv");
static TEXTURED_VERT_SPV: &'static [u8] = include_bytes_align_as!(u32, "../spirv/textured.vert.spv");
static TEXTURED_FRAG_SPV: &'static [u8] = include_bytes_align_as!(u32, "../spirv/textured.frag.spv");

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
	surface: backend::Surface,
	gpu: gfx_hal::adapter::Gpu<backend::Backend>,
	swapchain_config: gfx_hal::window::SwapchainConfig,
	colour_graphics_pipeline: backend::GraphicsPipeline,
	texture_graphics_pipeline: backend::GraphicsPipeline,

	command_pool: backend::CommandPool,
	command_buffer: backend::CommandBuffer,

	render_pass: backend::RenderPass,
	
	vertex_buffer: backend::Buffer,
	vertex_memory: backend::Memory,
	index_buffer: backend::Buffer,
	index_memory: backend::Memory,

	submission_fence: backend::Fence,

	pub(crate) texture_descriptor_set_layout: backend::DescriptorSetLayout,
}

//TODO: Builder pattern, to allow for more configuration?
impl Renderer {
	/// Creates a new renderer, initialising the `gfx_hal` backend. This method assumes the raw window handle
	/// was created legitimately. Technically, that's my problem, but if you're not making your window properly, I'm not
	/// going to take responsibility for the resulting crash.
	/// 
	/// # Arguments
	/// * `window`: A valid window compatible with `raw_window_handle`.
	/// * `size`: The size of the window in pixels, in the order (width, height). For window implementations which
	///           differentiate between physical and logical size, this refers to the logical size
	pub fn new(window: &impl HasRawWindowHandle, (width, height): (u32, u32)) -> Renderer {
		//MARK: New Renderer

		//Note: Keep up-to-date.         X0.X4.X0_XX
		const POLYSTRIP_VERSION: u32 = 0x00_04_00_00;
		let instance = backend::Instance::create("polystrip", POLYSTRIP_VERSION).unwrap();
		let mut surface = unsafe { instance.create_surface(window).unwrap() };

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
				gfx_hal::Features::CORE_MASK
			).unwrap()		
		};
		
		let swapchain_config = gfx_hal::window::SwapchainConfig::new(width, height, gfx_hal::format::Format::Bgra8Srgb, 2);
		unsafe { surface.configure_swapchain(&gpu.device, swapchain_config.clone()).unwrap(); }

		let (command_pool, command_buffer) = unsafe {
			let mut command_pool = gpu.device.create_command_pool(gpu.queue_groups[0].family, gfx_hal::pool::CommandPoolCreateFlags::empty()).unwrap();
			let command_buffer = command_pool.allocate_one(gfx_hal::command::Level::Primary);
			(command_pool, command_buffer)
		};
		
		let vertex_buffer = unsafe { gpu.device.create_buffer(
			1024 * std::mem::size_of::<TextureVertex>() as u64,
			gfx_hal::buffer::Usage::VERTEX
		).unwrap() };

		let index_buffer = unsafe { gpu.device.create_buffer(
			1024 * std::mem::size_of::<u16>() as u64,
			gfx_hal::buffer::Usage::INDEX
		).unwrap() };

		let memory_types = adapter.physical_device.memory_properties().memory_types;

		let req = unsafe { gpu.device.get_buffer_requirements(&vertex_buffer) };
		let vertex_memory_type = memory_types.iter()
			.enumerate()
			.find(|(id, memory_type)| {
				req.type_mask & (1_u32 << id) != 0 &&
				memory_type.properties.contains(gfx_hal::memory::Properties::CPU_VISIBLE)
			})
			.map(|(id, _)| gfx_hal::MemoryTypeId(id))
			.unwrap();
		
		let vertex_memory = unsafe { gpu.device.allocate_memory( //TODO: Free memory on drop
			vertex_memory_type,
			1024 * std::mem::size_of::<TextureVertex>() as u64,
		)}.unwrap();
		
		let req = unsafe { gpu.device.get_buffer_requirements(&index_buffer) };
		let index_memory_type = memory_types.iter()
			.enumerate()
			.find(|(id, memory_type)| {
				req.type_mask & (1_u32 << id) != 0 &&
				memory_type.properties.contains(gfx_hal::memory::Properties::CPU_VISIBLE)
			})
			.map(|(id, _)| gfx_hal::MemoryTypeId(id))
			.unwrap();

		let index_memory = unsafe {gpu.device.allocate_memory(
			index_memory_type,
			1024 * std::mem::size_of::<u16>() as u64,
		)}.unwrap();

		unsafe {
			gpu.device.bind_buffer_memory(
				&vertex_memory,
				0,
				&mut vertex_buffer,
			).unwrap();
			gpu.device.bind_buffer_memory(
				&index_memory,
				0,
				&mut index_buffer,
			).unwrap();
		}

		let main_pass = unsafe { gpu.device.create_render_pass(
			&[gfx_hal::pass::Attachment {
				format: Some(swapchain_config.format),
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
						constants: std::borrow::Cow::Borrowed(&[]),
						data: std::borrow::Cow::Borrowed(&[]),
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
					binding: 0,
					ty: gfx_hal::pso::DescriptorType::Sampler,
					count: 1,
					stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
					immutable_samplers: false,
				}
			],
			&[]
		)}.unwrap();

		let texture_vs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(TEXTURED_VERT_SPV)) }.unwrap();
		let texture_fs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(TEXTURED_FRAG_SPV)) }.unwrap();

		let texture_graphics_pipeline_layout = unsafe { gpu.device.create_pipeline_layout(&[texture_descriptor_set_layout], &[]) }.unwrap();
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
						constants: std::borrow::Cow::Borrowed(&[]),
						data: std::borrow::Cow::Borrowed(&[]),
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

		let submission_fence = unsafe { gpu.device.create_fence(false) }.unwrap();

		Renderer {
			surface, gpu, swapchain_config, colour_graphics_pipeline, texture_graphics_pipeline,
			command_pool, command_buffer,
			render_pass: main_pass,
			vertex_buffer, index_buffer, vertex_memory, index_memory,
			submission_fence,
			texture_descriptor_set_layout,
		}
	}

	/// Returns the next `Frame`, which can be drawn to and will present on drop. This `Renderer` is borrowed mutably while the
	/// frame is alive. Any operations on this renderer must be done through the `Frame`, which implements `Deref<Target = Renderer>`.
	pub fn get_next_frame(&mut self) -> Frame<'_> {
		match unsafe { self.surface.acquire_image(1_000_000 /* 1 ms */) } {
			Ok((image, _)) => self.generate_frame(image),
			Err(gfx_hal::window::AcquireError::OutOfDate) => {
				unsafe { self.surface.configure_swapchain(&self.gpu.device, self.swapchain_config.clone()) }.unwrap();
				match unsafe { self.surface.acquire_image(0) } {
					Ok((image, _)) => self.generate_frame(image),
					Err(e) => panic!("{}", e),
				}
			},
			Err(e) => panic!("{}", e),
		}
	}

	fn generate_frame(&mut self, image: <backend::Surface as gfx_hal::window::PresentationSurface<backend::Backend>>::SwapchainImage) -> Frame<'_> {
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
		let framebuffer = unsafe { self.gpu.device.create_framebuffer(&self.render_pass, vec![image.borrow()], self.swapchain_config.extent.to_extent()) }.unwrap();

		Frame {
			swap_chain_frame: image,
			framebuffer,
			viewport,
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
		self.swapchain_config.extent.width = size.0;
		self.swapchain_config.extent.height = size.1;
		unsafe { self.surface.configure_swapchain(&self.gpu.device, self.swapchain_config.clone()) }.unwrap();
	}

	/// Gets the underlying `gfx_hal::Gpu` used internally to render.
	/// 
	/// The device is requested with no special features, the default limits and shader validation enabled.
	/// The device is opened with one 0.9 priority queue from one graphics-supporting queue family.
	pub fn device(&self) -> &gfx_hal::adapter::Gpu<backend::Backend> {
		&self.gpu
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

/// A frame to be drawn to. The frame gets presented on drop.
/// 
/// Since a `Frame` borrows the [`Renderer`](struct.Renderer.html) it was created for, any functions which would normally
/// be called on a `&Renderer` must be called on the `Frame`, which implements `Deref<Target = Renderer>`.
/// 
/// More methods are implemented in the [`FrameGeometryExt`](../geometry/trait.FrameGeometryExt.html) trait.
pub struct Frame<'a> {
	renderer: &'a mut Renderer,
	swap_chain_frame: <backend::Surface as gfx_hal::window::PresentationSurface<backend::Backend>>::SwapchainImage,
	framebuffer: backend::Framebuffer,
	viewport: gfx_hal::pso::Viewport,
}

//MARK: Frame API
impl<'a> Frame<'a> {
	/// Draws a [`ColoredShape`](../vertex/struct.ColoredShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	pub fn draw_colored(&mut self, shape: ColoredShape) {
		if shape.vertices.len() > 1024 {
			panic!("Maximum size of shape is 1024 vertices, found {}", shape.vertices.len());
		}

		let index_data = shape.indices.iter().flatten().copied().collect::<Vec<_>>();
		if index_data.len() > 1024 {
			panic!("Maximum size of shape is 1024 indices, found {}", index_data.len());
		}

		unsafe {
			self.command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
				
			self.command_buffer.set_viewports(0, &[self.viewport.clone()]);
			self.command_buffer.set_scissors(0, &[self.viewport.rect]);

			self.command_buffer.begin_render_pass(&self.render_pass, &self.framebuffer, self.viewport.rect, &[], gfx_hal::command::SubpassContents::Inline);

			let vertex_buffer = self.gpu.device.map_memory(&self.vertex_memory, gfx_hal::memory::Segment::ALL).unwrap();
			let vertices = bytemuck::cast_slice(&shape.vertices);
			std::ptr::copy_nonoverlapping(vertices.as_ptr(), vertex_buffer, vertices.len());
			self.gpu.device.unmap_memory(&self.vertex_memory);

			let index_buffer = self.gpu.device.map_memory(&self.index_memory, gfx_hal::memory::Segment::ALL).unwrap();
			let indices = bytemuck::cast_slice(&index_data);
			std::ptr::copy_nonoverlapping(indices.as_ptr(), index_buffer, indices.len());
			self.gpu.device.unmap_memory(&self.index_memory);

			self.command_buffer.bind_graphics_pipeline(&self.colour_graphics_pipeline);
			self.command_buffer.bind_vertex_buffers(0, vec![(self.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			self.command_buffer.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
				buffer: &self.index_buffer,
				range: gfx_hal::buffer::SubRange::WHOLE,
				index_type: gfx_hal::IndexType::U16,
			});
			self.command_buffer.draw_indexed(0..index_data.len() as u32, 0, 0..1);

			self.command_buffer.end_render_pass();
			self.command_buffer.finish();

			match self.gpu.device.wait_for_fence(&self.submission_fence, 1_000_000 /* 1 ms */) {
				Ok(true) => {},
				Ok(false) => { panic!("Render pass took >1ms"); }
				Err(e) => { panic!("{}", e); }
			}

			self.gpu.queue_groups[0].queues[0].submit::<_, _, &backend::Semaphore, _, _>(gfx_hal::queue::Submission {
				command_buffers: &[self.command_buffer],
				wait_semaphores: None,
				signal_semaphores: None,
			}, Some(&self.submission_fence));
		}
	}

	/// Draws a [`TexturedShape`](../vertex/struct.TexturedShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	/// 
	/// # Arguments
	/// * `shape`: The `TexturedShape` to be rendered. 
	/// * `texture`: The `Texture` to be drawn to the geometry of the shape.
	pub fn draw_textured(&mut self, shape: TexturedShape, texture: &'a crate::texture::Texture) {
		if shape.vertices.len() > 1024 {
			panic!("Maximum size of shape is 1024 vertices, found {}", shape.vertices.len());
		}

		let index_data = shape.indices.iter().flatten().copied().collect::<Vec<_>>();
		if index_data.len() > 1024 {
			panic!("Maximum size of shape is 1024 indices, found {}", index_data.len());
		}

		unsafe {
			self.command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
				
			self.command_buffer.set_viewports(0, &[self.viewport.clone()]);
			self.command_buffer.set_scissors(0, &[self.viewport.rect]);

			self.command_buffer.begin_render_pass(&self.render_pass, &self.framebuffer, self.viewport.rect, &[], gfx_hal::command::SubpassContents::Inline);

			let vertex_buffer = self.gpu.device.map_memory(&self.vertex_memory, gfx_hal::memory::Segment::ALL).unwrap();
			let vertices = bytemuck::cast_slice(&shape.vertices);
			std::ptr::copy_nonoverlapping(vertices.as_ptr(), vertex_buffer, vertices.len());
			self.gpu.device.unmap_memory(&self.vertex_memory);

			let index_buffer = self.gpu.device.map_memory(&self.index_memory, gfx_hal::memory::Segment::ALL).unwrap();
			let indices = bytemuck::cast_slice(&index_data);
			std::ptr::copy_nonoverlapping(indices.as_ptr(), index_buffer, indices.len());
			self.gpu.device.unmap_memory(&self.index_memory);

			self.command_buffer.bind_graphics_pipeline(&self.texture_graphics_pipeline);
			//TODO: Bind descriptor sets
			self.command_buffer.bind_vertex_buffers(0, vec![(self.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			self.command_buffer.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
				buffer: &self.index_buffer,
				range: gfx_hal::buffer::SubRange::WHOLE,
				index_type: gfx_hal::IndexType::U16,
			});
			self.command_buffer.draw_indexed(0..index_data.len() as u32, 0, 0..1);

			self.command_buffer.end_render_pass();
			self.command_buffer.finish();

			match self.gpu.device.wait_for_fence(&self.submission_fence, 1_000_000 /* 1 ms */) {
				Ok(true) => {},
				Ok(false) => { panic!("Render pass took >1ms"); }
				Err(e) => { panic!("{}", e); }
			}

			self.gpu.queue_groups[0].queues[0].submit::<_, _, &backend::Semaphore, _, _>(gfx_hal::queue::Submission {
				command_buffers: &[self.command_buffer],
				wait_semaphores: None,
				signal_semaphores: None,
			}, None);
		}
	}

	/// Draws a [`ShapeSet`](../vertex/enum.ShapeSet.html). All shapes in the set will be drawn in front of shapes drawn before
	/// the set. The render order of shapes in the set is unspecified.
	pub fn draw_shape_set(&mut self, set: ShapeSet<'a>) {
		let index_count;
		match set {
			ShapeSet::Colored(shapes) => {
				//TODO: Merge these two iterations into one which produces two vectors?
				let vertex_data = shapes.iter().flat_map(|shape| shape.vertices.iter()).copied().collect::<Vec<_>>();
				let mut index_offset: u16 = 0;
				let index_data = shapes.iter().flat_map(|shape| {
					let indices = shape.indices.iter()
						.flatten()
						.map(|&index| index + index_offset)
						.collect::<Vec<_>>();
					index_offset += u16::try_from(shape.vertices.len()).unwrap();
					indices
				}).collect::<Vec<u16>>();
				index_count = index_data.len();

				if vertex_data.len() > 1024 {
					panic!("Maximum size of shape is 1024 vertices, found {}", index_data.len());
				}
		
				if index_data.len() > 1024 {
					panic!("Maximum size of shape is 1024 indices, found {}", index_data.len());
				}
				unsafe {
					let vertex_buffer = self.gpu.device.map_memory(&self.vertex_memory, gfx_hal::memory::Segment::ALL).unwrap();
					let vertices = bytemuck::cast_slice(&vertex_data);
					std::ptr::copy_nonoverlapping(vertices.as_ptr(), vertex_buffer, vertices.len());
					self.gpu.device.unmap_memory(&self.vertex_memory);

					let index_buffer = self.gpu.device.map_memory(&self.index_memory, gfx_hal::memory::Segment::ALL).unwrap();
					let indices = bytemuck::cast_slice(&index_data);
					std::ptr::copy_nonoverlapping(indices.as_ptr(), index_buffer, indices.len());
					self.gpu.device.unmap_memory(&self.index_memory);

					self.command_buffer.bind_graphics_pipeline(&self.colour_graphics_pipeline);
				}
			},
			ShapeSet::Textured(shapes, texture) => {
				// ! Duplicated code from above branch
				let vertex_data = shapes.iter().flat_map(|shape| shape.vertices.iter()).copied().collect::<Vec<_>>();
				let mut index_offset: u16 = 0;
				let index_data = shapes.iter().flat_map(|shape| {
					let indices = shape.indices.iter()
						.flatten()
						.map(|&index| index + index_offset)
						.collect::<Vec<_>>();
					index_offset += u16::try_from(shape.vertices.len()).unwrap();
					indices
				}).collect::<Vec<u16>>();
				index_count = index_data.len();

				if vertex_data.len() > 1024 {
					panic!("Maximum size of shape is 1024 vertices, found {}", index_data.len());
				}
		
				if index_data.len() > 1024 {
					panic!("Maximum size of shape is 1024 indices, found {}", index_data.len());
				}
				unsafe {
					let vertex_buffer = self.gpu.device.map_memory(&self.vertex_memory, gfx_hal::memory::Segment::ALL).unwrap();
					let vertices = bytemuck::cast_slice(&vertex_data);
					std::ptr::copy_nonoverlapping(vertices.as_ptr(), vertex_buffer, vertices.len());
					self.gpu.device.unmap_memory(&self.vertex_memory);

					let index_buffer = self.gpu.device.map_memory(&self.index_memory, gfx_hal::memory::Segment::ALL).unwrap();
					let indices = bytemuck::cast_slice(&index_data);
					std::ptr::copy_nonoverlapping(indices.as_ptr(), index_buffer, indices.len());
					self.gpu.device.unmap_memory(&self.index_memory);
				// ! End of duplicated code

					self.command_buffer.bind_graphics_pipeline(&self.texture_graphics_pipeline);
					// ! Bind descriptor set
				}
			}
		}

		self.command_buffer.bind_vertex_buffers(0, vec![(self.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
		self.command_buffer.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
			buffer: &self.index_buffer,
			range: gfx_hal::buffer::SubRange::WHOLE,
			index_type: gfx_hal::IndexType::U16,
		});
		self.command_buffer.draw_indexed(0..index_count as u32, 0, 0..1);

		match self.gpu.device.wait_for_fence(&self.submission_fence, 1_000_000 /* 1 ms */) {
			Ok(true) => {},
			Ok(false) => { panic!("Render pass took >1ms"); }
			Err(e) => { panic!("{}", e); }
		}

		self.gpu.queue_groups[0].queues[0].submit::<_, _, &backend::Semaphore, _, _>(gfx_hal::queue::Submission {
			command_buffers: &[self.command_buffer],
			wait_semaphores: None,
			signal_semaphores: None,
		}, None);
	}

	/// Clears the entire frame with the specified color, setting every pixel to its value.
	/// 
	/// Note: The sRGB conversion in this function uses a gamma of 2.0
	pub fn clear(&mut self, color: Color) {
		todo!();
		// let mut encoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
		// 	label: Some("polystrip_render_encoder"),
		// });

		// let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
		// 	color_attachments: &[
		// 		wgpu::RenderPassColorAttachmentDescriptor {
		// 			attachment: &self.swap_chain_frame.output.view,
		// 			resolve_target: None,
		// 			ops: wgpu::Operations {
		// 				load: wgpu::LoadOp::Clear(wgpu::Color {
		// 					r: (color.r as f64).powi(2) / 65_025.0,
		// 					g: (color.g as f64).powi(2) / 65_025.0,
		// 					b: (color.b as f64).powi(2) / 65_025.0,
		// 					a: color.a as f64 / 255.0,
		// 				}),
		// 				store: true,
		// 			}
		// 		}
		// 	],
		// 	depth_stencil_attachment: None,
		// });

		// std::mem::drop(render_pass);

		// self.renderer.queue.submit(std::iter::once(encoder.finish()));
	}

	/// Gets the internal `SwapChainFrame` for use in custom rendering.
	pub fn swap_chain_frame(&self) -> ! {
		todo!()
	}
}

impl<'a> Drop for Frame<'a> {
	fn drop(&mut self) {
		if !std::thread::panicking() {
			//TODO
		}
	}
}

impl<'a> std::ops::Deref for Frame<'a> {
	type Target = Renderer;

	fn deref(&self) -> &Renderer {
		&self.renderer
	}
}